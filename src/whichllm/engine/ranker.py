"""Model ranking: score and select the best models for the user's hardware."""

from __future__ import annotations

import math
import re

from whichllm.constants import QUANT_PREFERENCE_ORDER
from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.engine.quantization import effective_quant_type, quant_quality_penalty
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.benchmark import (
    BenchmarkEvidence,
    build_line_bucket_index,
    build_score_index,
    lookup_benchmark_evidence,
)
from whichllm.models.types import GGUFVariant, ModelInfo


def _family_selection_key(
    result: CompatibilityResult,
    require_direct_top: bool,
) -> tuple[float, float, float]:
    """Family-level selection key."""
    fit_priority = {
        "full_gpu": 2.0,
        "partial_offload": 1.0,
        "cpu_only": 0.0,
    }.get(result.fit_type, 0.0)
    if require_direct_top:
        return (
            fit_priority,
            1.0 if result.benchmark_status == "direct" else 0.0,
            result.quality_score,
        )
    return (fit_priority, 0.0, result.quality_score)


def _iter_candidate_variants(
    model: ModelInfo,
    quant_filter: str | None = None,
) -> list[GGUFVariant | None]:
    """Build candidate variants to evaluate for a model."""
    quant_filter_upper = quant_filter.upper() if quant_filter else None

    if not model.gguf_variants:
        quant_type = effective_quant_type(model, None)
        if quant_filter_upper and quant_type != quant_filter_upper:
            return []
        return [None]

    # Filter by quant type if specified
    candidates: list[GGUFVariant] = model.gguf_variants
    if quant_filter_upper:
        candidates = [v for v in candidates if v.quant_type.upper() == quant_filter_upper]
        if not candidates:
            return []
    else:
        # Exclude extreme quantizations unless explicitly requested
        _EXTREME_QUANTS = {"Q2_K", "IQ2_XXS", "IQ3_XXS"}
        filtered = [v for v in candidates if v.quant_type.upper() not in _EXTREME_QUANTS]
        if filtered:
            candidates = filtered

    # Sort by preference order
    def variant_sort_key(v: GGUFVariant) -> int:
        try:
            return QUANT_PREFERENCE_ORDER.index(v.quant_type.upper())
        except ValueError:
            return len(QUANT_PREFERENCE_ORDER)

    candidates = sorted(candidates, key=variant_sort_key)

    return list(candidates)


_OFFICIAL_ORGS = frozenset({
    "Qwen", "meta-llama", "google", "mistralai", "deepseek-ai",
    "microsoft", "nvidia", "01-ai", "tiiuae", "apple",
    "CohereForAI", "bigcode",
})

# Trusted GGUF converters — format converters that don't change model quality
_TRUSTED_CONVERTERS = frozenset({
    "bartowski", "lmstudio-community", "QuantFactory", "unsloth",
    "ggml-org", "Mungert",
})

# Known repackagers — typically reupload others' models without added value
_REPACKAGER_ORGS = frozenset({
    "MaziyarPanahi", "TheBloke", "SanctumAI", "solidrust",
    "mradermacher",
})


def _detect_specializations(model_id: str) -> set[str]:
    """モデルIDから用途特化タグを検出する。"""
    lower = model_id.lower()
    tags: set[str] = set()
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.add("coding")
    if re.search(r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)", lower):
        tags.add("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.add("math")
    return tags


def _matches_profile(model: ModelInfo, task_profile: str) -> bool:
    """指定プロファイルにモデルが合致するか判定する。"""
    profile = task_profile.lower()
    tags = _detect_specializations(model.id)
    if profile == "any":
        return True
    if profile == "general":
        return len(tags) == 0
    return profile in tags


def _effective_params_b(model: ModelInfo) -> float:
    """Return effective parameter size in billions."""
    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active / 1e9
    return model.parameter_count / 1e9


def _passes_evidence_filter(source: str, evidence_filter: str) -> bool:
    """判定根拠フィルタに合致するかを返す。"""
    mode = evidence_filter.lower()
    if mode == "strict":
        return source == "direct"
    if mode == "base":
        return source in {"direct", "variant", "base_model"}
    return True


def _is_gguf_only_backend(hardware: HardwareInfo) -> bool:
    """実行基盤の都合でGGUFのみを許可すべきか判定する。"""
    # Apple Silicon(macOS/Metal)とCPU-onlyは、実運用の安定性を優先してGGUFに限定する。
    if hardware.os == "darwin":
        return True
    if not hardware.gpus:
        return True

    # Linux + NVIDIA (CUDA) は AWQ/GPTQ 含む非GGUFも許可する。
    has_linux_nvidia = hardware.os == "linux" and any(g.vendor == "nvidia" for g in hardware.gpus)
    return not has_linux_nvidia


def _compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
    fit_type: str,
    family_downloads: int = 0,
    family_likes: int = 0,
    benchmark_avg: float | None = None,
    benchmark_is_direct: bool = False,
) -> float:
    """Compute a quality score (0-100) for ranking.

    Factors:
    - Benchmark score (direct or line-estimated)
    - Model size (log scale base)
    - Quantization penalty
    - Fit type penalty (partial offload / CPU-only heavily penalized)
    - Speed bonus (practical usability)
    - Popularity (downloads/likes as tiebreaker)
    - Official repo bonus
    """
    params_b = model.parameter_count / 1e9
    if model.is_moe and model.parameter_count_active:
        effective_b = model.parameter_count_active / 1e9
    else:
        effective_b = params_b

    if effective_b <= 0:
        return 0.0

    # 品質主軸: ベンチマークを最重視し、サイズは補助にする
    size_score = 4.2 * math.log2(max(effective_b, 0.5)) + 9
    size_score = min(size_score, 20)

    benchmark_score = 0.0
    has_benchmark = benchmark_avg is not None and benchmark_avg > 0
    if has_benchmark:
        raw = min(100.0, benchmark_avg)
        if benchmark_is_direct:
            benchmark_score = raw * 0.62  # 最大62点
        else:
            benchmark_score = raw * 0.40  # 推定値は大きく割り引く

    # Quantization penalty
    quant_penalty = quant_quality_penalty(model, variant)
    quality_core = (benchmark_score + size_score) * (1 - quant_penalty)

    # 根拠の弱いベンチはさらに下げる
    if not has_benchmark:
        quality_core *= 0.55
    elif not benchmark_is_direct:
        quality_core *= 0.78

    # 実行形態ペナルティ
    if fit_type == "partial_offload":
        quality_core *= 0.72
    elif fit_type == "cpu_only":
        quality_core *= 0.50

    # 速度は順位を決める主因ではなく「使い物になるか」の補正にする
    speed_score = 0.0
    required_speed = 8.0 if fit_type == "full_gpu" else (4.0 if fit_type == "partial_offload" else 1.5)
    if tok_per_sec > 0:
        if tok_per_sec < required_speed:
            # 閾値未満は強く減点
            speed_score = -8.0 * (1 - (tok_per_sec / required_speed))
        else:
            # 閾値超えは緩やかに加点
            speed_score = min(8.0, math.log2(tok_per_sec / required_speed + 1.0) * 3.2)
    else:
        speed_score = -8.0

    # 人気度は同点近傍のタイブレーク専用
    downloads = max(model.downloads, family_downloads)
    likes = max(model.likes, family_likes)
    pop_score_raw = 0.0
    if downloads > 0:
        pop_score_raw += min(1.0, math.log10(max(downloads, 1)) / 6 * 1.0)
    if likes > 0:
        pop_score_raw += min(1.0, math.log10(max(likes, 1)) / 4 * 1.0)

    if benchmark_is_direct:
        pop_weight = 0.0
    elif has_benchmark:
        pop_weight = 0.2
    else:
        pop_weight = 0.6
    pop_score = pop_score_raw * pop_weight

    # 供給元信頼は軽い補正に留める
    source_bonus_raw = 0.0
    org = model.id.split("/")[0] if "/" in model.id else ""
    if org in _OFFICIAL_ORGS:
        source_bonus_raw = 5.0
    elif org in _REPACKAGER_ORGS:
        source_bonus_raw = -5.0
    elif model.base_model:
        base_org = model.base_model.split("/")[0] if "/" in model.base_model else ""
        if base_org in _OFFICIAL_ORGS:
            if org in _TRUSTED_CONVERTERS:
                # Known GGUF converter — same quality as official
                source_bonus_raw = 5.0
            else:
                # Unknown fine-tune/fork — no bonus (quality unproven)
                source_bonus_raw = 0.0

    if benchmark_is_direct:
        source_weight = 0.2
    elif has_benchmark:
        source_weight = 0.4
    else:
        source_weight = 0.6
    source_bonus = source_bonus_raw * source_weight

    return max(0.0, min(100.0, quality_core + speed_score + pop_score + source_bonus))


def rank_models(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    context_length: int = 4096,
    top_n: int = 10,
    quant_filter: str | None = None,
    min_speed: float | None = None,
    benchmark_scores: dict[str, float] | None = None,
    task_profile: str = "general",
    require_direct_top: bool = True,
    min_params_b: float | None = None,
    evidence_filter: str = "any",
) -> list[CompatibilityResult]:
    """Rank models by quality for the given hardware. Returns top N results."""
    results: list[CompatibilityResult] = []
    gguf_only_backend = _is_gguf_only_backend(hardware)

    # Pre-compute max downloads/likes per family so GGUF converters
    # inherit popularity from the official base model
    family_max_downloads: dict[str, int] = {}
    family_max_likes: dict[str, int] = {}
    for m in models:
        fid = m.family_id
        family_max_downloads[fid] = max(family_max_downloads.get(fid, 0), m.downloads)
        family_max_likes[fid] = max(family_max_likes.get(fid, 0), m.likes)

    # Deduplicate by family: pick best variant per family
    seen_families: set[str] = set()

    # Sort models by downloads (popular first) to process best candidates first
    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)

    # Build benchmark indices once (case-insensitive + model line)
    if benchmark_scores:
        bench_ci_index, bench_line_index = build_score_index(benchmark_scores)
        bench_line_buckets = build_line_bucket_index(benchmark_scores)
    else:
        bench_ci_index, bench_line_index = {}, {}
        bench_line_buckets = {}

    best_gpu = None
    for gpu in hardware.gpus:
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu

    for model in sorted_models:
        if not _matches_profile(model, task_profile):
            continue
        if min_params_b is not None and _effective_params_b(model) < min_params_b:
            continue

        candidates = _iter_candidate_variants(model, quant_filter)
        if not candidates:
            continue

        fid = model.family_id
        bench_evidence = BenchmarkEvidence(score=None, confidence=0.0, source="none")
        if benchmark_scores:
            bench_evidence = lookup_benchmark_evidence(
                model.id,
                model.base_model,
                benchmark_scores,
                ci_index=bench_ci_index,
                line_index=bench_line_index,
                line_bucket_index=bench_line_buckets,
            )
        if not _passes_evidence_filter(bench_evidence.source, evidence_filter):
            continue

        # 各variantを評価し、そのモデルで最もスコアが高いものを採用する
        best_for_model: CompatibilityResult | None = None
        for variant in candidates:
            if gguf_only_backend and variant is None:
                continue
            compat = check_compatibility(model, variant, hardware, context_length)
            if not compat.can_run:
                continue

            tok_per_sec = estimate_tok_per_sec(model, variant, best_gpu, compat.fit_type)
            if min_speed is not None and tok_per_sec < min_speed:
                continue

            bench_avg = None
            bench_is_direct = False
            if bench_evidence.score is not None:
                bench_is_direct = bench_evidence.source == "direct"
                if bench_is_direct:
                    bench_avg = bench_evidence.score
                else:
                    # 推定根拠が弱いほどベンチ寄与を下げる（line継承の過大評価を抑制）
                    confidence = max(0.0, min(1.0, bench_evidence.confidence))
                    bench_avg = bench_evidence.score * (0.75 + 0.25 * confidence)

            compat.estimated_tok_per_sec = tok_per_sec
            compat.quality_score = _compute_quality_score(
                model,
                variant,
                tok_per_sec,
                compat.fit_type,
                family_downloads=family_max_downloads.get(fid, 0),
                family_likes=family_max_likes.get(fid, 0),
                benchmark_avg=bench_avg,
                benchmark_is_direct=bench_is_direct,
            )
            if bench_evidence.score is not None:
                compat.benchmark_status = "direct" if bench_evidence.source == "direct" else "estimated"
            else:
                compat.benchmark_status = "none"

            if best_for_model is None or compat.quality_score > best_for_model.quality_score:
                best_for_model = compat

        if best_for_model is None:
            continue

        # Deduplicate by family: keep the one with highest quality score
        family_key = model.family_id
        if family_key in seen_families:
            # Check if this is better than existing
            existing = next(
                (r for r in results if r.model.family_id == family_key), None
            )
            if existing and _family_selection_key(
                best_for_model,
                require_direct_top,
            ) > _family_selection_key(existing, require_direct_top):
                results.remove(existing)
                results.append(best_for_model)
            continue

        seen_families.add(family_key)
        results.append(best_for_model)

    if require_direct_top:
        results.sort(
            key=lambda r: _family_selection_key(r, require_direct_top),
            reverse=True,
        )
    else:
        results.sort(key=lambda r: _family_selection_key(r, require_direct_top), reverse=True)

    return results[:top_n]
