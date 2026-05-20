"""Model ranking: score and select the best models for the user's hardware."""

from __future__ import annotations

import math
import re

from whichllm.constants import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
    QUANT_BYTES_PER_WEIGHT,
    QUANT_PREFERENCE_ORDER,
)
from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.performance import estimate_speed_uncertainty, estimate_tok_per_sec
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

# Pre-compile lineage regex tables once at import time.
_LINEAGE_REGEX: dict[str, list[tuple[re.Pattern[str], int]]] = {
    family: [(re.compile(pat), idx) for pat, idx in entries]
    for family, entries in MODEL_LINEAGE_VERSIONS.items()
}
_LINEAGE_FAMILY_MAX: dict[str, int] = {
    family: max(idx for _, idx in entries) for family, entries in _LINEAGE_REGEX.items()
}


def _family_selection_key(
    result: CompatibilityResult,
    require_direct_top: bool,
) -> tuple[float]:
    """Family-level selection key — single composite score.

    Combines quality, fit type, and evidence tier into one number so the
    sort is fully transitive and edge cases resolve sensibly:

    - ``fit_bonus`` (+15 / 0 / -15) is large enough that "estimated,
      full-GPU" still beats "direct, partial-offload" of comparable
      quality (users on small VRAM prefer the responsive option),
      but small enough that a quality-17 Q1_0 full-GPU model loses to
      a quality-57 partial-offload 27B model
    - ``direct_bonus`` (+5) gives independent leaderboard evidence a
      small edge at the same fit; cannot overturn a 6+ point quality gap
    """
    fit_bonus = {
        "full_gpu": 15.0,
        "partial_offload": 0.0,
        "cpu_only": -15.0,
    }.get(result.fit_type, -15.0)
    if require_direct_top and result.benchmark_status == "direct":
        direct_bonus = 5.0
    else:
        direct_bonus = 0.0
    return (result.quality_score + fit_bonus + direct_bonus,)


# Per-source benchmark weight applied to the raw 0-100 score before it is
# combined with size, quant penalty, etc. The widest gap is between "direct"
# (independent leaderboard) and "self_reported" (uploader card claim).
_SOURCE_WEIGHTS: dict[str, float] = {
    "direct": 0.62,
    "base_model": 0.55,
    "variant": 0.50,
    "line_interp": 0.40,
    "self_reported": 0.30,
    "none": 0.0,
}


_SYNTHETIC_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
_PREQUANTIZED_REPO_RE = re.compile(
    r"-(awq|gptq|bnb|fp8|fp16|bf16|nvfp4|int4|int8|4bit|8bit|gguf)$",
    re.IGNORECASE,
)


def _synthesize_variants_for_official_repo(
    model: ModelInfo, quant_filter_upper: str | None
) -> list[GGUFVariant]:
    """Return synthetic GGUF variants for popular safetensors-only repos.

    HuggingFace doesn't always index GGUF siblings for an official model
    (e.g. ``Qwen/Qwen3.6-27B`` ships only safetensors), but bartowski /
    lmstudio-community / QuantFactory invariably publish Q4_K_M and Q8_0
    conversions within a day of release. Without synthetic variants, we'd
    score these models at BF16 file sizes (~2x larger than realistic), which
    forces a partial_offload penalty on otherwise-runnable mid-size models.

    Skips repos that already advertise a specific quantization in their name
    (``...-AWQ``, ``...-GPTQ``, ``...-FP8`` etc.) — those are non-GGUF formats
    and synthesizing a Q4_K_M alternative would misrepresent what the repo
    actually contains.
    """
    org = model.id.split("/", 1)[0] if "/" in model.id else ""
    if org not in _OFFICIAL_ORGS:
        return []
    if _PREQUANTIZED_REPO_RE.search(model.id):
        return []
    out: list[GGUFVariant] = []
    for quant in _SYNTHETIC_QUANTS:
        if quant_filter_upper and quant != quant_filter_upper:
            continue
        bpw = QUANT_BYTES_PER_WEIGHT.get(quant, 0.5625)
        out.append(
            GGUFVariant(
                filename=f"{model.name}.{quant}.gguf",
                quant_type=quant,
                file_size_bytes=int(model.parameter_count * bpw),
            )
        )
    return out


def _iter_candidate_variants(
    model: ModelInfo,
    quant_filter: str | None = None,
) -> list[GGUFVariant | None]:
    """Build candidate variants to evaluate for a model."""
    quant_filter_upper = quant_filter.upper() if quant_filter else None

    if not model.gguf_variants:
        synthetic = _synthesize_variants_for_official_repo(model, quant_filter_upper)
        if synthetic:
            return list(synthetic)
        quant_type = effective_quant_type(model, None)
        if quant_filter_upper and quant_type != quant_filter_upper:
            return []
        return [None]

    # Filter by quant type if specified
    candidates: list[GGUFVariant] = model.gguf_variants
    if quant_filter_upper:
        candidates = [
            v for v in candidates if v.quant_type.upper() == quant_filter_upper
        ]
        if not candidates:
            return []
    else:
        # Sub-3-bit GGUFs lose 25-60% of model quality and rarely produce
        # a meaningfully better candidate than a smaller model at Q4_K_M.
        # Exclude them unless explicitly requested via --quant.
        _EXTREME_QUANTS = {
            "Q2_K",
            "Q2_0",
            "Q1_0",
            "TQ2_0",
            "TQ1_0",
            "IQ3_XXS",
            "IQ2_XXS",
            "IQ2_S",
            "IQ2_M",
            "IQ1_M",
            "IQ1_S",
        }
        filtered = [
            v for v in candidates if v.quant_type.upper() not in _EXTREME_QUANTS
        ]
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


_OFFICIAL_ORGS = frozenset(
    {
        "Qwen",
        "meta-llama",
        "google",
        "mistralai",
        "deepseek-ai",
        "microsoft",
        "nvidia",
        "01-ai",
        "tiiuae",
        "apple",
        "CohereForAI",
        "bigcode",
        # 2025+ frontier open-weights labs that publish safetensors-only
        # repos which the community immediately converts to GGUF.
        "openai",
        "zai-org",
        "moonshotai",
        "MiniMaxAI",
        "XiaomiMiMo",
        "allenai",
        "ibm-granite",
        "stepfun-ai",
    }
)

# Trusted GGUF converters — format converters that don't change model quality
_TRUSTED_CONVERTERS = frozenset(
    {
        "bartowski",
        "lmstudio-community",
        "QuantFactory",
        "unsloth",
        "ggml-org",
        "Mungert",
    }
)

# Known repackagers — typically reupload others' models without added value
_REPACKAGER_ORGS = frozenset(
    {
        "MaziyarPanahi",
        "TheBloke",
        "SanctumAI",
        "solidrust",
        "mradermacher",
    }
)

# Orgs whose repositories ship CI fixtures, deprecated research artifacts, or
# debug binaries that are not viable production LLMs. Exclude them outright so
# they cannot occupy ranking slots regardless of download counts.
_EXCLUDED_ORGS = frozenset(
    {
        "openai-community",  # gpt2 family, 2019 research
        "distilbert",  # distilgpt2 etc.
        "facebook",  # opt-125m research scaffolds
        "EleutherAI",  # pythia/gpt-neo research
        "trl-internal-testing",  # TRL CI fixtures
        "hmellor",  # random tiny test models
        "HuggingFaceH4",  # often staging / fixtures
        "transformersbook",
        "togethercomputer",  # mostly inference endpoints, no GGUFs
    }
)

# Substring patterns in *names* that strongly suggest non-production usage.
_EXCLUDED_NAME_PATTERNS = (
    "tiny-",
    "-tiny",
    "tiny_",
    "_tiny",
    "test-only",
    "debug-",
    "playground",
    "-fixture",
    "for-testing",
    "tiny-random",
    "ci-",
)

# Naming patterns that indicate a fine-tune / merge / "uncensoring" derivative
# of a real base model. These derivatives inherit the base model's benchmark
# score via line_interp, but the derivative itself is rarely benchmarked
# independently and frequently degrades quality. Apply a soft score penalty
# rather than full exclusion so they can still surface when nothing better is
# available.
_DUBIOUS_DERIVATIVE_PATTERNS = (
    "heretic",
    "abliterat",
    "uncensored",
    "obliterat",
    "abliter",
    "horror",
    "erotic",
    "nsfw",
    "rp-",
    "-rp",
    "roleplay",
    "darkidol",
    "darkforest",
    "tiefigh",
    "smaug",
    "personalityengine",
    "lexi",
    "violence",
    "violet",
    "schizo",
    "dark-",
    "twilight",
    "celeste",
    "midnight-rose",
    "moistral",
    "stheno",
    "fimbulvetr",
    "wizard-vicuna",
    "kunoichi",
)


def _derivative_name_penalty(model_id: str) -> float:
    """Return a score penalty (in raw quality points) for fine-tune /
    "uncensored" / merge derivatives that ride on a real base model's
    benchmark line. The penalty is gentle (≤ 12pt) so a derivative can
    still win when its size class has no better option.
    """
    if not model_id:
        return 0.0
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in _DUBIOUS_DERIVATIVE_PATTERNS:
        if pat in name:
            return -10.0
    return 0.0


def _is_excluded_model(model_id: str) -> bool:
    """Return True for CI/research/fixture models that should never rank."""
    if not model_id:
        return True
    org = model_id.split("/", 1)[0] if "/" in model_id else ""
    if org in _EXCLUDED_ORGS:
        return True
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in _EXCLUDED_NAME_PATTERNS:
        if pat in name:
            return True
    return False


def _generation_bonus(model_id: str) -> float:
    """Return a small additive bonus reflecting how new a model's
    generation is within its family. The newest version of each
    recognized family gets +MODEL_GENERATION_BONUS_MAX. Older
    versions get a smaller bonus (or a small penalty for the
    legacy generation). Unknown families return 0.

    This is purely an additive correction to the quality score
    and is small enough that strong benchmark evidence will still
    dominate.
    """
    if not model_id:
        return 0.0
    lower = model_id.lower()
    best_bonus = 0.0
    for family, patterns in _LINEAGE_REGEX.items():
        for regex, idx in patterns:
            if regex.search(lower):
                top = _LINEAGE_FAMILY_MAX[family]
                if top <= 1:
                    contribution = 0.0
                else:
                    # Map oldest -> -PENALTY_MAX, newest -> +BONUS_MAX.
                    norm = (idx - 1) / (top - 1)  # 0 .. 1
                    span = MODEL_GENERATION_BONUS_MAX + MODEL_GENERATION_PENALTY_MAX
                    contribution = norm * span - MODEL_GENERATION_PENALTY_MAX
                if abs(contribution) > abs(best_bonus):
                    best_bonus = contribution
                break  # first match wins for this family
    return best_bonus


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


def _knowledge_capacity_b(model: ModelInfo) -> float:
    """Return the knowledge capacity in billions for size filtering.

    For dense models this is the parameter count. For MoE models, total
    parameters (all expert weights live in VRAM and contribute to the
    knowledge encoded in the model) is the right yardstick — ``min_params``
    is asking "how much does this model know?" not "how much does it
    compute per token". Using *active* params here was the bug that hid
    Qwen3-Next-80B-A3B from the H100 ranking — its 3B active was below the
    12B auto-floor for 30GB+ GPUs even though its 80B total clearly fits.
    """
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
    has_linux_nvidia = hardware.os == "linux" and any(
        g.vendor == "nvidia" for g in hardware.gpus
    )
    return not has_linux_nvidia


def _compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
    fit_type: str,
    family_downloads: int = 0,
    family_likes: int = 0,
    benchmark_avg: float | None = None,
    benchmark_source: str = "none",
) -> float:
    """Compute a quality score (0-100) for ranking.

    Factors:
    - Benchmark score weighted by source tier
    - Model size (log scale)
    - Quantization penalty
    - Fit type penalty (partial offload / CPU-only heavily penalized)
    - Speed bonus / penalty (practical usability)
    - Popularity (downloads/likes) as soft tie-breaker
    - Official org bonus (vs known repackagers)
    - Generation-lineage bonus (newest family member > legacy generation)
    """
    params_b = model.parameter_count / 1e9
    if model.is_moe and model.parameter_count_active:
        effective_b = model.parameter_count_active / 1e9
    else:
        effective_b = params_b

    if effective_b <= 0:
        return 0.0

    # Benchmarks lead, but raw model size also matters: a 70B at Q4_K_M
    # carries far more world knowledge than a 7B Q4_K_M even when the
    # leaderboard score gap is modest. For MoE models, knowledge capacity
    # tracks *total* params (every expert contributes to what the model
    # knows), while routing keeps per-token compute small. Use total params
    # for the size score and let the speed term separately reward MoE
    # efficiency.
    size_basis_b = params_b
    size_score = 4.2 * math.log2(max(size_basis_b, 0.5)) + 9
    size_score = min(size_score, 35)

    has_benchmark = benchmark_avg is not None and benchmark_avg > 0
    is_direct = benchmark_source == "direct"
    is_self_reported = benchmark_source == "self_reported"
    is_inherited = benchmark_source in {"variant", "base_model", "line_interp"}

    bench_weight = _SOURCE_WEIGHTS.get(benchmark_source, 0.0)
    benchmark_score = 0.0
    if has_benchmark:
        raw = min(100.0, benchmark_avg)
        benchmark_score = raw * bench_weight

    # Quantization penalty
    quant_penalty = quant_quality_penalty(model, variant)
    quality_core = (benchmark_score + size_score) * (1 - quant_penalty)

    # Weak / unverifiable evidence gets an extra discount.
    if not has_benchmark:
        quality_core *= 0.55
    elif is_self_reported:
        quality_core *= 0.55  # uploader claim, easily fabricated
    elif is_inherited:
        quality_core *= 0.78

    # Runtime form factor penalty
    if fit_type == "partial_offload":
        quality_core *= 0.72
    elif fit_type == "cpu_only":
        quality_core *= 0.50

    # Speed acts as a usability gate rather than a ranking primary.
    required_speed = (
        8.0
        if fit_type == "full_gpu"
        else (4.0 if fit_type == "partial_offload" else 1.5)
    )
    if tok_per_sec > 0:
        if tok_per_sec < required_speed:
            speed_score = -8.0 * (1 - (tok_per_sec / required_speed))
        else:
            speed_score = min(8.0, math.log2(tok_per_sec / required_speed + 1.0) * 3.2)
    else:
        speed_score = -8.0

    # Popularity is a tie-breaker, never primary.
    downloads = max(model.downloads, family_downloads)
    likes = max(model.likes, family_likes)
    pop_score_raw = 0.0
    if downloads > 0:
        pop_score_raw += min(1.0, math.log10(max(downloads, 1)) / 6 * 1.0)
    if likes > 0:
        pop_score_raw += min(1.0, math.log10(max(likes, 1)) / 4 * 1.0)

    if is_direct:
        pop_weight = 0.0
    elif is_self_reported:
        pop_weight = 0.4  # uploader claim is weak — popularity acts as sanity check
    elif has_benchmark:
        pop_weight = 0.2
    else:
        pop_weight = 0.6
    pop_score = pop_score_raw * pop_weight

    # Source-trust bonus stays small.
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
                source_bonus_raw = 5.0
            else:
                source_bonus_raw = 0.0

    if is_direct:
        source_weight = 0.2
    elif is_self_reported:
        source_weight = 0.5
    elif has_benchmark:
        source_weight = 0.4
    else:
        source_weight = 0.6
    source_bonus = source_bonus_raw * source_weight

    # Generation lineage bonus: newest in a known family gets a small boost,
    # confirmed legacy versions get a small penalty. Helps surface Qwen3.6,
    # DeepSeek V4, Gemma 4, etc. against accumulated download leaders.
    gen_bonus = _generation_bonus(model.id)
    # When benchmark evidence is missing or self-reported, the lineage signal
    # carries more weight (we have less else to go on).
    if not has_benchmark or is_self_reported:
        gen_bonus *= 1.5
    elif is_direct:
        gen_bonus *= 0.6

    # Penalty for "uncensored / abliterated / heretic / RP" derivatives that
    # ride on a base model's score without independent benchmarking.
    derivative_penalty = _derivative_name_penalty(model.id)

    return max(
        0.0,
        min(
            100.0,
            quality_core
            + speed_score
            + pop_score
            + source_bonus
            + gen_bonus
            + derivative_penalty,
        ),
    )


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
    # Track the parameter count of the family's dominant member (highest
    # downloads). Used to detect quasi-fork uploads whose params differ
    # drastically from the family proper (e.g. a 6.6B MTP-head extracted
    # from a 158B base ending up tagged with the same family_id).
    family_dominant_params: dict[str, int] = {}
    family_dominant_downloads: dict[str, int] = {}
    for m in models:
        fid = m.family_id
        family_max_downloads[fid] = max(family_max_downloads.get(fid, 0), m.downloads)
        family_max_likes[fid] = max(family_max_likes.get(fid, 0), m.likes)
        if m.parameter_count and m.downloads >= family_dominant_downloads.get(fid, -1):
            family_dominant_downloads[fid] = m.downloads
            family_dominant_params[fid] = m.parameter_count

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
        if _is_excluded_model(model.id):
            continue
        if not _matches_profile(model, task_profile):
            continue
        if min_params_b is not None and _knowledge_capacity_b(model) < min_params_b:
            continue

        candidates = _iter_candidate_variants(model, quant_filter)
        if not candidates:
            continue

        fid = model.family_id
        # Uploader-reported evalResults are only ever last-resort evidence.
        self_reported = None
        if isinstance(model.benchmark_scores, dict):
            v = model.benchmark_scores.get("hf_eval")
            if isinstance(v, (int, float)) and v > 0:
                self_reported = float(v)

        bench_evidence = BenchmarkEvidence(score=None, confidence=0.0, source="none")
        if benchmark_scores or self_reported is not None:
            actual_params_b = (
                (model.parameter_count or 0) / 1e9 if model.parameter_count else None
            )
            bench_evidence = lookup_benchmark_evidence(
                model.id,
                model.base_model,
                benchmark_scores or {},
                ci_index=bench_ci_index,
                line_index=bench_line_index,
                line_bucket_index=bench_line_buckets,
                self_reported_score=self_reported,
                actual_params_b=actual_params_b,
            )
            # Family-size sanity check: if this model inherited benchmarks
            # via family/base_model lookup but its own params disagree
            # sharply with the family's dominant member, reject the
            # inheritance. Catches MTP heads / draft / abliterated forks
            # that share a family_id with their base but are effectively
            # different models.
            if bench_evidence.source in ("variant", "base_model", "line_interp"):
                dom_params = family_dominant_params.get(model.family_id)
                if dom_params and model.parameter_count and dom_params > 0:
                    ratio = model.parameter_count / dom_params
                    if ratio < 0.5 or ratio > 2.0:
                        bench_evidence = BenchmarkEvidence(
                            score=None, confidence=0.0, source="none"
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

            tok_per_sec = estimate_tok_per_sec(
                model, variant, best_gpu, compat.fit_type
            )
            if min_speed is not None and tok_per_sec < min_speed:
                continue

            bench_avg = None
            if bench_evidence.score is not None:
                if bench_evidence.source in {"direct", "self_reported"}:
                    bench_avg = bench_evidence.score
                else:
                    # Inherited evidence: scale by confidence so weak inheritance
                    # (e.g. line_interp at conf 0.22) gets discounted on top of
                    # the per-source weight in _compute_quality_score.
                    confidence = max(0.0, min(1.0, bench_evidence.confidence))
                    bench_avg = bench_evidence.score * (0.75 + 0.25 * confidence)

            compat.estimated_tok_per_sec = tok_per_sec
            (
                compat.speed_confidence,
                compat.speed_range_tok_per_sec,
                compat.speed_notes,
            ) = estimate_speed_uncertainty(
                model,
                variant,
                best_gpu,
                compat.fit_type,
                tok_per_sec,
            )
            compat.quality_score = _compute_quality_score(
                model,
                variant,
                tok_per_sec,
                compat.fit_type,
                family_downloads=family_max_downloads.get(fid, 0),
                family_likes=family_max_likes.get(fid, 0),
                benchmark_avg=bench_avg,
                benchmark_source=bench_evidence.source,
            )
            # Map evidence source to a 4-value display status. "self_reported"
            # is shown distinctly so users can spot uploader-claimed numbers.
            if bench_evidence.score is None:
                compat.benchmark_status = "none"
            elif bench_evidence.source == "direct":
                compat.benchmark_status = "direct"
            elif bench_evidence.source == "self_reported":
                compat.benchmark_status = "self_reported"
            else:
                compat.benchmark_status = "estimated"
            compat.benchmark_source = bench_evidence.source
            compat.benchmark_confidence = bench_evidence.confidence

            if (
                best_for_model is None
                or compat.quality_score > best_for_model.quality_score
            ):
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
        results.sort(
            key=lambda r: _family_selection_key(r, require_direct_top), reverse=True
        )

    # Junk floor: when at least one candidate scores ≥ 30, drop anything
    # below 20. This stops Q1_0 / Q2_0 derivatives (and other extreme-quant
    # repos) from occupying ranking slots when a *real* option exists. If
    # every candidate is junk (very tiny GPU + no fitting Q4) we keep the
    # whole list so the user still sees what they can run.
    if any(r.quality_score >= 30 for r in results):
        results = [r for r in results if r.quality_score >= 20]

    # Speed floor: a model that scores well on quality but runs at <1.5 t/s
    # in practice (e.g. DeepSeek-V4-Flash 158B partial-offloading 100GB to
    # CPU RAM from a 4GB GTX 1650) is not actually usable. Drop these
    # candidates unless every remaining option is sub-1.5 too, in which
    # case the user has hardware that cannot run anything responsively
    # and we still want to show what's available.
    if any(r.estimated_tok_per_sec >= 5.0 for r in results):
        results = [r for r in results if r.estimated_tok_per_sec >= 1.5]

    return results[:top_n]
