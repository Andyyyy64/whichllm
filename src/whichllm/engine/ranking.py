"""Top-level model ranking orchestration."""

from __future__ import annotations

from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.performance import estimate_speed_uncertainty, estimate_tok_per_sec
from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.ranking_filters import (
    _detect_specializations,
    _is_excluded_model,
    _is_gguf_only_backend,
    _knowledge_capacity_b,
    _is_auxiliary_model,
    _matches_profile,
    _newer_generation,
    _non_gguf_format_note,
    _passes_evidence_filter,
)
from whichllm.engine.ranking_score import _compute_quality_score, _family_selection_key
from whichllm.engine.ranking_variants import _iter_candidate_variants
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.benchmark import (
    BenchmarkEvidence,
    build_line_bucket_index,
    build_score_index,
    build_score_name_index,
    lookup_benchmark_evidence,
)
from whichllm.models.roles import find_base_checkpoints
from whichllm.models.types import ModelInfo

# Quality points a code-tuned model gains under --profile coding when it has
# no coding benchmark results of its own.
_CODING_SPECIALIST_BONUS = 2.0
_TASK_EVIDENCE_SOURCES = frozenset({"direct", "variant", "base_model"})


def _hide_older_generations(
    results: list[CompatibilityResult],
) -> list[CompatibilityResult]:
    """Drop a model when a newer generation of its line ranks above it.

    Qwen3.8-27B, Qwen3.6-27B and Qwen3.5-27B filling the top three says less
    than three different models would. The newer row lists what it replaced,
    and an older generation that outranks the newer one stays.
    """
    kept: list[CompatibilityResult] = []
    for result in results:
        newer = next(
            (k for k in kept if _newer_generation(k.model, result.model)), None
        )
        if newer is None:
            kept.append(result)
        else:
            newer.older_generations.append(result.model.id)
    return kept


def _score_indices(scores: dict[str, float] | None) -> tuple:
    if not scores:
        return {}, {}, {}, {}
    ci_index, line_index = build_score_index(scores)
    return (
        ci_index,
        line_index,
        build_line_bucket_index(scores),
        build_score_name_index(scores),
    )


def _lookup_evidence(
    model: ModelInfo,
    scores: dict[str, float],
    indices: tuple,
    family_dominant_params: dict[str, int],
    self_reported: float | None,
) -> BenchmarkEvidence:
    ci_index, line_index, line_buckets, name_index = indices
    actual_params_b = (
        (model.parameter_count or 0) / 1e9 if model.parameter_count else None
    )
    evidence = lookup_benchmark_evidence(
        model.id,
        model.base_model,
        scores,
        ci_index=ci_index,
        line_index=line_index,
        line_bucket_index=line_buckets,
        self_reported_score=self_reported,
        actual_params_b=actual_params_b,
        name_index=name_index,
    )
    # Family-size sanity check: if this model inherited benchmarks
    # via family/base_model lookup but its own params disagree
    # sharply with the family's dominant member, reject the
    # inheritance. Catches MTP heads / draft / abliterated forks
    # that share a family_id with their base but are effectively
    # different models.
    if evidence.source in ("variant", "base_model", "line_interp"):
        dom_params = family_dominant_params.get(model.family_id)
        if dom_params and model.parameter_count and dom_params > 0:
            ratio = model.parameter_count / dom_params
            if ratio < 0.5 or ratio > 2.0:
                return BenchmarkEvidence(score=None, confidence=0.0, source="none")
    return evidence


def rank_models(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    context_length: int = 4096,
    top_n: int = 10,
    quant_filter: str | None = None,
    min_speed: float | None = None,
    benchmark_scores: dict[str, float] | None = None,
    task_profile: str = "general",
    task_scores: dict[str, float] | None = None,
    require_direct_top: bool = True,
    min_params_b: float | None = None,
    evidence_filter: str = "any",
    fit_filter: str = "any",
) -> list[CompatibilityResult]:
    """Rank models by quality for the given hardware. Returns top N results.

    ``task_scores`` are the active profile's scores (coding, math, vision).
    A model with task evidence is ranked on it instead of its general score.
    """
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

    # GGUF uploads by the repo they quantize: an official repo's synthetic
    # variants take the real size of the upload file they resolve to.
    gguf_uploads: dict[str, list[ModelInfo]] = {}
    for m in models:
        if m.gguf_variants and m.base_model and m.base_model_relation == "quantized":
            gguf_uploads.setdefault(m.base_model.casefold(), []).append(m)

    # Deduplicate by family: pick best variant per family
    seen_families: set[str] = set()

    # Sort models by downloads (popular first) to process best candidates first
    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)
    # Pretrain-only checkpoints are not chat models; never recommend them.
    base_checkpoints = find_base_checkpoints(models)
    coding_profile = task_profile.lower() == "coding"

    # Build benchmark indices once (case-insensitive, model line, repo name)
    general_indices = _score_indices(benchmark_scores)
    task_indices = _score_indices(task_scores)

    best_gpu = None
    for gpu in hardware.gpus:
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu

    for model in sorted_models:
        if _is_excluded_model(model.id) or model.id in base_checkpoints:
            continue
        if _is_auxiliary_model(model):
            continue
        if not _matches_profile(model, task_profile):
            continue
        if min_params_b is not None and _knowledge_capacity_b(model) < min_params_b:
            continue

        candidates = _iter_candidate_variants(
            model, quant_filter, gguf_uploads.get(model.id.casefold(), ())
        )
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
            bench_evidence = _lookup_evidence(
                model,
                benchmark_scores or {},
                general_indices,
                family_dominant_params,
                self_reported,
            )
        basis = "general"
        if task_scores:
            task_evidence = _lookup_evidence(
                model, task_scores, task_indices, family_dominant_params, None
            )
            # Interpolating a task score from other sizes of the line is too
            # loose; only the model itself, a variant or its base count.
            if task_evidence.source in _TASK_EVIDENCE_SOURCES:
                bench_evidence = task_evidence
                basis = task_profile.lower()
        if not _passes_evidence_filter(bench_evidence.source, evidence_filter):
            continue

        # 各variantを評価し、そのモデルで最もスコアが高いものを採用する
        best_for_model: CompatibilityResult | None = None
        for variant in candidates:
            if gguf_only_backend and variant is None:
                continue
            format_note = ""
            if variant is None:
                format_note = _non_gguf_format_note(
                    effective_quant_type(model, None), hardware
                )
                if format_note is None:
                    continue
            compat = check_compatibility(model, variant, hardware, context_length)
            if not compat.can_run:
                continue
            if format_note:
                compat.warnings.append(format_note)
            if fit_filter == "full_gpu" and compat.fit_type != "full_gpu":
                continue

            tok_per_sec = estimate_tok_per_sec(
                model,
                variant,
                best_gpu,
                compat.fit_type,
                placement=compat.placement,
                ram_bandwidth_gbps=hardware.ram_bandwidth_gbps,
            )
            if compat.uses_multi_gpu:
                tok_per_sec *= compat.multi_gpu_speed_factor
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
                placement=compat.placement,
                ram_bandwidth_source=hardware.ram_bandwidth_source,
            )
            if compat.uses_multi_gpu:
                compat.speed_confidence = "low"
                if tok_per_sec > 0:
                    compat.speed_range_tok_per_sec = (
                        round(tok_per_sec * 0.35, 1),
                        round(tok_per_sec * 2.0, 1),
                    )
                compat.speed_notes.append(
                    "Multi-GPU speed depends on layer/tensor split mode, "
                    "PCIe/NVLink bandwidth, and backend support; this estimate "
                    "does not assume ideal scaling."
                )
            compat.quality_score = _compute_quality_score(
                model,
                variant,
                tok_per_sec,
                compat.fit_type,
                offload_ratio=compat.offload_ratio,
                placement=compat.placement.kind if compat.placement else None,
                family_downloads=family_max_downloads.get(fid, 0),
                family_likes=family_max_likes.get(fid, 0),
                benchmark_avg=bench_avg,
                benchmark_source=bench_evidence.source,
            )
            if (
                coding_profile
                and basis == "general"
                and "coding" in _detect_specializations(model.id)
            ):
                # Between otherwise equal candidates, prefer the code-tuned
                # variant (Qwen3-Coder-30B-A3B over Qwen3-30B-A3B). Small
                # enough that an older coder cannot pass a stronger new model.
                compat.quality_score = min(
                    100.0, compat.quality_score + _CODING_SPECIALIST_BONUS
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
            compat.benchmark_basis = basis

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

    results = _hide_older_generations(results)

    # Clamp top_n: a negative value would slice from the end
    # (``results[:-5]``) and silently return a truncated, unrequested subset,
    # while 0 legitimately yields an empty list. The CLI rejects non-positive
    # --top, but guard here too so direct callers of this public helper never
    # get a truncated ranking from a stray negative count.
    return results[: max(top_n, 0)]
