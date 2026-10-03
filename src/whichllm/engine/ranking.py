"""Top-level model ranking orchestration."""

from __future__ import annotations

from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.performance import estimate_speed_uncertainty, estimate_tok_per_sec
from whichllm.engine.ranking_filters import (
    _is_excluded_model,
    _is_gguf_only_backend,
    _knowledge_capacity_b,
    _matches_profile,
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
    lookup_benchmark_evidence,
)
from whichllm.models.types import ModelInfo

_MULTI_GPU_SPEED_FACTOR = 0.70


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
    fit_filter: str = "any",
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
            if fit_filter == "full_gpu" and compat.fit_type != "full_gpu":
                continue

            tok_per_sec = estimate_tok_per_sec(
                model, variant, best_gpu, compat.fit_type
            )
            if compat.uses_multi_gpu:
                tok_per_sec *= _MULTI_GPU_SPEED_FACTOR
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

    # Clamp top_n: a negative value would slice from the end
    # (``results[:-5]``) and silently return a truncated, unrequested subset,
    # while 0 legitimately yields an empty list. The CLI rejects non-positive
    # --top, but guard here too so direct callers of this public helper never
    # get a truncated ranking from a stray negative count.
    return results[: max(top_n, 0)]
