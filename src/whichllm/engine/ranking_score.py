"""Quality scoring helpers for model ranking."""

from __future__ import annotations

import math

from whichllm.engine.quantization import quant_quality_penalty
from whichllm.engine.ranking_filters import (
    _derivative_name_penalty,
    _generation_bonus,
)
from whichllm.engine.ranking_sources import (
    _OFFICIAL_ORGS,
    _REPACKAGER_ORGS,
    _TRUSTED_CONVERTERS,
)
from whichllm.engine.types import CompatibilityResult
from whichllm.models.types import GGUFVariant, ModelInfo

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


def _family_selection_key(
    result: CompatibilityResult,
    require_direct_top: bool,
) -> tuple[float]:
    """Family-level selection key — single composite score.

    ``quality_score`` already includes the runtime fit penalty and speed
    adjustment. Keep final selection close to that displayed score so strong
    partial-offload candidates do not get discounted again while sorting.

    - ``direct_bonus`` (+5) gives independent leaderboard evidence a
      small edge at the same fit; cannot overturn a 6+ point quality gap
    """
    if require_direct_top and result.benchmark_status == "direct":
        direct_bonus = 5.0
    else:
        direct_bonus = 0.0
    cpu_penalty = -6.0 if result.fit_type == "cpu_only" else 0.0
    ctx_penalty = -20.0 if not result.context_fits else 0.0
    return (result.quality_score + direct_bonus + cpu_penalty + ctx_penalty,)


def _partial_offload_quality_factor(model: ModelInfo, offload_ratio: float) -> float:
    """Discount partial-offload candidates by how much leaves VRAM."""
    ratio = max(0.0, min(1.0, offload_ratio))
    if ratio >= 0.75:
        factor = 0.42
    elif ratio >= 0.60:
        factor = 0.52
    elif ratio >= 0.40:
        factor = 0.62
    elif ratio >= 0.25:
        factor = 0.76
    else:
        factor = 0.86

    # MoE offload is more nuanced: inactive experts and router/runtime
    # placement do not hurt equally. If the GPU can plausibly hold the
    # active expert working set, do not treat inactive-expert spill like
    # dense-layer spill.
    if model.is_moe and model.parameter_count_active:
        active_ratio = (
            model.parameter_count_active / model.parameter_count
            if model.parameter_count > 0
            else 1.0
        )
        active_ratio = max(0.0, min(1.0, active_ratio))
        active_set_fits = ratio <= max(0.0, 1.0 - active_ratio)
        if active_set_fits:
            if ratio >= 0.75:
                factor = max(factor, 0.66)
            elif ratio >= 0.60:
                factor = max(factor, 0.70)
            elif ratio >= 0.40:
                factor = max(factor, 0.76)
            elif ratio >= 0.25:
                factor = max(factor, 0.82)
            else:
                factor = max(factor, 0.88)
        else:
            factor = min(0.76, factor + 0.08)

    return factor


def _compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
    fit_type: str,
    offload_ratio: float = 0.0,
    family_downloads: int = 0,
    family_likes: int = 0,
    benchmark_avg: float | None = None,
    benchmark_source: str = "none",
) -> float:
    """Compute a quality score (0-100) for ranking."""
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
        quality_core *= _partial_offload_quality_factor(model, offload_ratio)
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
        if fit_type == "partial_offload":
            if offload_ratio >= 0.70:
                speed_score = -24.0
            elif offload_ratio >= 0.40:
                speed_score = -18.0
            else:
                speed_score = -12.0
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


__all__ = [
    "_SOURCE_WEIGHTS",
    "_compute_quality_score",
    "_family_selection_key",
    "_partial_offload_quality_factor",
]
