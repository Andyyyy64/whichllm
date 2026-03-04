"""Model ranking: score and select the best models for the user's hardware."""

from __future__ import annotations

import math

from local_llm_checker.constants import QUANT_PREFERENCE_ORDER, QUANT_QUALITY_PENALTY
from local_llm_checker.engine.compatibility import check_compatibility
from local_llm_checker.engine.performance import estimate_tok_per_sec
from local_llm_checker.engine.types import CompatibilityResult
from local_llm_checker.hardware.types import HardwareInfo
from local_llm_checker.models.types import GGUFVariant, ModelInfo


def _pick_best_variant(
    model: ModelInfo,
    hardware: HardwareInfo,
    context_length: int,
    quant_filter: str | None = None,
) -> tuple[GGUFVariant | None, CompatibilityResult]:
    """Pick the best GGUF variant that fits in hardware, or None."""
    if not model.gguf_variants:
        # No GGUF: check if FP16 fits
        result = check_compatibility(model, None, hardware, context_length)
        return None, result

    # Filter by quant type if specified
    candidates = model.gguf_variants
    if quant_filter:
        filtered = [v for v in candidates if v.quant_type.upper() == quant_filter.upper()]
        if filtered:
            candidates = filtered

    # Sort by preference order
    def variant_sort_key(v: GGUFVariant) -> int:
        try:
            return QUANT_PREFERENCE_ORDER.index(v.quant_type.upper())
        except ValueError:
            return len(QUANT_PREFERENCE_ORDER)

    candidates = sorted(candidates, key=variant_sort_key)

    # Try each variant, pick first that fits
    best_result: CompatibilityResult | None = None
    best_variant: GGUFVariant | None = None

    for variant in candidates:
        result = check_compatibility(model, variant, hardware, context_length)
        if result.can_run and result.fit_type == "full_gpu":
            return variant, result
        if result.can_run and (best_result is None or best_result.fit_type != "full_gpu"):
            best_variant = variant
            best_result = result

    if best_result and best_variant:
        return best_variant, best_result

    # Nothing fits: return the smallest variant's result for reporting
    smallest = min(candidates, key=lambda v: v.file_size_bytes)
    result = check_compatibility(model, smallest, hardware, context_length)
    return smallest, result


def _compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
) -> float:
    """Compute a quality score (0-100) for ranking.

    Factors:
    - Model size (larger = better quality, up to a point)
    - Quantization penalty
    - Speed bonus
    - Popularity (downloads/likes as tiebreaker)
    """
    # Base quality from parameter count (log scale, 7B=50, 70B=80, 405B=95)
    params_b = model.parameter_count / 1e9
    if model.is_moe and model.parameter_count_active:
        effective_b = model.parameter_count_active / 1e9
    else:
        effective_b = params_b

    if effective_b <= 0:
        return 0.0

    # Log scale: 1B->30, 7B->53, 14B->62, 34B->72, 70B->80, 405B->92
    size_score = 15 * math.log2(max(effective_b, 0.5)) + 30
    size_score = min(size_score, 95)

    # Quantization penalty
    quant_penalty = 0.0
    if variant:
        quant_penalty = QUANT_QUALITY_PENALTY.get(variant.quant_type.upper(), 0.05)
    quality_after_quant = size_score * (1 - quant_penalty)

    # Speed bonus: prefer models that are reasonably fast (>10 tok/s gets bonus)
    speed_bonus = 0.0
    if tok_per_sec > 0:
        speed_bonus = min(5.0, tok_per_sec / 10.0 * 2.5)

    # Popularity tiebreaker (0-3 points)
    pop_score = 0.0
    if model.downloads > 0:
        pop_score += min(1.5, math.log10(max(model.downloads, 1)) / 6 * 1.5)
    if model.likes > 0:
        pop_score += min(1.5, math.log10(max(model.likes, 1)) / 4 * 1.5)

    return min(100.0, quality_after_quant + speed_bonus + pop_score)


def rank_models(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    context_length: int = 4096,
    top_n: int = 10,
    quant_filter: str | None = None,
    min_speed: float | None = None,
) -> list[CompatibilityResult]:
    """Rank models by quality for the given hardware. Returns top N results."""
    results: list[CompatibilityResult] = []

    # Deduplicate by family: pick best variant per family
    seen_families: set[str] = set()

    # Sort models by downloads (popular first) to process best candidates first
    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)

    best_gpu = None
    for gpu in hardware.gpus:
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu

    for model in sorted_models:
        variant, compat = _pick_best_variant(model, hardware, context_length, quant_filter)
        if not compat.can_run:
            continue

        # Estimate speed
        tok_per_sec = estimate_tok_per_sec(model, variant, best_gpu, compat.fit_type)
        compat.estimated_tok_per_sec = tok_per_sec

        # Apply min speed filter
        if min_speed is not None and tok_per_sec < min_speed:
            continue

        # Compute quality score
        compat.quality_score = _compute_quality_score(model, variant, tok_per_sec)

        # Deduplicate by family: keep the one with highest quality score
        family_key = model.family_id
        if family_key in seen_families:
            # Check if this is better than existing
            existing = next(
                (r for r in results if r.model.family_id == family_key), None
            )
            if existing and compat.quality_score > existing.quality_score:
                results.remove(existing)
                results.append(compat)
            continue

        seen_families.add(family_key)
        results.append(compat)

    # Sort by quality score descending
    results.sort(key=lambda r: r.quality_score, reverse=True)

    return results[:top_n]
