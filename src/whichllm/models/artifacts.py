"""Resolve runnable/downloadable model artifacts for ranked recommendations."""

from __future__ import annotations

from whichllm.engine.types import CompatibilityResult
from whichllm.models.types import GGUFVariant, ModelInfo


def find_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    """Return the model's GGUF variant for a quantization type."""
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def is_same_model_family(candidate: ModelInfo, selected: ModelInfo) -> bool:
    """Return whether two repos represent the same base model family."""
    if candidate.id == selected.id:
        return True
    if candidate.family_id and selected.family_id:
        if candidate.family_id == selected.family_id:
            return True
    if candidate.base_model and candidate.base_model == selected.id:
        return True
    if selected.base_model and selected.base_model == candidate.id:
        return True
    if candidate.base_model and selected.base_model:
        return candidate.base_model == selected.base_model
    return False


def has_compatible_parameter_count(candidate: ModelInfo, selected: ModelInfo) -> bool:
    """Reject artifact repos that are clearly a different model size."""
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def resolve_ranked_gguf_artifact(
    selected_model: ModelInfo,
    selected_variant: GGUFVariant,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    """Resolve a ranked GGUF candidate to a real HF repo/file.

    The ranker may synthesize GGUF variants for official safetensors-only repos
    so they can be scored realistically. Output surfaces and `run` need the
    actual GGUF repository and filename when one exists.
    """
    desired_quant = quant_filter or selected_variant.quant_type

    if selected_model.gguf_variants:
        variant = find_gguf_variant(selected_model, desired_quant)
        return (selected_model, variant) if variant else None

    candidates: list[tuple[bool, int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not is_same_model_family(model, selected_model):
            continue
        if not has_compatible_parameter_count(model, selected_model):
            continue
        variant = find_gguf_variant(model, desired_quant)
        if not variant:
            continue
        explicit_base = model.base_model == selected_model.id
        candidates.append(
            (
                explicit_base,
                model.downloads,
                model.likes,
                model,
                variant,
            )
        )

    if not candidates:
        return None

    _, _, _, model, variant = max(candidates, key=lambda item: item[:3])
    return model, variant


def attach_resolved_artifacts(
    results: list[CompatibilityResult],
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> None:
    """Populate artifact fields on ranked results when a real artifact exists."""
    for result in results:
        result.artifact_model = None
        result.artifact_variant = None
        if not result.gguf_variant:
            continue
        resolved = resolve_ranked_gguf_artifact(
            result.model,
            result.gguf_variant,
            models,
            quant_filter=quant_filter,
        )
        if resolved:
            result.artifact_model, result.artifact_variant = resolved
