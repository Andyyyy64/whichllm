"""Resolve runnable/downloadable model artifacts for ranked recommendations."""

from __future__ import annotations

import re

from whichllm.engine.types import CompatibilityResult
from whichllm.models.types import GGUFVariant, ModelInfo

_DERIVATIVE_PROVENANCE_TAGS = frozenset({"adapter", "finetune", "merge"})


def find_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    """Return the model's GGUF variant for a quantization type."""
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def has_compatible_parameter_count(candidate: ModelInfo, selected: ModelInfo) -> bool:
    """Reject artifact repos that are clearly a different model size."""
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def is_equivalent_quantization(candidate: ModelInfo, selected: ModelInfo) -> bool:
    """Return whether provenance identifies a direct quantization of selected."""
    candidate_name = re.sub(r"[^a-z0-9]+", "", candidate.name.casefold())
    if candidate_name.endswith("gguf"):
        candidate_name = candidate_name.removesuffix("gguf")
    selected_name = re.sub(r"[^a-z0-9]+", "", selected.id.rsplit("/", 1)[-1].casefold())
    selected_owner = (
        re.sub(r"[^a-z0-9]+", "", selected.id.split("/", 1)[0].casefold())
        if "/" in selected.id
        else ""
    )
    accepted_names = {selected_name, f"{selected_owner}{selected_name}"}
    normalized_tags = {tag.casefold() for tag in candidate.tags}
    base_relations = {
        parts[1]
        for tag in normalized_tags
        if len(parts := tag.split(":", 2)) == 3
        and parts[0] == "base_model"
        and parts[2] == selected.id.casefold()
    }
    return (
        candidate.base_model is not None
        and candidate.base_model.casefold() == selected.id.casefold()
        and candidate.base_model_relation == "quantized"
        and base_relations == {"quantized"}
        and candidate_name in accepted_names
        and not normalized_tags.intersection(_DERIVATIVE_PROVENANCE_TAGS)
    )


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

    candidates: list[tuple[int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not is_equivalent_quantization(
            model, selected_model
        ):
            continue
        if not has_compatible_parameter_count(model, selected_model):
            continue
        variant = find_gguf_variant(model, desired_quant)
        if not variant:
            continue
        candidates.append(
            (
                model.downloads,
                model.likes,
                model,
                variant,
            )
        )

    if not candidates:
        return None

    _, _, model, variant = max(candidates, key=lambda item: item[:2])
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
