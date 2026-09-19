"""Model family grouping logic."""

from __future__ import annotations

import re

from whichllm.models.types import ModelFamily, ModelInfo


def _normalize_name(model_id: str) -> str:
    """Normalize model ID for grouping by removing org prefix and packaging/quantization suffixes."""
    name = model_id.lower()
    # Strip org prefix (e.g. "bartowski/Meta-Llama-3.1" -> "meta-llama-3.1")
    if "/" in name:
        name = name.split("/", 1)[1]
    # Strip common org prefixes in model names (e.g. "qwen_qwen3-8b" -> "qwen3-8b")
    name = re.sub(r"^(qwen_|meta-llama_|google_)", "", name)
    # Remove common suffixes (applied repeatedly to handle stacked suffixes)
    suffixes = [
        r"-gguf$",
        r"-gptq$",
        r"-awq$",
        r"-hf$",
        r"-fp8$",
        r"-fp16$",
        r"-bf16$",
        r"-mxfp4$",
        r"-nvfp4$",
        r"-\d+bit$",
    ]
    for _ in range(3):  # multiple passes to strip stacked suffixes
        prev = name
        for suffix in suffixes:
            name = re.sub(suffix, "", name)
        if name == prev:
            break

    return name


def group_models(models: list[ModelInfo]) -> list[ModelFamily]:
    """Group quantizations without erasing checkpoint identifiers."""
    # Only explicit quantizations share their upstream checkpoint's identity.
    keys = {
        model.id.lower(): (
            model.id.lower()
            if model.base_model and model.base_model_relation != "quantized"
            else _normalize_name(model.id)
        )
        for model in models
    }
    groups: dict[str, list[ModelInfo]] = {}
    for model in models:
        key = keys[model.id.lower()]
        if model.base_model and model.base_model_relation == "quantized":
            key = keys.get(model.base_model.lower(), _normalize_name(model.base_model))
        groups.setdefault(key, []).append(model)

    # Build families
    families: list[ModelFamily] = []

    for group_key, group in groups.items():
        if not group:
            continue

        # Pick the base model. Priority order:
        #   1. Models that are referenced by another group member's base_model
        #      field, even when a quantization has more downloads.
        #   2. Models without GGUF/quant suffixes and no base_model of their
        #      own (the original checkpoint).
        #   3. Anything left in the group.
        # Within the chosen tier, pick highest downloads as a tiebreaker.
        referenced_as_base: set[str] = {m.base_model for m in group if m.base_model}
        upstream_candidates = [m for m in group if m.id in referenced_as_base]
        if upstream_candidates:
            base_candidates = upstream_candidates
        else:
            base_candidates = [
                m for m in group if not m.gguf_variants or m.base_model is None
            ]
            if not base_candidates:
                base_candidates = group

        base = max(base_candidates, key=lambda m: m.downloads)
        variants = [m for m in group if m.id != base.id]

        # Set family_id on all members
        family_id = group_key
        base.family_id = family_id
        for v in variants:
            v.family_id = family_id

        # Collect best benchmark scores across family
        best_bench: dict[str, float] = {}
        for m in group:
            for k, v in m.benchmark_scores.items():
                if k not in best_bench or v > best_bench[k]:
                    best_bench[k] = v

        families.append(
            ModelFamily(
                family_id=family_id,
                display_name=base.name,
                base_model=base,
                variants=variants,
                best_benchmark=best_bench,
            )
        )

    return families
