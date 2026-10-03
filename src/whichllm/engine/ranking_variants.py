"""Candidate GGUF variant selection for model ranking."""

from __future__ import annotations

import re

from whichllm.constants import QUANT_BYTES_PER_WEIGHT, QUANT_PREFERENCE_ORDER
from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.ranking_sources import _OFFICIAL_ORGS
from whichllm.models.types import GGUFVariant, ModelInfo

_SYNTHETIC_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
_PREQUANTIZED_REPO_RE = re.compile(
    r"-(awq|gptq|bnb|fp8|fp16|bf16|mxfp4|nvfp4|int4|int8|4bit|8bit|gguf)$",
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


__all__ = [
    "_PREQUANTIZED_REPO_RE",
    "_SYNTHETIC_QUANTS",
    "_iter_candidate_variants",
    "_synthesize_variants_for_official_repo",
]
