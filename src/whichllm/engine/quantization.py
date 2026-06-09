"""Quantization helpers shared across ranking and estimators."""

from __future__ import annotations

import re

from whichllm.constants import QUANT_QUALITY_PENALTY
from whichllm.models.types import GGUFVariant, ModelInfo

# GGUFでないリポジトリ名から量子化方式を推定する
_NON_GGUF_PATTERNS: list[tuple[str, str]] = [
    (r"(^|[-_/])awq($|[-_/])", "AWQ"),
    (r"(^|[-_/])gptq($|[-_/])", "GPTQ"),
    # 4-bit microscaling float formats. Anchored so they only match a distinct
    # repo-name token, never a substring of an unrelated id.
    (r"(^|[-_/])mxfp4($|[-_/])", "MXFP4"),
    (r"(^|[-_/])nvfp4($|[-_/])", "NVFP4"),
    (r"(bnb[-_/]?4bit|nf4|int4|4bit)", "BNB_4BIT"),
    (r"(int8|8bit)", "INT8"),
    (r"(^|[-_/])fp8($|[-_/])", "FP8"),
    (r"(^|[-_/])bf16($|[-_/])", "BF16"),
    (r"(^|[-_/])(fp16|f16)($|[-_/])", "FP16"),
]

# GGUF以外の簡易推定: 重み1つあたりのバイト数
_NON_GGUF_BYTES_PER_WEIGHT: dict[str, float] = {
    "AWQ": 0.5,
    "GPTQ": 0.5,
    "BNB_4BIT": 0.5,
    "MXFP4": 0.53125,
    "NVFP4": 0.5625,
    "INT8": 1.0,
    "FP8": 1.0,
    "BF16": 2.0,
    "FP16": 2.0,
}

# GGUF以外の簡易推定: 品質低下率
_NON_GGUF_QUALITY_PENALTY: dict[str, float] = {
    "AWQ": 0.05,
    "GPTQ": 0.05,
    "BNB_4BIT": 0.07,
    "MXFP4": 0.06,
    "NVFP4": 0.05,
    "INT8": 0.02,
    "FP8": 0.02,
    "BF16": 0.0,
    "FP16": 0.0,
}


def infer_non_gguf_quant_type(model_id: str) -> str:
    """Infer non-GGUF quantization type from a model repo ID."""
    lower = model_id.lower()
    for pattern, quant_type in _NON_GGUF_PATTERNS:
        if re.search(pattern, lower):
            return quant_type
    return "FP16"


def effective_quant_type(model: ModelInfo, variant: GGUFVariant | None) -> str:
    """Return effective quantization type for a model+variant pair."""
    if variant:
        return variant.quant_type.upper()
    return infer_non_gguf_quant_type(model.id)


def estimate_weight_bytes(model: ModelInfo, variant: GGUFVariant | None) -> int:
    """Estimate model weight size in bytes."""
    if variant:
        return variant.file_size_bytes
    quant_type = infer_non_gguf_quant_type(model.id)
    bytes_per_weight = _NON_GGUF_BYTES_PER_WEIGHT.get(quant_type, 2.0)
    return int(model.parameter_count * bytes_per_weight)


def quant_quality_penalty(model: ModelInfo, variant: GGUFVariant | None) -> float:
    """Return quality penalty fraction for a quantization format."""
    quant_type = effective_quant_type(model, variant).upper()
    if quant_type in QUANT_QUALITY_PENALTY:
        return QUANT_QUALITY_PENALTY[quant_type]
    return _NON_GGUF_QUALITY_PENALTY.get(quant_type, 0.05)
