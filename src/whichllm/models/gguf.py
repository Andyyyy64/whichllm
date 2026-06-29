"""GGUF filename parsing and variant extraction helpers."""

from __future__ import annotations

import re

from whichllm.constants import QUANT_BYTES_PER_WEIGHT
from whichllm.models.types import GGUFVariant

_GGUF_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Filename quant tokens that name the same format under a different spelling
# than the canonical key used by QUANT_BYTES_PER_WEIGHT / QUANT_QUALITY_PENALTY.
_QUANT_ALIASES = {
    "FP16": "F16",
    "FP32": "F32",
}


def _extract_quant_type(filename: str) -> str:
    """Extract quantization type from a GGUF filename.

    The returned key is canonicalized to match QUANT_BYTES_PER_WEIGHT, so
    callers can look it up directly. Returns "unknown" when nothing matches.
    """
    patterns = [
        r"[.-](Q\d+_K_[SMLA])",
        r"[.-](Q\d+_\d+)",
        r"[.-](Q\d+_K)",
        r"[.-](TQ\d+_\d+)",
        r"[.-](IQ\d+_\w+)",
        r"[.-](MXFP4|NVFP4)",
        r"[.-](F16|FP16|BF16|F32|FP32)",
    ]
    upper = filename.upper()
    for pattern in patterns:
        m = re.search(pattern, upper)
        if m:
            quant = m.group(1)
            return _QUANT_ALIASES.get(quant, quant)
    return "unknown"


def _estimate_gguf_size(param_count: int, quant_type: str) -> int:
    """Estimate GGUF file size from parameter count and quantization type."""
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant_type.upper(), 0.5625)  # default Q4_K_M
    return int(param_count * bpw)


def _extract_gguf_variants(data: dict, param_count: int) -> list[GGUFVariant]:
    """Extract GGUF variants from HF sibling metadata."""
    quant_sizes: dict[str, int] = {}
    quant_first_filename: dict[str, str] = {}
    siblings = data.get("siblings", []) or []
    for sib in siblings:
        fname = sib.get("rfilename", "")
        if not fname.endswith(".gguf") or fname.startswith("."):
            continue
        quant = _extract_quant_type(fname)
        if quant == "unknown":
            continue
        size = sib.get("size", 0)
        if not isinstance(size, int) or size < 0:
            size = 0

        # Split GGUF files are summed into one candidate per quant.
        quant_sizes[quant] = quant_sizes.get(quant, 0) + size
        if quant not in quant_first_filename or _GGUF_SPLIT_RE.search(
            quant_first_filename[quant]
        ):
            quant_first_filename[quant] = fname

    gguf_variants = []
    for quant, total_size in quant_sizes.items():
        if total_size <= 0:
            total_size = _estimate_gguf_size(param_count, quant)
        gguf_variants.append(
            GGUFVariant(
                filename=quant_first_filename[quant],
                quant_type=quant,
                file_size_bytes=total_size,
            )
        )
    return gguf_variants
