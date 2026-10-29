"""GGUF filename parsing and variant extraction helpers."""

from __future__ import annotations

import re

from whichllm.constants import QUANT_BYTES_PER_WEIGHT
from whichllm.models.types import GGUFVariant

_GGUF_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
# Files published next to the weights that are not the model: vision
# projectors and speculative-decoding drafts (DFlash, DSpark).
_AUXILIARY_GGUF_RE = re.compile(
    r"mmproj|(?:^|[-_./])(?:dflash\d*|dspark|draft)(?:[-_.]|$)", re.IGNORECASE
)
# No quant of a whole model is this small (IQ1_S is about 1.6 bits per
# weight), so a real file below it holds something else.
_MIN_BITS_PER_WEIGHT = 1.0

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


def _split_part(filename: str) -> int:
    """0 for a single file, else the part number of a split GGUF."""
    match = _GGUF_SPLIT_RE.search(filename)
    return int(match.group(1)) if match else 0


def _extract_gguf_variants(data: dict, param_count: int) -> list[GGUFVariant]:
    """Extract GGUF variants from HF sibling metadata.

    One variant per quant: a single file, or the parts of one split file
    (-00001-of-00003) summed. Other files with the same quant are separate
    downloads (an MTP build next to the plain one, Q4_0's ARM repacks), so
    they do not add to its size. Vision projectors (mmproj) and draft
    models are not the model's weights, and neither is a file too small to
    hold it.
    """
    # quant -> file stem (split suffix removed) -> [size, size missing, file]
    files: dict[str, dict[str, list]] = {}
    siblings = data.get("siblings", []) or []
    for sib in siblings:
        fname = sib.get("rfilename", "")
        if not fname.endswith(".gguf") or fname.startswith("."):
            continue
        if _AUXILIARY_GGUF_RE.search(fname):
            continue
        quant = _extract_quant_type(fname)
        if quant == "unknown":
            continue
        size = sib.get("size", 0)
        missing = not isinstance(size, int) or isinstance(size, bool) or size <= 0
        entry = files.setdefault(quant, {}).setdefault(
            _GGUF_SPLIT_RE.sub("", fname), [0, False, fname]
        )
        entry[0] += 0 if missing else size
        entry[1] = entry[1] or missing
        # The file to download: part 1 of a split file.
        if _split_part(fname) < _split_part(entry[2]):
            entry[2] = fname

    gguf_variants = []
    for quant, stems in files.items():
        candidates = [
            entry
            for entry in stems.values()
            # A draft or MTP head published next to the weights.
            if entry[1]
            or param_count <= 0
            or entry[0] * 8 / param_count >= _MIN_BITS_PER_WEIGHT
        ]
        if not candidates:
            continue
        # A single file before a split one, else the first listed.
        size, missing, filename = min(candidates, key=lambda e: _split_part(e[2]) > 0)
        is_estimated = missing or size <= 0
        gguf_variants.append(
            GGUFVariant(
                filename=filename,
                quant_type=quant,
                file_size_bytes=(
                    _estimate_gguf_size(param_count, quant) if is_estimated else size
                ),
                is_estimated=is_estimated,
            )
        )
    return gguf_variants
