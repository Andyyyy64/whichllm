"""Lookup tables that runtimes read a row at a time instead of keeping resident.

Some models carry very large embedding tables that each token reads only a
few rows of: Qwen3.8-Flash-Next's per-layer n-gram table (51.2B parameters)
and DeepSeek V4.1's Engram tables (196.6B). llama.cpp keeps such a tensor
memory-mapped on disk and reads the rows it needs (``--lazy-mode``, on by
default for tensors over 4 GiB), so the table needs disk space but neither
VRAM nor RAM, and adds nothing to what a token reads from memory.
"""

from __future__ import annotations

from whichllm.models.types import ModelInfo

# llama.cpp's --lazy-mode auto leaves tensors over 4 GiB on disk.
_LAZY_MIN_BYTES = 4 * 1024**3
# GGUF uploads keep these tables at IQ4_NL or better (ISTA-DASLab's packs hold
# the n-gram table at 4.5 bits whatever the rest of the file uses).
_MIN_BYTES_PER_PARAM = 4.5 / 8
# The rest of a checkpoint is never this small next to its table.
_MAX_FILE_SHARE = 0.9


def lookup_params(model: ModelInfo) -> int:
    """Parameters in the model's lookup tables, 0 when it has none."""
    shape = model.shape
    params = shape.lookup_params if shape else 0
    if params <= 0 or params >= model.parameter_count:
        return 0
    return params


def lazy_lookup_params(model: ModelInfo) -> int:
    """Lookup parameters a runtime leaves on disk, 0 when it keeps them."""
    params = lookup_params(model)
    if params * _MIN_BYTES_PER_PARAM < _LAZY_MIN_BYTES:
        return 0
    return params


def lazy_lookup_bytes(model: ModelInfo, file_bytes: int) -> int:
    """Bytes of ``file_bytes`` that are lookup tables left on disk."""
    params = lazy_lookup_params(model)
    if not params or file_bytes <= 0:
        return 0
    per_param = max(file_bytes / model.parameter_count, _MIN_BYTES_PER_PARAM)
    return min(int(params * per_param), int(file_bytes * _MAX_FILE_SHARE))


def resident_params(model: ModelInfo) -> int:
    """Parameters a runtime keeps in VRAM or RAM."""
    return model.parameter_count - lazy_lookup_params(model)


__all__ = [
    "lazy_lookup_bytes",
    "lazy_lookup_params",
    "lookup_params",
    "resident_params",
]
