"""RAM and disk space detection."""

from __future__ import annotations

import os
import shutil

import psutil


def detect_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    return psutil.virtual_memory().total


def estimate_usable_ram(total: int) -> int:
    """Estimate RAM available for model loading after OS/background reserve.

    Uses a bounded-reserve formula: total - clamp(total * 0.15, 4 GiB, 32 GiB).
    """
    _GiB = 1024**3
    reserve = int(total * 0.15)
    reserve = max(4 * _GiB, min(reserve, 32 * _GiB))
    return max(0, total - reserve)


def detect_disk_free_bytes(path: str | None = None) -> int:
    """Get free disk space in bytes at the given path.

    Defaults to the user's home directory, which is more accurate
    on macOS where / may be a read-only system volume.
    """
    if path is None:
        path = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except OSError:
        return 0
