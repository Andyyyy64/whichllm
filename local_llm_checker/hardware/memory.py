"""RAM and disk space detection."""

from __future__ import annotations

import os
import shutil

import psutil


def detect_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    return psutil.virtual_memory().total


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
