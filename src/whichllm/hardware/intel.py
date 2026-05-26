"""Intel integrated GPU detection on Linux."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)


_DISPLAY_CLASSES = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)


def _normalize_lspci_name(line: str) -> str:
    parts = [p.strip() for p in line.split('"') if p.strip() and p.strip() != "\t"]
    for i, part in enumerate(parts):
        if part.lower() == "intel corporation" and i + 1 < len(parts):
            return parts[i + 1]
    return "Intel Integrated Graphics"


def _detect_from_lspci() -> list[str]:
    try:
        result = subprocess.run(
            ["lspci", "-mm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("lspci not available or timed out")
        return []

    if result.returncode != 0:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        line_lower = line.lower()
        if "intel" not in line_lower or not any(
            display_class in line_lower for display_class in _DISPLAY_CLASSES
        ):
            continue
        name = _normalize_lspci_name(line)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _detect_from_sysfs(drm_path: Path = Path("/sys/class/drm")) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []

    for card in cards:
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x8086":
            continue

        name = "Intel Integrated Graphics"
        try:
            product_name = (device / "product_name").read_text().strip()
            if product_name:
                name = product_name
        except OSError:
            pass

        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def detect_intel_gpus() -> list[GPUInfo]:
    """Detect Linux Intel iGPUs. Returns empty list on failure."""
    names = _detect_from_lspci() or _detect_from_sysfs()

    return [
        GPUInfo(
            name=name,
            vendor="intel",
            vram_bytes=0,
            shared_memory=True,
        )
        for name in names
    ]
