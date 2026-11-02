"""NVIDIA Jetson (Tegra) module identification from the kernel device tree.

Jetson boards need their own identification step because the two things NVIDIA
GPU detection normally relies on are both unavailable there:

1. **The GPU name does not identify the board.** Every Orin module reports the
   same ``"Orin (nvgpu)"`` through NVML and ``nvidia-smi``, from the 4 GB Orin
   Nano (34 GB/s) to the 64 GB AGX Orin (204.8 GB/s) — a 6x bandwidth spread
   behind one string.

2. **The memory queries fail.** Tegra is a unified-memory SoC with no dedicated
   VRAM, so the driver answers ``NVML_ERROR_NOT_SUPPORTED`` for
   ``nvmlDeviceGetMemoryInfo`` and ``nvmlDeviceGetMaxClockInfo``, and
   ``nvidia-smi`` prints ``[N/A]`` for both ``memory.total`` and
   ``clocks.max.memory`` (observed on the board below). The memory-clock trick
   used to separate same-name desktop variants (GTX 1650 GDDR5/GDDR6) is
   therefore not available either.

The module part number in the device tree is the remaining discriminator. It is
exported by the kernel, needs no root, and no NVIDIA tooling:

    $ cat /proc/device-tree/compatible | tr '\\0' '\\n'
    nvidia,p3768-0000+p3767-0000    # carrier board + module
    nvidia,p3767-0000               # module
    nvidia,tegra234                 # SoC generation

Only Orin part numbers are claimed. Anything else returns ``None`` and detection
behaves as it did before, rather than guessing a product from an SoC generation
— ``tegra210`` alone, for instance, covers both Jetson Nano and TX1. Xavier and
earlier are deliberately absent: their last JetPack has neither a Tegra NVML nor
a Tegra ``nvidia-smi``, so no GPU name reaches this code to begin with.

Measured on a Jetson Orin NX 16GB (``p3767-0000``, JetPack 6.2.3 / L4T
R36.5.0): before this module, ``whichllm hardware`` reported "No GPU detected —
CPU-only mode" while llama.cpp was serving Qwen3-8B on that same GPU.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEVICE_TREE = Path("/proc/device-tree")
_COMPATIBLE = _DEVICE_TREE / "compatible"

# A compatible entry is "vendor,value"; module part numbers look like "p3767-0000".
# fullmatch so a stray trailing newline cannot turn an exact match into a miss.
_PART_NUMBER_RE = re.compile(r"p\d{4}-\d{4}", re.IGNORECASE)
_TEGRA_SOC_RE = re.compile(r"tegra\d+", re.IGNORECASE)
# The Tegra driver appends "(nvgpu)" to the integrated GPU's name ("Orin
# (nvgpu)", "Xavier (nvgpu)"); older stacks report a "Tegra ..." name, which may
# carry the vendor prefix ("NVIDIA Tegra X1"), so this is a word search rather
# than an anchored one.
_TEGRA_GPU_NAME_RE = re.compile(r"\(nvgpu\)|\btegra\b", re.IGNORECASE)


@dataclass(frozen=True)
class JetsonModule:
    """An identified Jetson module.

    ``name`` is the canonical marketing name and is used as the GPU name, so the
    curated ``GPU_BANDWIDTH`` / ``NVIDIA_COMPUTE_CAPABILITY`` tables resolve it
    through the same substring lookup used for every other GPU.
    """

    name: str
    soc: str
    part_number: str


# Memory bandwidth is a property of the product line, not of the individual
# module SKU, so a part-number prefix identifies this family unambiguously:
# every P3701 is an AGX Orin at 204.8 GB/s whether it carries 32 or 64 GB.
_MODULE_FAMILIES: dict[str, str] = {
    "p3701": "Jetson AGX Orin",
}

# P3767 spans two product lines whose bandwidth differs (Orin NX 102.4 GB/s;
# Orin Nano 68 GB/s at 8 GB, 34 GB/s at 4 GB), so it is resolved per SKU. The
# mapping is published in the Jetson Linux Developer Guide's Quick Start
# supported-devices table, e.g. "Jetson Orin NX 16GB-DRAM (P3767-0000)".
_MODULE_SKUS: dict[str, str] = {
    "p3767-0000": "Jetson Orin NX 16GB",
    "p3767-0001": "Jetson Orin NX 8GB",
    "p3767-0003": "Jetson Orin Nano 8GB",
    "p3767-0004": "Jetson Orin Nano 4GB",
    "p3767-0005": "Jetson Orin Nano 8GB",
}


def _read_device_tree_strings(path: Path) -> list[str]:
    """Read a NUL-separated device-tree string property."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        # Distinguishable from "not a Jetson" only in the log, so say so.
        logger.debug("Could not read %s: %s", path, exc)
        return []
    return [s.strip() for s in raw.decode("utf-8", "replace").split("\0") if s.strip()]


@functools.lru_cache(maxsize=1)
def detect_jetson_module() -> JetsonModule | None:
    """Identify the Jetson module, or ``None`` when it cannot be identified.

    Cached: the device tree does not change while the process runs.

    Never raises. A missing or unreadable device tree, a non-Tegra board, or a
    Tegra board whose module is not in the tables above all return ``None``.
    """
    entries = _read_device_tree_strings(_COMPATIBLE)
    if not entries:
        return None

    part_numbers: list[str] = []
    soc: str | None = None
    for entry in entries:
        # "nvidia,p3768-0000+p3767-0000" -> the module half is considered too.
        for candidate in re.split(r"[,+]", entry):
            candidate = candidate.strip()
            if _PART_NUMBER_RE.fullmatch(candidate):
                part_numbers.append(candidate.lower())
            elif soc is None and _TEGRA_SOC_RE.fullmatch(candidate):
                soc = candidate.lower()

    if soc is None:
        return None

    for part_number in part_numbers:
        name = _MODULE_SKUS.get(part_number) or _MODULE_FAMILIES.get(
            part_number.split("-")[0]
        )
        if name:
            logger.debug("Jetson module %s -> %s", part_number, name)
            return JetsonModule(name=name, soc=soc, part_number=part_number)

    logger.debug("Unrecognised Jetson module (compatible=%s)", entries)
    return None


def is_tegra_gpu_name(name: str) -> bool:
    """True when a driver-reported GPU name is the Tegra integrated GPU.

    Checked before applying any Jetson-specific handling, so that running on a
    Jetson never changes how some other NVIDIA GPU would be described.
    """
    return bool(name) and bool(_TEGRA_GPU_NAME_RE.search(name))
