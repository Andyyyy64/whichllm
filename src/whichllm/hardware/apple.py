"""Apple Silicon detection via system_profiler (macOS) and sysfs (Asahi Linux)."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from whichllm.constants import GPU_BANDWIDTH
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_MiB = 1024**2


def _lookup_bandwidth(chip_name: str) -> float | None:
    chip_upper = chip_name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in chip_upper:
            return GPU_BANDWIDTH[key]
    return None


def _detect_iogpu_wired_limit_bytes() -> int | None:
    """Return the macOS GPU wired-memory limit when it is available."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        limit_mb = int(result.stdout.strip())
    except ValueError:
        return None
    return limit_mb * _MiB if limit_mb > 0 else None


def _detect_metal_recommended_working_set_bytes() -> int | None:
    """Return Metal's ``recommendedMaxWorkingSetSize`` for the default device.

    This is the GPU budget macOS actually enforces on Apple Silicon. It is well
    below total physical memory (0.74x on a stock 16 GB machine), and it is not
    exposed through ``sysctl`` or ``system_profiler`` — the Metal API is the
    only way to read it. Uses ``ctypes`` against the system Metal and objc
    libraries, so it adds no dependency.

    Returns ``None`` on non-macOS hosts or if the frameworks cannot be reached.
    """
    if sys.platform != "darwin":
        return None

    try:
        metal_path = ctypes.util.find_library("Metal")
        objc_path = ctypes.util.find_library("objc")
        if not metal_path or not objc_path:
            return None

        metal = ctypes.CDLL(metal_path)
        objc = ctypes.CDLL(objc_path)

        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None

        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        selector = objc.sel_registerName(b"recommendedMaxWorkingSetSize")

        send = objc.objc_msgSend
        send.restype = ctypes.c_uint64
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        working_set = send(device, selector)
    except (OSError, AttributeError, ValueError) as e:
        logger.debug(f"Metal working set size unavailable: {e}")
        return None

    return working_set if working_set > 0 else None


def detect_apple_gpu() -> list[GPUInfo]:
    """Detect Apple Silicon GPU. Returns empty list on non-macOS or failure."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("system_profiler not available (not macOS)")
        return []

    try:
        hw_items = data["SPHardwareDataType"]
        hw = hw_items[0]
        chip_name = hw.get("chip_type", "")
        if not chip_name:
            return []

        # Apple Silicon uses unified memory - get total physical memory
        memory_str = hw.get("physical_memory", "0 GB")
        # Parse "32 GB" -> bytes
        parts = memory_str.split()
        mem_value = int(parts[0])
        mem_unit = parts[1].upper() if len(parts) > 1 else "GB"
        multiplier = {"GB": 1024**3, "TB": 1024**4, "MB": 1024**2}.get(
            mem_unit, 1024**3
        )
        unified_memory = mem_value * multiplier
        wired_limit = _detect_iogpu_wired_limit_bytes()
        if wired_limit is not None:
            # The user raised or set iogpu.wired_limit_mb explicitly: honour it.
            unified_memory = min(unified_memory, wired_limit)
        else:
            # Stock machine (iogpu.wired_limit_mb == 0). The GPU still cannot
            # address all of physical memory, so fall back to the budget Metal
            # reports rather than assuming 100% of RAM is usable.
            metal_working_set = _detect_metal_recommended_working_set_bytes()
            if metal_working_set is not None:
                unified_memory = min(unified_memory, metal_working_set)

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,  # unified memory
                memory_bandwidth_gbps=_lookup_bandwidth(chip_name),
                shared_memory=True,
            )
        ]
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse Apple hardware info: {e}")
        return []


# ---- Asahi Linux (Apple Silicon on Linux) ----

_ASAHI_DRIVER_NAMES = ("asahi", "apple")


def _chip_name_from_devicetree() -> str | None:
    """Extract Apple chip name from Linux device tree."""
    try:
        raw = Path("/sys/firmware/devicetree/base/model").read_bytes()
        model = raw.decode("utf-8", errors="replace").strip().rstrip("\x00")
        if not model:
            return None
        m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
        if m:
            return f"Apple {m.group(1)}"
        return model
    except OSError:
        return None


def detect_apple_gpu_linux(
    drm_path: Path = Path("/sys/class/drm"),
) -> list[GPUInfo]:
    """Detect Apple Silicon GPU on Linux (Asahi driver).

    Returns empty list when no Asahi/Apple DRM device is found.
    """
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []

    for card in cards:
        driver = card / "device" / "driver"
        try:
            driver_name = driver.resolve().name
        except OSError:
            continue
        if driver_name not in _ASAHI_DRIVER_NAMES:
            continue

        chip_name = _chip_name_from_devicetree() or "Apple Silicon"

        # Unified memory — total system RAM is shared with the GPU.
        import psutil

        unified_memory = psutil.virtual_memory().total

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,
                memory_bandwidth_gbps=_lookup_bandwidth(chip_name),
                shared_memory=True,
            )
        ]

    return []
