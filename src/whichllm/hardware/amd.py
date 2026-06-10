"""AMD GPU detection via rocm-smi with Linux fallback probes."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

from whichllm.constants import AMD_SHARED_MEMORY_APU_MARKERS, _GiB
from whichllm.hardware.gpu_db import _static_bandwidth, resolve_detected_bandwidth
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_DISPLAY_CLASSES = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)


def _lookup_bandwidth(name: str) -> float | None:
    """Curated GPU_BANDWIDTH lookup, compound-lspci aware. Kept for regression
    tests; live detection goes through ``resolve_detected_bandwidth``, which
    also consults dbgpu."""
    return _static_bandwidth(name)


def _is_shared_memory_apu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in AMD_SHARED_MEMORY_APU_MARKERS)


def _normalize_apu_vram(name: str, vram_bytes: int) -> int:
    if _is_shared_memory_apu(name) and vram_bytes < 2 * _GiB:
        return 0
    return vram_bytes


def _make_gpu(
    name: str,
    *,
    vram_bytes: int = 0,
    rocm_version: str | None = None,
) -> GPUInfo:
    shared_memory = _is_shared_memory_apu(name)
    return GPUInfo(
        name=name,
        vendor="amd",
        vram_bytes=_normalize_apu_vram(name, vram_bytes),
        rocm_version=rocm_version,
        memory_bandwidth_gbps=resolve_detected_bandwidth(name, vram_bytes),
        shared_memory=shared_memory,
    )


_AMD_VENDOR_MARKERS = (
    "advanced micro devices",
    "amd/ati",
    "[amd]",
    "[ati]",
    "ati technologies",
)


def _vendor_is_amd(vendor: str) -> bool:
    vendor_lower = vendor.lower()
    return any(marker in vendor_lower for marker in _AMD_VENDOR_MARKERS)


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
        # `lspci -mm` is the machine-parsable format:
        #   <slot> "<class>" "<vendor>" "<device>" [flags] ["<subvendor>" "<subdevice>"]
        # Parse the quoted columns properly and check the vendor field
        # specifically, instead of substring-matching the whole line (which
        # would treat e.g. "Intel Corpor[ati]on" as AMD).
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) < 4:
            continue
        device_class, vendor, device = tokens[1], tokens[2], tokens[3]
        if device_class.lower() not in _DISPLAY_CLASSES:
            continue
        if not _vendor_is_amd(vendor):
            continue
        name = device.strip() or "AMD Graphics"
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _read_int(path: Path) -> int:
    try:
        text = path.read_text().strip()
    except OSError:
        return 0
    try:
        return int(text, 0)
    except ValueError:
        return 0


def _detect_from_sysfs(drm_path: Path = Path("/sys/class/drm")) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
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
        if vendor != "0x1002":
            continue

        name = "AMD Graphics"
        try:
            product_name = (device / "product_name").read_text().strip()
            if product_name:
                name = product_name
        except OSError:
            pass

        vram_bytes = _read_int(device / "mem_info_vram_total")
        key = f"{name}:{vram_bytes}"
        if key in seen:
            continue
        seen.add(key)
        gpus.append(_make_gpu(name, vram_bytes=vram_bytes))
    return gpus


def _read_sysfs_amd_vram(drm_path: Path = Path("/sys/class/drm")) -> list[int]:
    """Read VRAM for each AMD GPU from sysfs, in card order."""
    result: list[int] = []
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
        if vendor != "0x1002":
            continue
        result.append(_read_int(device / "mem_info_vram_total"))
    return result


def _detect_amd_gpus_fallback() -> list[GPUInfo]:
    # Prefer sysfs: it provides VRAM and sometimes a clean product name.
    sysfs_gpus = _detect_from_sysfs()

    if sysfs_gpus:
        # If sysfs names are generic ("AMD Graphics"), enrich with lspci names.
        has_generic = any(g.name == "AMD Graphics" for g in sysfs_gpus)
        if has_generic:
            lspci_names = _detect_from_lspci()
            if lspci_names and len(lspci_names) == len(sysfs_gpus):
                return [
                    _make_gpu(
                        lspci_names[i] if gpu.name == "AMD Graphics" else gpu.name,
                        vram_bytes=gpu.vram_bytes,
                    )
                    for i, gpu in enumerate(sysfs_gpus)
                ]
        return sysfs_gpus

    # sysfs unavailable; fall back to lspci (name only), enriching with
    # sysfs VRAM when possible.
    names = _detect_from_lspci()
    if names:
        vram_list = _read_sysfs_amd_vram()
        return [
            _make_gpu(name, vram_bytes=vram_list[i] if i < len(vram_list) else 0)
            for i, name in enumerate(names)
        ]
    return []


def detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs. Returns empty list on failure."""
    gpus: list[GPUInfo] = []

    # Get product names
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return _detect_amd_gpus_fallback()
        product_data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("rocm-smi not available or failed")
        return _detect_amd_gpus_fallback()

    # Get VRAM info
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return _detect_amd_gpus_fallback()
        mem_data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("Failed to get AMD VRAM info")
        return _detect_amd_gpus_fallback()

    # Get ROCm version
    rocm_version = None
    try:
        result = subprocess.run(
            ["rocm-smi", "--showdriverversion", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            driver_data = json.loads(result.stdout)
            # Extract version from first card entry
            for key, val in driver_data.items():
                if isinstance(val, dict) and "Driver version" in val:
                    rocm_version = val["Driver version"]
                    break
    except Exception:
        pass

    # Parse GPU info - rocm-smi JSON keys are like "card0", "card1"
    for key in sorted(product_data.keys()):
        if not key.startswith("card"):
            continue
        card_info = product_data[key]
        # rocm-smi JSON uses "Card Series" (capital S) for the human-readable name.
        # Fall back through SKU only as a last resort.
        name = (
            card_info.get("Card Series")
            or card_info.get("Card series")
            or card_info.get("Card SKU")
            or "Unknown AMD GPU"
        )

        vram_total = 0
        if key in mem_data:
            vram_str = mem_data[key].get("VRAM Total Memory (B)", "0")
            try:
                vram_total = int(vram_str)
            except (ValueError, TypeError):
                pass

        gpus.append(_make_gpu(name, vram_bytes=vram_total, rocm_version=rocm_version))

    return gpus
