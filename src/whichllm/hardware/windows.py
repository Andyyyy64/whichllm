"""Windows GPU detection via Win32_VideoController."""

from __future__ import annotations

import json
import logging
import re
import subprocess

from whichllm.constants import AMD_SHARED_MEMORY_APU_MARKERS, _GiB
from whichllm.hardware.gpu_db import resolve_detected_bandwidth
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_WINDOWS_DISCRETE_VRAM_FLOORS: tuple[tuple[str, int], ...] = (
    # AMD sells RX 9060 XT in 8 GB and 16 GB variants. WMI AdapterRAM is a
    # uint32 and can report ~4 GB for larger cards, so keep a conservative
    # known floor instead of trusting the capped value.
    ("RX 9060 XT", 8 * _GiB),
)


def _vendor_from_name(name: str) -> str | None:
    name_lower = name.lower()
    if any(token in name_lower for token in ("amd", "radeon")):
        return "amd"
    if "intel" in name_lower:
        return "intel"
    return None


def _parse_memory_value(value: object) -> int:
    if value is None:
        return 0
    try:
        ram = int(value)
    except (TypeError, ValueError):
        return 0
    return max(ram, 0)


def _is_amd_shared_memory_apu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in AMD_SHARED_MEMORY_APU_MARKERS)


def _is_intel_discrete_gpu(name: str) -> bool:
    return (
        re.search(
            r"\barc(?:\(tm\))?\s+(?:pro\s+)?[ab]\d{2,3}",
            name,
            re.IGNORECASE,
        )
        is not None
    )


def _is_intel_shared_memory_gpu(name: str, vram_bytes: int) -> bool:
    name_lower = name.lower()
    if _is_intel_discrete_gpu(name):
        return False
    if any(
        token in name_lower
        for token in (
            "uhd",
            "iris",
            " xe",
            "hd graphics",
            "arc(tm) graphics",
            "intel(r) graphics",
        )
    ):
        return True
    return vram_bytes < 2 * _GiB


def _is_shared_memory_gpu(name: str, vendor: str, vram_bytes: int) -> bool:
    if vendor == "amd":
        return _is_amd_shared_memory_apu(name)
    if vendor == "intel":
        return _is_intel_shared_memory_gpu(name, vram_bytes)
    return False


def _apply_discrete_vram_floor(name: str, vram_bytes: int) -> int:
    name_upper = name.upper()
    for marker, floor in _WINDOWS_DISCRETE_VRAM_FLOORS:
        if marker in name_upper and 0 < vram_bytes < floor:
            return floor
    return vram_bytes


def _memory_from_entry(entry: dict) -> int:
    dedicated = _parse_memory_value(entry.get("DedicatedVideoMemory"))
    if dedicated > 0:
        return dedicated
    return _parse_memory_value(entry.get("AdapterRAM"))


def detect_windows_gpus() -> list[GPUInfo]:
    """Detect non-NVIDIA Windows GPUs. Returns empty list on failure."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "$controllers = Get-CimInstance Win32_VideoController; "
                    "$controllers | ForEach-Object { "
                    "$dedicated = $null; "
                    "if ($_.PNPDeviceID) { "
                    "try { "
                    "$enumPath = 'Registry::HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Enum\\' "
                    "+ $_.PNPDeviceID + '\\Device Parameters'; "
                    "$enumProps = Get-ItemProperty -LiteralPath $enumPath -ErrorAction Stop; "
                    "if ($enumProps.VideoID) { "
                    "$videoPath = 'Registry::HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Video\\' "
                    "+ $enumProps.VideoID + '\\0000'; "
                    "$videoProps = Get-ItemProperty -LiteralPath $videoPath -ErrorAction Stop; "
                    "$dedicated = $videoProps.'HardwareInformation.qwMemorySize'; "
                    "} "
                    "} catch {} "
                    "} "
                    "[PSCustomObject]@{"
                    "Name=$_.Name; "
                    "AdapterRAM=$_.AdapterRAM; "
                    "DedicatedVideoMemory=$dedicated"
                    "} "
                    "} | ConvertTo-Json -Depth 3"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        logger.debug(f"Windows GPU detection failed: {e}")
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.debug("Failed to parse Windows GPU JSON")
        return []

    entries = data if isinstance(data, list) else [data]
    gpus: list[GPUInfo] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("Name") or "").strip()
        if not name:
            continue
        vendor = _vendor_from_name(name)
        if vendor is None:
            continue
        vram_bytes = _memory_from_entry(entry)
        shared_memory = _is_shared_memory_gpu(name, vendor, vram_bytes)
        if shared_memory:
            vram_bytes = 0
        else:
            vram_bytes = _apply_discrete_vram_floor(name, vram_bytes)
        key = f"{vendor}:{name}:{vram_bytes}"
        if key in seen:
            continue
        seen.add(key)
        gpus.append(
            GPUInfo(
                name=name,
                vendor=vendor,
                vram_bytes=vram_bytes,
                memory_bandwidth_gbps=resolve_detected_bandwidth(name, vram_bytes),
                shared_memory=shared_memory,
            )
        )
    return gpus
