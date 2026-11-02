"""System RAM bandwidth: how fast CPU-side decode can read weights.

CPU-only runs, MoE experts kept in RAM and layers that do not fit on the GPU
all read their weights from system memory once per generated token, so this
number bounds their speed the way VRAM bandwidth bounds a GPU.

Bandwidth is transfer rate x bus width. Windows reports both per memory module
(Win32_PhysicalMemory), and so does systemd's udev on Linux (the DMI memory
properties it exports without root). WSL asks Windows through powershell.exe.
Elsewhere the CPU family gives a typical figure, and anything unknown falls
back to dual-channel DDR4-3200. ``--ram-bandwidth`` overrides all of it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from whichllm.hardware.types import GPUInfo
from whichllm.utils import _cache_dir

logger = logging.getLogger(__name__)

# Dual-channel DDR4-3200: the slowest memory a current desktop is likely to
# have, so an unknown system is never promised more than it can read.
DEFAULT_RAM_BANDWIDTH_GBPS = 51.2

# SMBIOS memory type codes reported by Win32_PhysicalMemory.SMBIOSMemoryType.
_SMBIOS_MEMORY_TYPES = {
    24: "DDR3",
    26: "DDR4",
    29: "LPDDR3",
    30: "LPDDR4",
    34: "DDR5",
    35: "LPDDR5",
}

# systemd's udev (70-memory.rules, systemd 248+) exports the DMI memory table
# as MEMORY_DEVICE_<n>_* properties readable without root.
_UDEV_DMI_DATA = Path("/run/udev/data/+dmi:id")
_UDEV_DEVICE_RE = re.compile(r"^MEMORY_DEVICE_(\d+)_([A-Z_]+)=(.*)$")

# Windows memory modules rarely change; reading them costs a PowerShell start.
_MODULE_CACHE_FILE = "ram-modules.json"
_MODULE_CACHE_TTL_SECONDS = 7 * 24 * 3600

# Widest memory bus a CPU family drives. Consumer desktop and laptop parts
# have two 64-bit channels, so four DIMMs still read at 128 bits.
_PLATFORM_BUS_BITS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bepyc\b", re.IGNORECASE), 768),
    (re.compile(r"threadripper|\bxeon\b", re.IGNORECASE), 512),
    (re.compile(r"ryzen\s+ai\s+max", re.IGNORECASE), 256),
)
_CONSUMER_BUS_BITS = 128

# Typical memory for CPU families that only ship with one generation of it.
# Families sold with either DDR4 or DDR5 (Intel 12th-14th gen) are left out so
# they get the conservative default instead.
_CPU_FAMILY_ESTIMATES: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (
        re.compile(r"ryzen\s+ai\s+max", re.IGNORECASE),
        256.0,
        "LPDDR5X-8000, 256-bit",
    ),
    (
        re.compile(r"\bultra\s+\d+\s+2\d\dv\b", re.IGNORECASE),
        136.5,
        "LPDDR5X-8533, 128-bit",
    ),
    (re.compile(r"\bcore\b.{0,8}\bultra\b", re.IGNORECASE), 89.6, "DDR5-5600, 128-bit"),
    (re.compile(r"ryzen\s+ai\s+\d", re.IGNORECASE), 89.6, "DDR5-5600, 128-bit"),
    (
        # AM5 desktop (7600X, 7950X3D, 8700G, 9950X) and Zen 4 laptops
        # (7840HS, 8845HS).
        re.compile(
            r"ryzen\s+\d+\s+(?:pro\s+)?(?:[789]\d{2}0(?:x3d|x|g|f)?|[78]\d4\d)",
            re.IGNORECASE,
        ),
        83.2,
        "DDR5-5200, 128-bit",
    ),
)


@dataclass(frozen=True)
class RamBandwidth:
    gbps: float
    source: str  # "detected" | "estimated" | "unified" | "default"
    detail: str | None = None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _platform_bus_bits(cpu_name: str) -> int:
    for pattern, bits in _PLATFORM_BUS_BITS:
        if pattern.search(cpu_name):
            return bits
    return _CONSUMER_BUS_BITS


def bandwidth_from_modules(modules: list[object], cpu_name: str) -> RamBandwidth | None:
    """Bandwidth from Win32_PhysicalMemory rows; None if they say too little.

    The slowest module sets the bus clock. Each module adds its data width (64
    bits for a DIMM, less for soldered LPDDR packages) up to the widest bus
    the CPU family supports.
    """
    rates: list[int] = []
    bus_bits = 0
    kinds: set[str] = set()
    for module in modules:
        if not isinstance(module, dict):
            continue
        rate = _positive_int(module.get("ConfiguredClockSpeed")) or _positive_int(
            module.get("Speed")
        )
        if rate is None or not 800 <= rate <= 12_000:
            continue
        rates.append(rate)
        bus_bits += min(_positive_int(module.get("DataWidth")) or 64, 64)
        kind = module.get("MemoryType") or _SMBIOS_MEMORY_TYPES.get(
            _positive_int(module.get("SMBIOSMemoryType"))
        )
        if kind:
            kinds.add(kind)
    if not rates:
        return None
    rate = min(rates)
    bus_bits = min(bus_bits, _platform_bus_bits(cpu_name))
    kind = kinds.pop() if len(kinds) == 1 else "RAM"
    return RamBandwidth(
        gbps=round(rate * bus_bits / 8 / 1000, 1),
        source="detected",
        detail=f"{kind}-{rate}, {bus_bits}-bit",
    )


def estimate_from_cpu_name(cpu_name: str) -> RamBandwidth | None:
    for pattern, gbps, detail in _CPU_FAMILY_ESTIMATES:
        if pattern.search(cpu_name):
            return RamBandwidth(gbps=gbps, source="estimated", detail=detail)
    return None


def _linux_memory_modules() -> list[object]:
    return _read_udev_modules(_UDEV_DMI_DATA)


def _read_udev_modules(path: Path) -> list[object]:
    """Memory modules from udev's DMI properties, as Win32-style rows."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    devices: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        match = _UDEV_DEVICE_RE.match(line.removeprefix("E:"))
        if match:
            index, key, value = match.groups()
            devices.setdefault(int(index), {})[key] = value.strip()
    modules: list[object] = []
    for props in devices.values():
        if props.get("PRESENT") == "0":
            continue
        modules.append(
            {
                "ConfiguredClockSpeed": props.get("CONFIGURED_SPEED_MTS"),
                "Speed": props.get("SPEED_MTS"),
                "DataWidth": props.get("DATA_WIDTH"),
                "MemoryType": props.get("TYPE") or None,
            }
        )
    return modules


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _cached_windows_memory_modules(executable: str) -> list[object]:
    cache = _cache_dir() / _MODULE_CACHE_FILE
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - data["cached_at"] < _MODULE_CACHE_TTL_SECONDS:
            return data["modules"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    modules = _windows_memory_modules(executable)
    if modules:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({"cached_at": time.time(), "modules": modules}),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("Could not cache memory modules: %s", exc)
    return modules


def _windows_memory_modules(executable: str = "powershell") -> list[object]:
    script = (
        "Get-CimInstance Win32_PhysicalMemory | Select-Object "
        "ConfiguredClockSpeed, Speed, DataWidth, SMBIOSMemoryType | "
        "ConvertTo-Json -Depth 2"
    )
    try:
        result = subprocess.run(
            [executable, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        logger.debug("Windows memory module query failed: %s", exc)
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def unified_bandwidth(gpus: list[GPUInfo]) -> RamBandwidth | None:
    """Bandwidth of an Apple or other unified-memory GPU, which is the RAM's."""
    for gpu in gpus:
        if (gpu.vendor == "apple" or gpu.shared_memory) and gpu.memory_bandwidth_gbps:
            return RamBandwidth(
                gbps=gpu.memory_bandwidth_gbps,
                source="unified",
                detail=f"unified memory ({gpu.name})",
            )
    return None


def detect_ram_bandwidth(
    os_name: str, cpu_name: str, gpus: list[GPUInfo]
) -> RamBandwidth:
    """Best available RAM bandwidth for this machine. Never fails."""
    if any(gpu.vendor == "apple" for gpu in gpus):
        unified = unified_bandwidth(gpus)
        if unified is not None:
            return unified
    modules: list[object] = []
    if os_name == "windows":
        modules = _cached_windows_memory_modules("powershell")
    elif os_name == "linux":
        modules = _linux_memory_modules()
        if not modules and _is_wsl():
            modules = _cached_windows_memory_modules("powershell.exe")
    detected = bandwidth_from_modules(modules, cpu_name)
    if detected is not None:
        return detected
    estimated = estimate_from_cpu_name(cpu_name)
    if estimated is not None:
        return estimated
    unified = unified_bandwidth(gpus)
    if unified is not None:
        return unified
    return RamBandwidth(
        gbps=DEFAULT_RAM_BANDWIDTH_GBPS,
        source="default",
        detail="assumed DDR4-3200, 128-bit",
    )
