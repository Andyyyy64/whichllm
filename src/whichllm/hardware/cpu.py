"""CPU detection: name, cores, AVX2/AVX512 support."""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _cpu_name_from_lscpu() -> str | None:
    """Try to get CPU model name from lscpu (works on ARM/aarch64)."""
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip().startswith("Model name"):
                    name = line.split(":", 1)[1].strip()
                    if name and name != "-":
                        return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _cpu_name_from_devicetree() -> str | None:
    """Extract CPU/chip name from device tree (ARM Linux, Asahi)."""
    try:
        raw = Path("/sys/firmware/devicetree/base/model").read_bytes()
        model = raw.decode("utf-8", errors="replace").strip().rstrip("\x00")
        if not model:
            return None
        # "Apple MacBook Air (M2, 2022)" → "Apple M2"
        m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
        if m:
            return f"Apple {m.group(1)}"
        return model
    except OSError:
        return None


def _clean_cpu_name(name: str | None) -> str | None:
    if name is None:
        return None
    cleaned = name.strip()
    if not cleaned or cleaned == "-" or cleaned.lower() == "name":
        return None
    return cleaned


def _cpu_name_from_wmic() -> str | None:
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        name = _clean_cpu_name(line)
        if name:
            return name
    return None


def _cpu_name_from_windows_cim() -> str | None:
    script = (
        "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"
    )
    for executable in ("powershell", "pwsh"):
        try:
            result = subprocess.run(
                [executable, "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            continue

        if result.returncode != 0:
            continue

        for line in result.stdout.splitlines():
            name = _clean_cpu_name(line)
            if name:
                return name
    return None


def detect_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
            # ARM/aarch64: /proc/cpuinfo has no model name field.
            # Try lscpu, then device tree.
            name = _cpu_name_from_lscpu() or _cpu_name_from_devicetree()
            if name:
                return name
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                name = _clean_cpu_name(result.stdout)
                if name:
                    return name
        elif system == "Windows":
            name = _cpu_name_from_wmic() or _cpu_name_from_windows_cim()
            if name:
                return name
    except Exception as e:
        logger.debug(f"Failed to detect CPU name: {e}")
    return "Unknown CPU"


def _count_physical_cores_linux() -> int | None:
    """Count unique physical cores from /proc/cpuinfo (handles WSL2)."""
    try:
        physical_ids: set[tuple[str, str]] = set()
        current_physical = ""
        current_core = ""
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    current_physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    current_core = line.split(":", 1)[1].strip()
                    physical_ids.add((current_physical, current_core))
        if physical_ids:
            return len(physical_ids)
    except Exception:
        pass
    return None


def detect_cpu_cores() -> int:
    """Get number of physical CPU cores."""
    import psutil

    cores = psutil.cpu_count(logical=False)
    if cores:
        return cores

    # Fallback for WSL2 where psutil may return None for physical cores
    if platform.system() == "Linux":
        linux_cores = _count_physical_cores_linux()
        if linux_cores:
            return linux_cores

    return psutil.cpu_count(logical=True) or 1


def _detect_avx_linux() -> tuple[bool, bool]:
    """Detect AVX2/AVX512 on Linux via /proc/cpuinfo."""
    has_avx2 = False
    has_avx512 = False
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            flags_line = ""
            for line in content.split("\n"):
                if line.startswith("flags"):
                    flags_line = line
                    break
            has_avx2 = "avx2" in flags_line
            has_avx512 = "avx512f" in flags_line
    except Exception:
        pass
    return has_avx2, has_avx512


def _detect_avx_darwin() -> tuple[bool, bool]:
    """Detect AVX2/AVX512 on macOS via sysctl."""
    has_avx2 = False
    has_avx512 = False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx2_0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx2 = result.stdout.strip() == "1"
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx512f"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx512 = result.stdout.strip() == "1"
    except Exception:
        pass
    return has_avx2, has_avx512


def detect_avx_support() -> tuple[bool, bool]:
    """Detect AVX2 and AVX512 support. Returns (has_avx2, has_avx512)."""
    system = platform.system()
    if system == "Linux":
        return _detect_avx_linux()
    elif system == "Darwin":
        return _detect_avx_darwin()
    # Windows / fallback: assume AVX2 on modern CPUs
    return True, False
