"""Tests for Asahi Linux (Apple Silicon on Linux) detection — Issue #29."""

from __future__ import annotations

import subprocess
from pathlib import Path

from whichllm.hardware import apple, cpu


# ---- CPU name fallback for ARM Linux ----


def test_cpu_name_lscpu_fallback(monkeypatch):
    """When /proc/cpuinfo has no model name (ARM), lscpu should be tried."""
    # Simulate ARM /proc/cpuinfo (no model name field)
    arm_cpuinfo = (
        "processor\t: 0\n"
        "BogoMIPS\t: 48.00\n"
        "Features\t: fp asimd evtstrm aes\n"
        "CPU implementer\t: 0x61\n"
    )
    monkeypatch.setattr("builtins.open", _fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    lscpu_output = "Architecture:            aarch64\nModel name:            Apple M2\n"

    def fake_run(args, **kwargs):
        if args == ["lscpu"]:
            return subprocess.CompletedProcess(args, 0, stdout=lscpu_output, stderr="")
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)

    assert cpu.detect_cpu_name() == "Apple M2"


def test_cpu_name_devicetree_fallback(monkeypatch, tmp_path):
    """When lscpu also fails, device tree model should be used."""
    arm_cpuinfo = "processor\t: 0\nFeatures\t: fp asimd\n"
    monkeypatch.setattr("builtins.open", _fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    # lscpu not available
    def fake_run(args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)

    # Device tree has the machine model
    dt_model = tmp_path / "model"
    dt_model.write_bytes(b"Apple MacBook Air (M2, 2022)\x00")
    monkeypatch.setattr(
        cpu, "_cpu_name_from_devicetree", lambda: _read_dt_model(dt_model)
    )

    assert cpu.detect_cpu_name() == "Apple M2"


def test_cpu_name_devicetree_extracts_chip_variants():
    """Verify chip name extraction from various device tree model strings."""
    assert _extract("Apple MacBook Air (M2, 2022)") == "Apple M2"
    assert _extract("Apple Mac Mini (M4 Pro, 2024)") == "Apple M4 Pro"
    assert _extract("Apple Mac Studio (M2 Ultra, 2023)") == "Apple M2 Ultra"
    assert _extract("Apple Mac Pro (M2 Max, 2023)") == "Apple M2 Max"
    assert _extract("Apple MacBook Pro (M1, 2020)") == "Apple M1"


def test_cpu_name_devicetree_non_apple():
    """Non-Apple device tree models should not produce an Apple chip name."""
    assert _extract("Raspberry Pi 4 Model B Rev 1.5") is None
    assert _extract("Qualcomm Snapdragon 8cx Gen 3") is None


def test_cpu_name_lscpu_ignores_dash(monkeypatch):
    """lscpu returning '-' as the model name should be ignored."""
    arm_cpuinfo = "processor\t: 0\n"
    monkeypatch.setattr("builtins.open", _fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    lscpu_output = "Architecture:            aarch64\nModel name:            -\n"

    def fake_run(args, **kwargs):
        if args == ["lscpu"]:
            return subprocess.CompletedProcess(args, 0, stdout=lscpu_output, stderr="")
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)
    monkeypatch.setattr(cpu, "_cpu_name_from_devicetree", lambda: "Apple M2")

    assert cpu.detect_cpu_name() == "Apple M2"


# ---- Apple GPU detection on Asahi Linux ----


def test_detect_asahi_gpu_from_sysfs(monkeypatch, tmp_path):
    """Asahi DRM driver should be detected and produce an Apple GPUInfo."""
    _setup_asahi_sysfs(tmp_path)
    monkeypatch.setattr(
        apple,
        "_chip_name_from_devicetree",
        lambda: "Apple M2",
    )
    monkeypatch.setattr("psutil.virtual_memory", _fake_vmem(24 * 1024**3))

    gpus = apple.detect_apple_gpu_linux(drm_path=tmp_path)

    assert len(gpus) == 1
    assert gpus[0].vendor == "apple"
    assert gpus[0].name == "Apple M2"
    assert gpus[0].vram_bytes == 24 * 1024**3
    assert gpus[0].shared_memory is True
    assert gpus[0].memory_bandwidth_gbps == 100.0  # M2 bandwidth


def test_detect_asahi_gpu_fallback_name(monkeypatch, tmp_path):
    """When device tree is unavailable, GPU should be named 'Apple Silicon'."""
    _setup_asahi_sysfs(tmp_path)
    monkeypatch.setattr(apple, "_chip_name_from_devicetree", lambda: None)
    monkeypatch.setattr("psutil.virtual_memory", _fake_vmem(8 * 1024**3))

    gpus = apple.detect_apple_gpu_linux(drm_path=tmp_path)

    assert len(gpus) == 1
    assert gpus[0].name == "Apple Silicon"


def test_detect_asahi_gpu_ignores_non_apple_drivers(tmp_path):
    """Non-Apple DRM drivers should not be detected as Apple GPUs."""
    card = tmp_path / "card0" / "device" / "driver"
    card.mkdir(parents=True)
    # Symlink to a non-Apple driver name
    target = tmp_path / "drivers" / "amdgpu"
    target.mkdir(parents=True)
    (tmp_path / "card0" / "device" / "driver").rmdir()
    (tmp_path / "card0" / "device" / "driver").symlink_to(target)

    assert apple.detect_apple_gpu_linux(drm_path=tmp_path) == []


def test_detect_asahi_gpu_no_drm(tmp_path):
    """When /sys/class/drm doesn't exist, return empty."""
    nonexistent = tmp_path / "no_drm"
    assert apple.detect_apple_gpu_linux(drm_path=nonexistent) == []


# ---- Helpers ----


def _setup_asahi_sysfs(tmp_path: Path) -> None:
    """Create a minimal sysfs tree mimicking the Asahi DRM driver."""
    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)
    # Symlink driver → .../asahi
    driver_target = tmp_path / "drivers" / "asahi"
    driver_target.mkdir(parents=True)
    (device_dir / "driver").symlink_to(driver_target)


def _fake_open(content: str):
    """Return a context manager that yields lines from *content*
    when the path is /proc/cpuinfo, and delegates otherwise."""
    import builtins
    import io

    real_open = builtins.open

    def patched_open(path, *a, **kw):
        if str(path) == "/proc/cpuinfo":
            return io.StringIO(content)
        return real_open(path, *a, **kw)

    return patched_open


def _fake_vmem(total: int):
    """Return a callable that mimics psutil.virtual_memory()."""
    from collections import namedtuple

    Vmem = namedtuple("svmem", ["total"])

    def _inner():
        return Vmem(total=total)

    return _inner


def _read_dt_model(path: Path) -> str | None:
    """Simulate _cpu_name_from_devicetree reading from a custom path."""
    import re

    try:
        raw = path.read_bytes()
        model = raw.decode("utf-8", errors="replace").strip().rstrip("\x00")
        if not model:
            return None
        m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
        if m:
            return f"Apple {m.group(1)}"
        return model
    except OSError:
        return None


def _extract(model: str) -> str | None:
    """Run the chip-name regex against a model string."""
    import re

    m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
    if m:
        return f"Apple {m.group(1)}"
    return None
