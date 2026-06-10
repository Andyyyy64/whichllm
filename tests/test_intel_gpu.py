"""Tests for Linux Intel integrated GPU detection."""

from __future__ import annotations

import subprocess
from io import StringIO

from rich.console import Console

from whichllm.hardware import intel
from whichllm.hardware.types import GPUInfo, HardwareInfo


def test_detect_intel_gpu_from_lspci(monkeypatch):
    output = (
        '00:02.0 "VGA compatible controller" "Intel Corporation" '
        '"Alder Lake-P GT1 [UHD Graphics]" -r0c "Dell" "Device 0b19"\n'
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)

    gpus = intel.detect_intel_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert "UHD Graphics" in gpus[0].name


def test_detect_intel_gpu_ignores_non_display_lspci(monkeypatch):
    output = '00:00.0 "Host bridge" "Intel Corporation" "Device 4621"\n'

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)
    monkeypatch.setattr(intel, "_detect_from_sysfs", lambda: [])

    assert intel.detect_intel_gpus() == []


def test_detect_intel_gpu_from_sysfs_when_lspci_missing(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "uevent").write_text("PCI_SLOT_NAME=0000:00:02.0\n")

    monkeypatch.setattr(intel, "_detect_from_lspci", lambda: [])
    original_sysfs = intel._detect_from_sysfs
    monkeypatch.setattr(intel, "_detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpus = intel.detect_intel_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].name == "Intel Integrated Graphics"


def test_display_intel_shared_memory_without_zero_kb(monkeypatch):
    from whichllm.output import _console as console_mod
    from whichllm.output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, force_terminal=False))

    display_mod.display_hardware(
        HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Alder Lake-P GT1 [UHD Graphics]",
                    vendor="intel",
                    vram_bytes=0,
                    shared_memory=True,
                )
            ],
            cpu_name="CPU",
            cpu_cores=8,
            ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            os="linux",
        )
    )

    output = buf.getvalue()
    assert "shared memory" in output
    assert "0 KB" not in output
