"""Tests for AMD GPU detection fallbacks."""

from __future__ import annotations

import subprocess
from io import StringIO

from rich.console import Console

from whichllm.hardware import amd
from whichllm.hardware.types import GPUInfo, HardwareInfo


def test_detect_amd_gpu_from_lspci_when_rocm_smi_missing(monkeypatch):
    output = (
        'c1:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. '
        '[AMD/ATI]" "Strix Halo [Radeon 8060S]" -r00 "Framework" "Device 0001"\n'
    )

    def fake_run(args, **kwargs):
        if args[0] == "rocm-smi":
            raise FileNotFoundError
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(amd.subprocess, "run", fake_run)

    gpus = amd.detect_amd_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].shared_memory is True
    assert gpus[0].memory_bandwidth_gbps == 256.0
    assert "Radeon 8060S" in gpus[0].name


def test_detect_strix_halo_rocm_smi_does_not_treat_aperture_as_vram(monkeypatch):
    def fake_run(args, **kwargs):
        if args[:2] == ["rocm-smi", "--showproductname"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='{"card0": {"Card SKU": "STRXLGEN"}}',
                stderr="",
            )
        if args[:3] == ["rocm-smi", "--showmeminfo", "vram"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='{"card0": {"VRAM Total Memory (B)": "536870912"}}',
                stderr="",
            )
        if args[:2] == ["rocm-smi", "--showdriverversion"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='{"card0": {"Driver version": "7.0.3"}}',
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(amd.subprocess, "run", fake_run)

    gpus = amd.detect_amd_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "STRXLGEN"
    assert gpus[0].vendor == "amd"
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 0
    assert gpus[0].rocm_version == "7.0.3"
    assert gpus[0].memory_bandwidth_gbps == 256.0


def test_detect_amd_gpu_ignores_intel_only_lspci(monkeypatch):
    """Regression: an Intel VGA row must not be reported as AMD just
    because 'Intel Corporation' contains the substring 'ati'."""
    output = (
        '00:02.0 "VGA compatible controller" "Intel Corporation" '
        '"Alder Lake-P GT1 [UHD Graphics]" -r0c -p00 '
        '"IP3 Tech (HK) Limited" "Device 8027"\n'
    )

    def fake_run(args, **kwargs):
        if args[0] == "rocm-smi":
            raise FileNotFoundError
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(amd.subprocess, "run", fake_run)
    # Isolate the lspci path: don't let a real sysfs probe leak in.
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: [])

    assert amd.detect_amd_gpus() == []


def test_detect_amd_gpu_from_sysfs_when_lspci_missing(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "product_name").write_text("AMD Radeon RX 9060 XT\n")
    (card / "mem_info_vram_total").write_text(str(16 * 1024**3))

    monkeypatch.setattr(amd, "_detect_from_lspci", lambda: [])
    original_sysfs = amd._detect_from_sysfs
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpus = amd._detect_amd_gpus_fallback()

    assert len(gpus) == 1
    assert gpus[0].vendor == "amd"
    assert gpus[0].name == "AMD Radeon RX 9060 XT"
    assert gpus[0].vram_bytes == 16 * 1024**3
    assert gpus[0].shared_memory is False


def test_display_amd_shared_memory_without_zero_kb(monkeypatch):
    from whichllm.output import _console as console_mod
    from whichllm.output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, force_terminal=False))

    display_mod.display_hardware(
        HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Strix Halo [Radeon 8060S]",
                    vendor="amd",
                    vram_bytes=0,
                    shared_memory=True,
                )
            ],
            cpu_name="CPU",
            cpu_cores=16,
            ram_bytes=128 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            os="linux",
        )
    )

    output = buf.getvalue()
    assert "shared memory" in output
    assert "256 GB/s" not in output
    assert "0 KB" not in output
