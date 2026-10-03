"""Tests for AMD GPU detection fallbacks."""

from __future__ import annotations

import subprocess
import json
from io import StringIO

import pytest

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


@pytest.mark.parametrize(
    "product",
    [
        {"Card SKU": "STRXLGEN"},
        {
            "Card Series": "AMD Radeon Graphics",
            "Card Model": "0x1586",
            "Card SKU": "STRXLGEN",
        },
        {"Card Series": "AMD Radeon Graphics"},
        {"Card Series": "AMD Radeon Graphics", "Card SKU": None},
    ],
)
def test_detect_strix_halo_rocm_smi_does_not_treat_aperture_as_vram(
    monkeypatch, product
):
    def fake_run(args, **kwargs):
        if args[:2] == ["rocm-smi", "--showproductname"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {
                        "card0": product,
                        "card1": {
                            "Card Series": "AMD Radeon RX 7900 XTX",
                            "Card SKU": "unknown",
                        },
                    }
                ),
                stderr="",
            )
        if args[:3] == ["rocm-smi", "--showmeminfo", "vram"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {
                        "card0": {"VRAM Total Memory (B)": "536870912"},
                        "card1": {"VRAM Total Memory (B)": str(24 * 1024**3)},
                    }
                ),
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

    assert len(gpus) == 2
    assert gpus[0].name == product.get("Card Series", "STRXLGEN")
    assert gpus[0].vendor == "amd"
    shared = product.get("Card SKU") == "STRXLGEN"
    assert gpus[0].shared_memory is shared
    assert gpus[0].vram_bytes == (0 if shared else 536870912)
    assert gpus[0].rocm_version == "7.0.3"
    if shared:
        assert gpus[0].memory_bandwidth_gbps == 256.0
    assert gpus[1].name == "AMD Radeon RX 7900 XTX"
    assert gpus[1].shared_memory is False
    assert gpus[1].vram_bytes == 24 * 1024**3


def test_detect_strix_halo_generic_pci_id_from_lspci(monkeypatch):
    aperture = 512 * 1024**2
    output = (
        'c6:00.0 "Display controller [0380]" "Advanced Micro Devices, Inc. '
        '[AMD/ATI] [1002]" "Device [1586]" -rc1 -p00 '
        '"Advanced Micro Devices, Inc. [AMD/ATI] [1002]" "Device [0124]"\n'
    )

    def fake_run(args, **kwargs):
        if args[0] == "rocm-smi":
            raise FileNotFoundError
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(amd.subprocess, "run", fake_run)
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: [])
    monkeypatch.setattr(amd, "_read_sysfs_amd_vram", lambda: [aperture])

    gpus = amd.detect_amd_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "Strix Halo [Radeon 8060S]"
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 0
    assert gpus[0].memory_bandwidth_gbps == 256.0


def test_detect_strix_halo_pci_id_from_sysfs(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "device").write_text("0x1586\n")
    (card / "mem_info_vram_total").write_text(str(512 * 1024**2))

    monkeypatch.setattr(amd, "_detect_from_lspci", lambda: [])
    original_sysfs = amd._detect_from_sysfs
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpus = amd._detect_amd_gpus_fallback()

    assert len(gpus) == 1
    assert gpus[0].name == "Strix Halo [Radeon 8060S]"
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 0
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


# ---------- Issue #61: RX 6750 XT detection ----------


def test_sysfs_generic_name_enriched_by_lspci(monkeypatch, tmp_path):
    """When sysfs gives 'AMD Graphics' and lspci gives a descriptive name,
    the fallback should use the lspci name with sysfs VRAM."""
    _GiB = 1024**3

    # sysfs: generic name but has VRAM
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "mem_info_vram_total").write_text(str(12 * _GiB))
    # no product_name → falls back to "AMD Graphics"

    lspci_name = "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"
    original_sysfs = amd._detect_from_sysfs
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: original_sysfs(tmp_path))
    monkeypatch.setattr(amd, "_detect_from_lspci", lambda: [lspci_name])

    gpus = amd._detect_amd_gpus_fallback()

    assert len(gpus) == 1
    assert gpus[0].name == lspci_name
    assert gpus[0].vram_bytes == 12 * _GiB
    assert gpus[0].shared_memory is False


def test_sysfs_product_name_preferred_over_lspci(monkeypatch, tmp_path):
    """When sysfs gives a real product name, it should be used even if
    lspci is also available."""
    _GiB = 1024**3

    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "product_name").write_text("AMD Radeon RX 6750 XT\n")
    (card / "mem_info_vram_total").write_text(str(12 * _GiB))

    original_sysfs = amd._detect_from_sysfs
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: original_sysfs(tmp_path))
    monkeypatch.setattr(
        amd,
        "_detect_from_lspci",
        lambda: ["Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"],
    )

    gpus = amd._detect_amd_gpus_fallback()

    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon RX 6750 XT"
    assert gpus[0].vram_bytes == 12 * _GiB
    assert gpus[0].memory_bandwidth_gbps == 432.0


def test_lspci_enriched_with_sysfs_vram_when_sysfs_detection_fails(
    monkeypatch, tmp_path
):
    """When _detect_from_sysfs returns nothing but _read_sysfs_amd_vram
    succeeds, lspci names should still get VRAM data."""
    _GiB = 1024**3

    # _detect_from_sysfs returns nothing (e.g. product_name absent AND
    # the card dir structure confuses the glob), but individual VRAM reads
    # via _read_sysfs_amd_vram still work.
    monkeypatch.setattr(amd, "_detect_from_sysfs", lambda: [])
    monkeypatch.setattr(
        amd,
        "_detect_from_lspci",
        lambda: ["Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"],
    )
    monkeypatch.setattr(amd, "_read_sysfs_amd_vram", lambda: [12 * _GiB])

    gpus = amd._detect_amd_gpus_fallback()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 12 * _GiB
    assert gpus[0].shared_memory is False


def test_lookup_bandwidth_compound_lspci_name():
    """The bandwidth lookup should resolve compound lspci names by
    splitting on '/' and re-applying the 'RX ' prefix."""
    # Direct substring match works for clean names
    assert amd._lookup_bandwidth("AMD Radeon RX 6750 XT") == 432.0
    assert amd._lookup_bandwidth("AMD Radeon RX 6700 XT") == 384.0

    # Compound lspci name — first matching segment wins
    compound = "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"
    bw = amd._lookup_bandwidth(compound)
    assert bw is not None
    assert bw > 0


def test_display_amd_dgpu_does_not_say_shared_memory(monkeypatch):
    """A discrete AMD GPU with VRAM must NOT display 'shared memory'."""
    from whichllm.output import _console as console_mod
    from whichllm.output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, force_terminal=False))

    display_mod.display_hardware(
        HardwareInfo(
            gpus=[
                GPUInfo(
                    name="AMD Radeon RX 6750 XT",
                    vendor="amd",
                    vram_bytes=12 * 1024**3,
                    memory_bandwidth_gbps=432.0,
                    shared_memory=False,
                )
            ],
            cpu_name="AMD Ryzen 9 5900X",
            cpu_cores=12,
            ram_bytes=128 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            os="linux",
        )
    )

    output = buf.getvalue()
    assert "12.0 GB" in output
    assert "432 GB/s" in output
    assert "shared memory" not in output


def test_display_amd_dgpu_zero_vram_does_not_say_shared_memory(monkeypatch):
    """An AMD dGPU with undetected VRAM should NOT be labelled
    'shared memory' — that would be a false positive."""
    from whichllm.output import _console as console_mod
    from whichllm.output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, force_terminal=False))

    display_mod.display_hardware(
        HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Navi 22 [Radeon RX 6750 XT]",
                    vendor="amd",
                    vram_bytes=0,
                    shared_memory=False,
                )
            ],
            cpu_name="CPU",
            cpu_cores=8,
            ram_bytes=32 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            os="linux",
        )
    )

    output = buf.getvalue()
    assert "shared memory" not in output
