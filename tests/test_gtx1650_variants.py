"""GTX 1650 GDDR5 vs GDDR6 memory-variant disambiguation.

The GTX 1650 shipped in two memory configurations the driver name and PCI
device id (0x1F82) cannot tell apart: the original GDDR5 (8 Gbps x 128-bit =
128 GB/s) and a later GDDR6 revision (12 Gbps x 128-bit = 192 GB/s). They are
resolved by the max memory clock reported at detection time.

Bandwidth evidence (measured on a GDDR6 board, VBIOS 90.17.4D.00.1E): Qwen3-1.7B
Q4_K_M decodes at 75.4 tok/s clock-locked, matching the 192 GB/s estimate (~78)
and not the GDDR5 128 GB/s estimate (~52).
"""

import subprocess

import pytest

from whichllm.constants import GPU_BANDWIDTH, GPU_MEMORY_CLOCK_VARIANTS
from whichllm.data.gpu import (
    GPU_MEMORY_CLOCK_VARIANTS as DATA_GPU_MEMORY_CLOCK_VARIANTS,
)
from whichllm.hardware import nvidia
from whichllm.hardware.gpu_db import resolve_detected_bandwidth
from whichllm.hardware.types import GPUInfo
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.models.types import GGUFVariant, ModelInfo

_NAME = "NVIDIA GeForce GTX 1650"
GDDR6_CLOCK = 6001.0  # measured on the GDDR6 board
GDDR5_CLOCK = 4001.0  # typical GDDR5 1650


def test_variant_table_present_and_reexported():
    assert "GTX 1650" in GPU_MEMORY_CLOCK_VARIANTS
    # constants is a shim over whichllm.data.gpu.
    assert GPU_MEMORY_CLOCK_VARIANTS is DATA_GPU_MEMORY_CLOCK_VARIANTS
    # Highest threshold first, descending — required for first-match resolution.
    thresholds = [t for t, _bw in GPU_MEMORY_CLOCK_VARIANTS["GTX 1650"]]
    assert thresholds == sorted(thresholds, reverse=True)


def test_curated_default_is_gddr5():
    # The base key stays the conservative GDDR5 value for unknown-clock cases.
    assert GPU_BANDWIDTH["GTX 1650"] == 128.0


def test_gddr6_clock_resolves_to_192():
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3, GDDR6_CLOCK) == 192.0


def test_gddr5_clock_resolves_to_128():
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3, GDDR5_CLOCK) == 128.0


def test_unknown_clock_falls_back_to_curated_default():
    # No clock => identical to pre-change behaviour (GDDR5 default).
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3) == 128.0
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3, None) == 128.0
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3, 0.0) == 128.0


@pytest.mark.parametrize(
    "clock, expected", [(5499.0, 128.0), (5500.0, 192.0), (12000.0, 192.0)]
)
def test_threshold_boundary(clock, expected):
    assert resolve_detected_bandwidth(_NAME, 4 * 1024**3, clock) == expected


def test_non_variant_card_ignores_memory_clock():
    # A single-memory-type card must resolve to its curated value regardless of
    # any clock passed in (clock only disambiguates listed variants).
    assert resolve_detected_bandwidth("NVIDIA GeForce GTX 1660", 6 * 1024**3, 9999.0) == 192.0
    assert resolve_detected_bandwidth("NVIDIA GeForce GTX 1660", 6 * 1024**3, 100.0) == 192.0


def test_gddr6_estimate_scales_with_bandwidth_and_matches_measured():
    # Bandwidth flows linearly into the tok/s estimate; the GDDR6 estimate must
    # exceed the GDDR5 one by the bandwidth ratio, and land near the measured
    # 75.4 tok/s for Qwen3-1.7B Q4_K_M.
    model = ModelInfo(
        id="Qwen/Qwen3-1.7B",
        family_id="Qwen/Qwen3-1.7B",
        name="Qwen3-1.7B",
        parameter_count=1_720_000_000,
    )
    variant = GGUFVariant(
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=1_353_000_000,
    )
    gpu6 = GPUInfo("NVIDIA GeForce GTX 1650", "nvidia", 4 * 1024**3,
                   compute_capability=(7, 5), memory_bandwidth_gbps=192.0)
    gpu5 = GPUInfo("NVIDIA GeForce GTX 1650", "nvidia", 4 * 1024**3,
                   compute_capability=(7, 5), memory_bandwidth_gbps=128.0)
    est6 = estimate_tok_per_sec(model, variant, gpu6)
    est5 = estimate_tok_per_sec(model, variant, gpu5)
    assert est6 > est5
    assert est6 / est5 == pytest.approx(192.0 / 128.0, rel=1e-3)
    # Measured 75.4 tok/s; estimate should be within the "high"-confidence band.
    assert 60.0 <= est6 <= 95.0


# --- nvidia-smi detection-path tests (the plumbing that feeds the resolver) ---


def _smi_bw(monkeypatch, stdout: str) -> float | None:
    monkeypatch.setattr(nvidia, "_run_smi_query", lambda fields: stdout)
    gpus = nvidia._detect_nvidia_gpus_via_smi()
    assert len(gpus) == 1
    return gpus[0].memory_bandwidth_gbps


def test_smi_gddr6_clock_resolves_192(monkeypatch):
    assert _smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, 6001\n") == 192.0


def test_smi_gddr5_clock_resolves_128(monkeypatch):
    assert _smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, 4001\n") == 128.0


def test_smi_na_clock_falls_back_to_curated(monkeypatch):
    # Cards/drivers that don't report the clock emit "[N/A]" (still exit 0).
    assert _smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, [N/A]\n") == 128.0


def test_smi_3field_query_failure_retries_without_clock(monkeypatch):
    # If clocks.max.memory makes the 3-field query fail, detection must retry the
    # 2-field query rather than returning zero GPUs (regression guard).
    def fake_query(fields: str) -> str:
        if "clocks" in fields:
            raise subprocess.CalledProcessError(6, "nvidia-smi")
        return "NVIDIA GeForce GTX 1650, 4096\n"

    monkeypatch.setattr(nvidia, "_run_smi_query", fake_query)
    gpus = nvidia._detect_nvidia_gpus_via_smi()
    assert len(gpus) == 1  # not wiped out
    assert gpus[0].memory_bandwidth_gbps == 128.0  # no clock => curated default


def test_smi_both_queries_fail_returns_empty(monkeypatch):
    def always_fail(fields: str) -> str:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia, "_run_smi_query", always_fail)
    assert nvidia._detect_nvidia_gpus_via_smi() == []
