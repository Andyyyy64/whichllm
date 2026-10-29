"""NVIDIA Jetson (Tegra) detection.

Before this, whichllm reported "No GPU detected — CPU-only mode" on every
Jetson: Tegra answers NVML_ERROR_NOT_SUPPORTED for nvmlDeviceGetMemoryInfo and
prints "[N/A]" for nvidia-smi's memory.total, and "Orin (nvgpu)" matched no
unified-memory marker, so both detection paths dropped the GPU. Every candidate
was then ranked cpu_only (quality x0.50, sort -6.0, CPU speed heuristic).

Bandwidth evidence, measured on a Jetson Orin NX 16GB (p3767-0000, JetPack
6.2.3 / L4T R36.5.0, nvpmodel MAXN, llama.cpp CUDA a94d563, llama-bench
-ngl 99 -p 512 -n 128 -fa 1 -r 3), sweeping candidate values through
estimate_tok_per_sec over four Q4_K_M models from 4.44 to 8.38 GiB:

    bandwidth   MAPE
     34.0        66.7%
     60.0        41.3%     (what a read-only CUDA kernel achieves here)
     68.0        33.5%
    102.4         1.0%     <- data-sheet peak
    136.5        33.6%
    204.8       100.4%

Per model at 102.4 GB/s: EXAONE-3.5-7.8B 11.65 measured / 11.82 predicted,
Llama-3.1-8B 11.42 / 11.46, Qwen3-8B 11.14 / 11.22, Qwen3-14B 6.36 / 6.26 —
three unrelated families all within 1.6%. Qwen3-8B Q4_K_M reproduced at 11.14
tok/s in a second session four days later.
"""

from __future__ import annotations

import builtins
import subprocess
from types import SimpleNamespace

import pytest

from whichllm.constants import GPU_BANDWIDTH
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.hardware import jetson, nvidia
from whichllm.hardware.gpu_db import resolve_detected_bandwidth
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import GGUFVariant, ModelInfo

# The real device tree of the board these values were measured on.
ORIN_NX_16GB_COMPATIBLE = (
    "nvidia,p3768-0000+p3767-0000\0nvidia,p3767-0000\0nvidia,tegra234\0"
)
# Carrier+module only, with no standalone module entry — this is what actually
# exercises the "+" split.
CARRIER_PLUS_MODULE_ONLY = "nvidia,p3768-0000+p3767-0000\0nvidia,tegra234\0"


@pytest.fixture(autouse=True)
def _isolate_device_tree(monkeypatch, tmp_path):
    """Point device-tree reads at tmp_path and clear the module cache.

    Scoped to this file so it cannot change what the rest of the suite sees.
    Without it these tests would read the host's real device tree, so on a
    Jetson the "not a Jetson" cases would pass vacuously.
    """
    monkeypatch.setattr(jetson, "_COMPATIBLE", tmp_path / "compatible")
    jetson.detect_jetson_module.cache_clear()
    yield
    jetson.detect_jetson_module.cache_clear()


def _write_dt(tmp_path, compatible: str) -> None:
    (tmp_path / "compatible").write_bytes(compatible.encode())
    jetson.detect_jetson_module.cache_clear()


# --- device-tree module identification ---------------------------------------


def test_orin_nx_16gb_identified(tmp_path):
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)
    module = jetson.detect_jetson_module()
    assert module is not None
    assert module.name == "Jetson Orin NX 16GB"
    assert module.part_number == "p3767-0000"
    assert module.soc == "tegra234"


def test_module_resolved_from_carrier_plus_module_entry(tmp_path):
    # Regression guard for the "+" split: without it only the carrier board
    # part number is seen and the module goes unidentified.
    _write_dt(tmp_path, CARRIER_PLUS_MODULE_ONLY)
    module = jetson.detect_jetson_module()
    assert module is not None
    assert module.name == "Jetson Orin NX 16GB"
    assert module.part_number == "p3767-0000"


@pytest.mark.parametrize(
    "part_number, expected",
    [
        ("p3767-0000", "Jetson Orin NX 16GB"),
        ("p3767-0001", "Jetson Orin NX 8GB"),
        ("p3767-0003", "Jetson Orin Nano 8GB"),
        ("p3767-0004", "Jetson Orin Nano 4GB"),
        ("p3767-0005", "Jetson Orin Nano 8GB"),
        ("p3701-0000", "Jetson AGX Orin"),
        ("p3701-0008", "Jetson AGX Orin"),
    ],
)
def test_orin_sku_mapping(tmp_path, part_number, expected):
    _write_dt(tmp_path, f"nvidia,{part_number}\0nvidia,tegra234\0")
    module = jetson.detect_jetson_module()
    assert module is not None and module.name == expected


def test_unrecognised_module_is_not_claimed(tmp_path):
    # A Tegra board whose module we cannot name returns None rather than
    # guessing a product from the SoC generation: tegra210 alone, for example,
    # covers both Jetson Nano and TX1.
    _write_dt(tmp_path, "nvidia,p9999-0000\0nvidia,tegra234\0")
    assert jetson.detect_jetson_module() is None


def test_unknown_soc_returns_none(tmp_path):
    _write_dt(tmp_path, "nvidia,tegra999\0")
    assert jetson.detect_jetson_module() is None


def test_non_tegra_board_is_not_a_jetson(tmp_path):
    _write_dt(tmp_path, "raspberrypi,4-model-b\0brcm,bcm2711\0")
    assert jetson.detect_jetson_module() is None


def test_missing_device_tree_returns_none():
    # The common case: an x86 desktop has no /proc/device-tree at all.
    assert jetson.detect_jetson_module() is None


@pytest.mark.parametrize(
    "compatible",
    [
        "",
        "\0\0\0",
        "nvidia,p3767-0000",  # module but no SoC entry
        "nvidia,tegra234\n\0nvidia,p3767-0000\n\0",  # trailing newlines
        "nvidia, tegra234\0nvidia, p3767-0000\0",  # spaces after the comma
        "tegra234\0",  # no vendor prefix
    ],
)
def test_malformed_device_tree_never_raises(tmp_path, compatible):
    _write_dt(tmp_path, compatible)
    jetson.detect_jetson_module()  # must not raise


def test_non_utf8_device_tree_never_raises(tmp_path):
    (tmp_path / "compatible").write_bytes(b"\xff\xfe nvidia,tegra234\0")
    jetson.detect_jetson_module.cache_clear()
    jetson.detect_jetson_module()  # must not raise


def test_trailing_newline_still_resolves(tmp_path):
    # Guards the fullmatch/strip handling: a stray newline must not silently
    # downgrade an exact module match.
    _write_dt(tmp_path, "nvidia,p3767-0000\n\0nvidia,tegra234\n\0")
    module = jetson.detect_jetson_module()
    assert module is not None and module.name == "Jetson Orin NX 16GB"


# --- Tegra GPU-name predicate ------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Orin (nvgpu)", True),
        ("Tegra X1", True),
        ("NVIDIA Tegra X1", True),  # vendor-prefixed, as NVML/lspci report it
        ("NVIDIA GeForce RTX 4090", False),
        ("NVIDIA H100 PCIe", False),
        ("", False),
    ],
)
def test_is_tegra_gpu_name(name, expected):
    assert jetson.is_tegra_gpu_name(name) is expected


# --- curated registry --------------------------------------------------------


@pytest.mark.parametrize(
    "name, bandwidth",
    [
        ("Jetson AGX Orin", 204.8),
        ("Jetson Orin NX 16GB", 102.4),
        ("Jetson Orin NX 8GB", 102.4),
        ("Jetson Orin Nano 8GB", 68.0),
        ("Jetson Orin Nano 4GB", 34.0),
    ],
)
def test_curated_bandwidth(name, bandwidth):
    assert GPU_BANDWIDTH[name] == bandwidth
    assert resolve_detected_bandwidth(name) == bandwidth


@pytest.mark.parametrize(
    "name, cc",
    [
        ("Jetson Orin NX 16GB", (8, 7)),
        ("Jetson Orin Nano 4GB", (8, 7)),
        ("Jetson AGX Orin", (8, 7)),
    ],
)
def test_compute_capability(name, cc):
    assert nvidia._lookup_compute_capability(name) == cc


# --- detection plumbing ------------------------------------------------------


def _fake_pynvml(memory_supported: bool):
    """Minimal NVML stand-in shaped like the Tegra driver's responses."""

    class NVMLError(Exception):
        pass

    def _not_supported(*_args, **_kwargs):
        raise NVMLError("Not Supported")

    return SimpleNamespace(
        NVMLError=NVMLError,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetName=lambda h: "Orin (nvgpu)",
        nvmlSystemGetDriverVersion=lambda: "540.4.0",
        nvmlSystemGetCudaDriverVersion_v2=lambda: 12060,
        nvmlDeviceGetMemoryInfo=(
            (lambda h: SimpleNamespace(total=16 * 1024**3))
            if memory_supported
            else _not_supported
        ),
        nvmlDeviceGetMaxClockInfo=_not_supported,
    )


def _fake_pynvml_import(monkeypatch, module):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_nvml_memory_not_supported_still_detects_jetson(monkeypatch, tmp_path):
    # The regression this change fixes: NVML raises on the memory query, and the
    # old code re-raised because "Orin (nvgpu)" matched no unified-memory marker.
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)
    _fake_pynvml_import(monkeypatch, _fake_pynvml(memory_supported=False))
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 16 * 1024**3
    )

    gpus = nvidia.detect_nvidia_gpus()

    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "Jetson Orin NX 16GB"
    assert gpu.vendor == "nvidia"
    assert gpu.shared_memory is True
    assert gpu.vram_bytes == 16 * 1024**3  # falls back to system RAM
    assert gpu.compute_capability == (8, 7)
    assert gpu.memory_bandwidth_gbps == 102.4
    assert gpu.cuda_version == "12.6"


def test_unrecognised_jetson_module_keeps_generic_name(monkeypatch, tmp_path):
    # Still detected as a unified-memory GPU — the board is not misread as
    # CPU-only — but no bandwidth is invented for an unidentified module.
    _write_dt(tmp_path, "nvidia,p9999-0000\0nvidia,tegra234\0")
    _fake_pynvml_import(monkeypatch, _fake_pynvml(memory_supported=False))
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 8 * 1024**3
    )

    gpus = nvidia.detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "Orin (nvgpu)"
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 8 * 1024**3
    assert gpus[0].memory_bandwidth_gbps is None


def test_smi_na_memory_is_kept_on_jetson(monkeypatch, tmp_path):
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)
    monkeypatch.setattr(
        nvidia, "_run_smi_query", lambda fields: "Orin (nvgpu), [N/A], [N/A]\n"
    )

    gpus = nvidia._detect_nvidia_gpus_via_smi()

    assert len(gpus) == 1
    assert gpus[0].name == "Jetson Orin NX 16GB"
    assert gpus[0].shared_memory is True
    assert gpus[0].memory_bandwidth_gbps == 102.4


def test_smi_na_memory_still_discarded_off_jetson(monkeypatch, tmp_path):
    # Regression guard: the "[N/A]" tolerance must stay Tegra-specific and not
    # start inventing system-RAM VRAM for an unrelated card that reports N/A.
    _write_dt(tmp_path, "raspberrypi,4-model-b\0brcm,bcm2711\0")
    monkeypatch.setattr(
        nvidia, "_run_smi_query", lambda fields: "NVIDIA GeForce RTX 4090, [N/A]\n"
    )
    assert nvidia._detect_nvidia_gpus_via_smi() == []


def test_desktop_gpu_unaffected_even_on_a_jetson_host(monkeypatch, tmp_path):
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)
    monkeypatch.setattr(
        nvidia,
        "_run_smi_query",
        lambda fields: "NVIDIA GeForce RTX 4090, 24564, 10501\n",
    )

    gpus = nvidia._detect_nvidia_gpus_via_smi()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[0].shared_memory is False
    assert gpus[0].memory_bandwidth_gbps == 1008.0


def test_no_nvidia_tooling_reports_nothing(monkeypatch, tmp_path):
    # Detection still requires the driver to enumerate a GPU. The device tree is
    # firmware data and says nothing about whether CUDA is usable, so a Jetson
    # with a broken or absent driver stack must keep reporting no GPU.
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def missing_smi(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", missing_smi)

    assert nvidia.detect_nvidia_gpus() == []


def test_smi_subprocess_error_path(monkeypatch, tmp_path):
    _write_dt(tmp_path, ORIN_NX_16GB_COMPATIBLE)

    def fail_three_field(fields: str) -> str:
        if "clocks" in fields:
            raise subprocess.CalledProcessError(6, "nvidia-smi")
        return "Orin (nvgpu), [N/A]\n"

    monkeypatch.setattr(nvidia, "_run_smi_query", fail_three_field)
    gpus = nvidia._detect_nvidia_gpus_via_smi()
    assert len(gpus) == 1
    assert gpus[0].name == "Jetson Orin NX 16GB"


# --- measured-speed cross-check ----------------------------------------------


def test_orin_nx_estimate_matches_measured_throughput():
    """102.4 GB/s reproduces the measured decode rate; a much lower value does not.

    Qwen3-8B Q4_K_M on the Orin NX 16GB measured 11.14 tok/s (llama-bench
    tg128, n=3), reproduced to the same 2 d.p. in a second session four days
    apart. The table stores the data-sheet peak, as every other entry here
    does; _QUANT_EFFICIENCY carries the achieved-versus-peak loss.
    """
    model = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="Qwen/Qwen3-8B",
        name="Qwen3-8B",
        parameter_count=8_190_735_360,
    )
    variant = GGUFVariant(
        filename="Qwen3-8B-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=5_027_783_488,  # the measured file on disk
    )

    def gpu(bandwidth: float) -> GPUInfo:
        return GPUInfo(
            name="Jetson Orin NX 16GB",
            vendor="nvidia",
            vram_bytes=16 * 1024**3,
            compute_capability=(8, 7),
            memory_bandwidth_gbps=bandwidth,
            shared_memory=True,
        )

    measured = 11.14
    at_peak = estimate_tok_per_sec(model, variant, gpu(102.4))
    at_achieved = estimate_tok_per_sec(model, variant, gpu(60.0))

    assert at_peak == pytest.approx(measured, rel=0.10)
    assert at_achieved < measured * 0.75
