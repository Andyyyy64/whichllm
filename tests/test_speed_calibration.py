"""Speed estimates against llama.cpp runs that someone actually measured.

``RUNS`` takes its bandwidth from the simulator, so a wrong GPU entry fails here
too. ``data/speed_runs.csv`` holds every published result behind the fit, one
median per GPU, file size and backend, with all layers on the GPU. A run stayed
in only if its GPU name matched the simulator's exactly. Macs carry their GPU
core count, which sets the bandwidth on the cut-down Max chips.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from whichllm.engine.performance import (
    estimate_speed_uncertainty,
    estimate_tok_per_sec,
)
from whichllm.hardware.apple import apple_bandwidth
from whichllm.hardware.gpu_simulator import create_synthetic_gpu
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3
_APPLE = "https://github.com/ggml-org/llama.cpp/discussions/4167"
_VULKAN = "https://github.com/ggml-org/llama.cpp/discussions/10879"
_INTEL = "https://github.com/ggml-org/llama.cpp/discussions/23313"
_SPARK = "https://github.com/ggml-org/llama.cpp/discussions/15013"
_STRIX = "https://github.com/visorcraft/strix-halo-llm-perf/blob/main/BENCHMARKS.md"
_LOCAL = "whichllm RTX 5080, llama-server b11977"
_ORIN = "tests/test_jetson_detection.py"
_TABLE = Path(__file__).parent / "data" / "speed_runs.csv"

# gpu, memory GB, parameters in billions, file GiB, quant, tok/s, source
RUNS = [
    ("RTX 5080", None, 27.8, 12.48, "Q3_K_M", 55.42, _LOCAL),
    ("Jetson Orin NX", 16, 8.2, 4.68, "Q4_K_M", 11.14, _ORIN),
    ("DGX Spark", 128, 7.0, 3.56, "Q4_0", 57.21, _SPARK),
    ("RX 7900 XTX", None, 7.0, 3.56, "Q4_0", 182.63, _VULKAN),
    ("RX 7900 XTX", None, 27.0, 16.04, "Q4_0", 35.25, _VULKAN),
    ("RX 7600", None, 7.0, 3.56, "Q4_0", 58.03, _VULKAN),
    ("RX 6800", None, 7.0, 3.56, "Q4_0", 101.47, _VULKAN),
    ("Ryzen AI MAX+ 395", 128, 6.7, 3.80, "Q4_K_M", 47.25, _STRIX),
    ("Ryzen AI MAX+ 395", 128, 14.8, 8.37, "Q4_K_M", 24.47, _STRIX),
    ("Ryzen AI MAX+ 395", 128, 32.8, 18.48, "Q4_K_M", 11.25, _STRIX),
    ("Ryzen AI MAX+ 395", 128, 70.6, 39.59, "Q4_K_M", 5.10, _STRIX),
    ("Arc B580", None, 7.0, 3.56, "Q4_0", 73.76, _INTEL),
    ("M1", 16, 7.0, 3.56, "Q4_0", 14.13, _APPLE),
    ("M1 Pro", 16, 7.0, 3.56, "Q4_0", 35.52, _APPLE),
    ("M1 Max", 64, 7.0, 3.56, "Q4_0", 61.19, _APPLE),
    ("M1 Ultra", 128, 7.0, 3.53, "Q4_0", 113.79, _APPLE),
    ("M2", 24, 7.0, 3.56, "Q4_0", 21.91, _APPLE),
    ("M2 Pro", 32, 7.0, 3.56, "Q4_0", 38.86, _APPLE),
    ("M2 Max", 96, 7.0, 3.56, "Q4_0", 64.55, _APPLE),
    ("M2 Ultra", 192, 7.0, 3.56, "Q4_0", 125.21, _APPLE),
    ("M3", 16, 7.0, 3.56, "Q4_0", 21.34, _APPLE),
    ("M3 Pro", 36, 7.0, 3.56, "Q4_0", 30.74, _APPLE),
    ("M3 Max", 36, 7.0, 3.56, "Q4_0", 56.85, _APPLE),
    ("M3 Ultra", 512, 7.0, 3.56, "Q4_0", 92.14, _APPLE),
    ("M4", 24, 7.6, 4.36, "Q4_K_M", 22.60, _APPLE),
    ("M4 Pro", 24, 7.0, 3.56, "Q4_0", 50.74, _APPLE),
    ("M4 Max", 128, 7.0, 3.56, "Q4_0", 83.06, _APPLE),
    ("M4 Max", 36, 27.0, 16.04, "Q4_0", 18.71, _APPLE),
]


def _model(params_b, file_gib, quant) -> tuple[ModelInfo, GGUFVariant]:
    variant = GGUFVariant(f"model-{quant}.gguf", quant, int(file_gib * _GiB))
    model = ModelInfo(
        id="org/Model-GGUF",
        family_id="model",
        name="Model",
        parameter_count=int(params_b * 1e9),
        gguf_variants=[variant],
    )
    return model, variant


def _estimate(gpu_name, memory_gb, params_b, file_gib, quant) -> float:
    model, variant = _model(params_b, file_gib, quant)
    gpu = create_synthetic_gpu(gpu_name, memory_gb)
    return estimate_tok_per_sec(model, variant, gpu, "full_gpu")


def _table() -> list[dict]:
    with _TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        capability = row["compute_capability"]
        gpu = GPUInfo(
            name=row["gpu"],
            vendor=row["vendor"],
            vram_bytes=128 * _GiB,
            memory_bandwidth_gbps=float(row["bandwidth_gbps"]),
            compute_capability=(
                tuple(int(part) for part in capability.split("."))
                if capability
                else None
            ),
        )
        model, variant = _model(7.0, float(row["file_gib"]), "Q4_K_M")
        estimate = estimate_tok_per_sec(model, variant, gpu, "full_gpu")
        measured = float(row["tok_per_sec"])
        confidence, speed_range, _ = estimate_speed_uncertainty(
            model, variant, gpu, "full_gpu", estimate
        )
        row["log_ratio"] = math.log(estimate / measured)
        row["confidence"] = confidence
        row["in_range"] = speed_range[0] <= measured <= speed_range[1]
    return rows


@pytest.fixture(scope="module")
def table() -> list[dict]:
    return _table()


@pytest.mark.parametrize(
    ("gpu", "memory_gb", "params_b", "file_gib", "quant", "measured", "source"),
    RUNS,
    ids=[f"{run[0]}-{run[3]}G-{run[4]}" for run in RUNS],
)
def test_estimate_is_within_a_third_of_the_run(
    gpu, memory_gb, params_b, file_gib, quant, measured, source
):
    estimate = _estimate(gpu, memory_gb, params_b, file_gib, quant)

    assert measured * 0.67 <= estimate <= measured * 1.33, (
        f"{gpu}: estimated {estimate:.1f}, measured {measured} ({source})"
    )


def test_the_quant_does_not_change_the_read_rate():
    # DGX Spark, Qwen2.5-7B Q8_0: 29.43 tok/s from a 7.54 GiB file, in line with
    # the Q4_0 run above once the larger file is accounted for.
    q8 = _estimate("DGX Spark", 128, 7.0, 7.54, "Q8_0")

    assert q8 == pytest.approx(29.43, rel=0.1)


@pytest.mark.parametrize(
    ("chip", "memory_gb", "bandwidth"),
    [
        ("M3 Max", 36, 300.0),
        ("M3 Max", 48, 400.0),
        ("M3 Max", 96, 300.0),
        ("M3 Max", 128, 400.0),
        ("M4 Max", 36, 410.0),
        ("M4 Max", 48, 546.0),
        ("M5 Max", 36, 460.0),
        ("M5 Max", 64, 614.0),
    ],
)
def test_memory_size_identifies_cut_down_max_chips(chip, memory_gb, bandwidth):
    simulated = create_synthetic_gpu(chip, memory_gb)

    assert simulated.memory_bandwidth_gbps == bandwidth
    assert apple_bandwidth(f"Apple {chip}", memory_gb * _GiB) == bandwidth


def test_published_runs_stay_close_overall(table):
    errors = [abs(math.exp(row["log_ratio"]) - 1) for row in table]

    assert len(table) >= 370
    assert statistics.median(errors) <= 0.10
    assert sum(error <= 0.33 for error in errors) / len(errors) >= 0.88


@pytest.mark.parametrize("vendor", ["nvidia", "amd", "apple", "intel"])
def test_no_vendor_is_biased(table, vendor):
    ratios = [row["log_ratio"] for row in table if row["vendor"] == vendor]

    assert 0.9 <= math.exp(statistics.median(ratios)) <= 1.1


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(GeForce (GTX|RTX 20)|Tesla [KPM]\d|Quadro [KPM]\d)",
        r"^(GeForce GTX (7|9)\d\d|Tesla [KM]\d|Quadro [KM]\d)",
        r"^(Radeon RX [45]\d\d|Radeon R[79] |FirePro)",
        r"^Arc A\d",
        r"^Apple M\d Ultra",
        r"\b(A100|H100|V100|P100|CMP)\b",
    ],
    ids=["pre-ampere", "pre-pascal", "gcn", "alchemist", "ultra", "hbm"],
)
def test_no_older_or_unusual_design_is_biased(table, pattern):
    ratios = [row["log_ratio"] for row in table if re.search(pattern, row["gpu"])]

    assert len(ratios) >= 4
    assert 0.8 <= math.exp(statistics.median(ratios)) <= 1.25


@pytest.mark.parametrize(("level", "points"), [("high", 250), ("low", 90)])
def test_each_speed_range_holds_the_published_runs(table, level, points):
    rows = [row for row in table if row["confidence"] == level]

    assert len(rows) >= points
    assert sum(row["in_range"] for row in rows) / len(rows) >= 0.9


@pytest.mark.parametrize(
    ("gpu", "memory_gb", "expected"),
    [
        ("RTX 4090", None, "high"),
        ("GTX 1080", None, "high"),
        ("RX 7900 XTX", None, "high"),
        ("M1", 16, "high"),
        ("M2 Ultra", 192, "high"),
        ("Ryzen AI MAX+ 395", 128, "medium"),
        ("DGX Spark", 128, "medium"),
        ("Jetson Orin NX", 16, "medium"),
        ("RTX 4060 Laptop GPU", None, "low"),
        ("RX 7600M XT", None, "low"),
        ("GTX 970", None, "low"),
        ("A100 PCIe 80 GB", None, "low"),
        ("RX 580", None, "low"),
        ("Arc B580", None, "low"),
    ],
)
def test_speed_confidence_follows_the_published_runs(gpu, memory_gb, expected):
    model, variant = _model(7.0, 3.56, "Q4_K_M")
    simulated = create_synthetic_gpu(gpu, memory_gb)

    confidence, _, _ = estimate_speed_uncertainty(
        model, variant, simulated, "full_gpu", 50.0
    )

    assert confidence == expected


def test_only_a_dense_gguf_with_a_real_file_size_on_the_gpu_reaches_high():
    model, variant = _model(7.0, 3.56, "Q4_K_M")
    gpu = create_synthetic_gpu("RTX 4090")
    estimated = replace(variant, is_estimated=True)
    # An official repo's quant takes the real size of the upload it resolves to.
    resolved = replace(variant, filename="Model.Q4_K_M.gguf")
    moe = replace(model, is_moe=True, parameter_count_active=int(1e9))

    def confidence(*args) -> str:
        return estimate_speed_uncertainty(*args, 50.0)[0]

    assert confidence(model, variant, gpu, "full_gpu") == "high"
    assert confidence(model, resolved, gpu, "full_gpu") == "high"
    assert confidence(model, estimated, gpu, "full_gpu") == "medium"
    assert confidence(moe, variant, gpu, "full_gpu") == "medium"
    assert confidence(model, None, gpu, "full_gpu") == "medium"
    assert confidence(model, variant, gpu, "partial_offload") == "low"
    assert confidence(model, variant, None, "cpu_only") == "low"


def test_multi_gpu_confidence_depends_on_the_cards_and_the_model():
    model, variant = _model(7.0, 3.56, "Q4_K_M")
    gpu = create_synthetic_gpu("RTX 4090")
    moe = replace(model, is_moe=True, parameter_count_active=int(1e9))

    def confidence(checked, identical: bool) -> str:
        return estimate_speed_uncertainty(
            checked,
            variant,
            gpu,
            "full_gpu",
            50.0,
            uses_multi_gpu=True,
            identical_gpus=identical,
        )[0]

    assert confidence(model, True) == "medium"
    assert confidence(model, False) == "low"
    assert confidence(moe, True) == "low"


@pytest.mark.parametrize(
    ("low", "high"), [(2, 5), (5, 10), (10, 20), (20, 140)], ids=str
)
def test_no_file_size_is_biased(table, low, high):
    ratios = [row["log_ratio"] for row in table if low <= float(row["file_gib"]) < high]

    assert 0.9 <= math.exp(statistics.median(ratios)) <= 1.1
