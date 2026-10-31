"""Token generation speed estimation."""

from __future__ import annotations

import re

from whichllm.engine.moe import active_expert_fraction, routed_expert_fraction
from whichllm.engine.placement import Placement
from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.engine.quantization import effective_quant_type
from whichllm.hardware.gpu_db import _MOBILE_MARKER_RE
from whichllm.hardware.ram_bandwidth import DEFAULT_RAM_BANDWIDTH_GBPS
from whichllm.hardware.types import GPUInfo
from whichllm.models.gguf import quant_table_key
from whichllm.models.types import GGUFVariant, ModelInfo


# Dense GGUF decode time per token: the whole file at this fraction of memory
# bandwidth, plus a fixed overhead. Fitted to 379 llama.cpp results on 194 GPUs,
# from 2 to 131 GiB (tests/data/speed_runs.csv). With each GPU held out of the
# fit, the median error is 9% and 90% of runs land within a third, against 33%
# and 50% for the quant-efficiency table below. The quant makes no measurable
# difference, because weight reads dominate.
_GGUF_READ_EFFICIENCY = 0.96
_GGUF_TOKEN_SECONDS = 0.0024
# The same runs per vendor, with CUDA as the reference. The estimate does not
# know the backend, and Vulkan and ROCm land within a few percent on Radeon.
_GGUF_BACKEND_FACTOR: dict[str, float] = {
    "nvidia": 1.00,
    "amd": 0.90,
    "apple": 0.90,
    "intel": 0.80,
}
# Designs whose compute, not memory, limits decode, measured the same way. These
# replace the vendor factor for NVIDIA and Jetson and multiply it elsewhere. The
# Jetson value comes from one Orin NX, which ran a 5 GB file at 11.14 tok/s.
_GGUF_PRE_AMPERE_FACTOR = 0.75
_GGUF_PRE_PASCAL_FACTOR = 0.55
_GGUF_JETSON_FACTOR = 0.60
_GGUF_GCN_FACTOR = 0.65
_GGUF_ALCHEMIST_FACTOR = 0.50
_GGUF_APPLE_ULTRA_FACTOR = 0.80
_PRE_AMPERE_CAPABILITY = (8, 0)
_PRE_PASCAL_CAPABILITY = (6, 0)
_JETSON_NAME_RE = re.compile(r"jetson|\borin\b|tegra|nvgpu", re.IGNORECASE)
_AMD_GCN_NAME_RE = re.compile(
    r"RX [45]\d\d\b|R[579] \d|R9 (FURY|Nano)|\bVega\b|Pro VII|Radeon VII|Pro V340"
    r"|Pro WX \d|FirePro|Radeon Pro \d{3}X?\b|HD \d",
    re.IGNORECASE,
)
_AMD_RDNA_NAME_RE = re.compile(r"\b(RX|PRO W|PRO R) ?\d{4}\b", re.IGNORECASE)
_AMD_MOBILE_NAME_RE = re.compile(r"\bRX \d{4}[MS]\b", re.IGNORECASE)
_INTEL_ALCHEMIST_NAME_RE = re.compile(r"Arc (Pro )?A\d", re.IGNORECASE)
# HBM cards never reach their nominal bandwidth in decode. The A100 and H100 are
# listed near 2,000 GB/s, and both read at about 1,100.
_GGUF_HBM_BANDWIDTH_GBPS = 1100.0
_HBM_NAME_RE = re.compile(
    r"\b(A100|H100|H200|B100|B200|GH200|P100|V100|GP100|TITAN V|Instinct|CMP)\b",
    re.IGNORECASE,
)

_PUBLISHED_RUNS_NOTE = (
    "Published llama.cpp runs on GPUs of this kind land close to this estimate, "
    "with every layer on the GPU."
)

# Per-quant efficiency factors for MoE and non-GGUF decode, which the runs
# above do not cover. MoE speed is calibrated separately on Qwen3.6-35B-A3B.
_QUANT_EFFICIENCY: dict[str, float] = {
    "F32": 0.30,
    "F16": 0.40,
    "BF16": 0.40,
    "Q8_K": 0.45,
    "Q8_0": 0.45,
    "Q6_K": 0.50,
    "Q5_K": 0.52,
    "Q5_K_M": 0.52,
    "Q5_K_S": 0.52,
    "Q5_0": 0.50,
    "Q5_1": 0.50,
    "Q4_K": 0.55,
    "Q4_K_M": 0.55,
    "Q4_K_S": 0.55,
    "Q4_0": 0.53,
    "Q4_1": 0.53,
    # 4-bit microscaling floats decode through native FP4 tensor-core paths
    # (e.g. Blackwell) where weight reads dominate, so they land in the same
    # high-efficiency band as the best 4-bit GGUF kernels.
    "NVFP4": 0.56,
    "MXFP4": 0.55,
    "Q3_K": 0.50,
    "Q3_K_M": 0.50,
    "Q3_K_S": 0.48,
    "Q3_K_L": 0.50,
    "Q2_K": 0.45,
    "IQ4_XS": 0.52,
    "IQ4_NL": 0.50,
    "IQ3_S": 0.45,
    "IQ3_M": 0.45,
    "IQ3_XS": 0.45,
    "IQ3_XXS": 0.42,
    "IQ2_S": 0.40,
    "IQ2_M": 0.40,
    "IQ2_XS": 0.40,
    "IQ2_XXS": 0.38,
    "IQ1_M": 0.35,
    "IQ1_S": 0.35,
    "Q2_0": 0.38,
    "Q1_0": 0.32,
    "TQ2_0": 0.35,
    "TQ1_0": 0.32,
}

_DEFAULT_QUANT_EFFICIENCY = 0.45

# Vendor / backend multiplier applied on top of quant efficiency for MoE and
# non-GGUF decode. CUDA on modern data-center GPUs is the reference (1.0);
# Apple's Metal kernel is behind on dequantization; ROCm trails further; older
# CUDA generations also drop.
_BACKEND_FACTOR: dict[str, float] = {
    "nvidia": 1.00,
    "amd": 0.78,
    "apple": 0.82,
    "intel": 0.65,
}

# MoE decode is partly bandwidth-bound and partly kernel/dispatch-bound.
# The old fixed 25% read floor matched high-bandwidth CUDA cards reasonably
# well, but badly under-estimated low-bandwidth unified-memory APUs such as
# Strix Halo where the active expert reads dominate. Model this as a floor
# that rises with bandwidth: ~5% at 256 GB/s, capped at the legacy 25%.
_MOE_REFERENCE_BANDWIDTH_GBPS = 256.0
_MOE_MIN_READ_RATIO_AT_REFERENCE = 0.05
_MOE_MAX_READ_RATIO_FLOOR = 0.25

# Fraction of theoretical RAM bandwidth that llama.cpp's CPU decode turns
# into weight reads at the Q4_K_M reference quant; other quants scale by
# their GPU efficiency ratio, since dequantization costs the CPU the same
# way. Whole layers on the CPU (CPU-only runs, layer offload) also run
# attention and the output head there, so they get less of the bandwidth
# than expert FFNs alone. A split run also pays a fixed GPU/CPU sync per
# token.
#
# Calibrated with llama.cpp b11977 on an RTX 5080 with dual-channel
# DDR5-5600 (Core Ultra 9 285K, 89.6 GB/s theoretical), Qwen3.6-35B-A3B
# Q4_K_M, medians of six tg128 runs: --n-cpu-moe 40/30/20/12 gave
# 44/49/57/83 tok/s (0.38 ms per CPU expert layer, ~3 ms fixed), CPU-only
# 13.6 tok/s.
_CPU_DECODE_EFFICIENCY = 0.30
_CPU_EXPERT_EFFICIENCY = 0.43
_CPU_EFFICIENCY_MAX = 0.6
_SPLIT_SYNC_SECONDS = 0.003

# Each range holds at least 90% of the published runs at its confidence level
# (tests/test_speed_calibration.py).
_SPEED_CONFIDENCE_RANGE_FACTORS: dict[str, tuple[float, float]] = {
    "high": (0.75, 1.30),
    "medium": (0.60, 1.60),
    "low": (0.35, 2.00),
}

_SPEED_CONFIDENCE_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _backend_factor(gpu: GPUInfo) -> float:
    if gpu.vendor in _BACKEND_FACTOR:
        return _BACKEND_FACTOR[gpu.vendor]
    return 0.7


def _quant_efficiency(model: ModelInfo, variant: GGUFVariant | None) -> float:
    quant = effective_quant_type(model, variant)
    if not quant:
        return _DEFAULT_QUANT_EFFICIENCY
    key = quant_table_key(quant, _QUANT_EFFICIENCY)
    return _QUANT_EFFICIENCY.get(key, _DEFAULT_QUANT_EFFICIENCY)


def _is_dense_gguf(model: ModelInfo, variant: GGUFVariant | None) -> bool:
    return variant is not None and not model.is_moe


def _gguf_backend_factor(gpu: GPUInfo) -> float:
    if _JETSON_NAME_RE.search(gpu.name):
        return _GGUF_JETSON_FACTOR
    capability = gpu.compute_capability
    if gpu.vendor == "nvidia" and capability:
        if capability < _PRE_PASCAL_CAPABILITY:
            return _GGUF_PRE_PASCAL_FACTOR
        if capability < _PRE_AMPERE_CAPABILITY:
            return _GGUF_PRE_AMPERE_FACTOR
    factor = _GGUF_BACKEND_FACTOR.get(gpu.vendor, 0.7)
    if gpu.vendor == "amd" and _AMD_GCN_NAME_RE.search(gpu.name):
        return factor * _GGUF_GCN_FACTOR
    if gpu.vendor == "intel" and _INTEL_ALCHEMIST_NAME_RE.search(gpu.name):
        return factor * _GGUF_ALCHEMIST_FACTOR
    if gpu.vendor == "apple" and "ultra" in gpu.name.lower():
        return factor * _GGUF_APPLE_ULTRA_FACTOR
    return factor


def _gguf_read_bandwidth(gpu: GPUInfo, bandwidth: float) -> float:
    if _HBM_NAME_RE.search(gpu.name):
        return min(bandwidth, _GGUF_HBM_BANDWIDTH_GBPS * 1e9)
    return bandwidth


def _active_ratio(model: ModelInfo) -> float:
    """Share of all stored weights a MoE reads per token (1.0 for dense)."""
    if not model.is_moe or not model.parameter_count_active:
        return 1.0
    if model.parameter_count <= 0:
        return 1.0
    active_ratio = model.parameter_count_active / model.parameter_count
    return min(1.0, active_ratio) if active_ratio > 0 else 1.0


def _moe_read_floor(gpu: GPUInfo) -> float:
    """Dispatch/kernel overhead of GPU MoE decode, as a read-ratio floor."""
    bandwidth = gpu.memory_bandwidth_gbps or 0.0
    if bandwidth > 0:
        floor = _MOE_MIN_READ_RATIO_AT_REFERENCE * max(
            1.0, bandwidth / _MOE_REFERENCE_BANDWIDTH_GBPS
        )
    else:
        floor = _MOE_MAX_READ_RATIO_FLOOR
    return min(_MOE_MAX_READ_RATIO_FLOOR, floor)


def _moe_effective_read_ratio(model: ModelInfo, gpu: GPUInfo) -> float:
    """Return fraction of stored weights read per generated token for MoE."""
    active_ratio = _active_ratio(model)
    if active_ratio >= 1.0:
        return 1.0
    return min(1.0, max(active_ratio, _moe_read_floor(gpu)))


def _cpu_decode_bytes_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    ram_bandwidth_gbps: float | None,
    efficiency: float = _CPU_DECODE_EFFICIENCY,
) -> float:
    efficiency = min(
        efficiency * _quant_efficiency(model, variant) / _QUANT_EFFICIENCY["Q4_K_M"],
        _CPU_EFFICIENCY_MAX,
    )
    return (ram_bandwidth_gbps or DEFAULT_RAM_BANDWIDTH_GBPS) * 1e9 * efficiency


def _split_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo,
    placement: Placement,
    gpu_bytes_per_sec: float,
    ram_bandwidth_gbps: float | None,
) -> float:
    """Decode speed when the GPU and the CPU each hold part of the weights.

    Per token, the GPU reads its share at VRAM speed and the CPU reads its
    share from RAM; the two run one after the other, so the times add up.
    With expert offload the CPU share is only the active experts.
    """
    model_size = estimate_weight_bytes(model, variant)
    if placement.kind == "expert_offload":
        experts = model_size * routed_expert_fraction(model)
        active = active_expert_fraction(model)
        gpu_experts = max(experts - placement.cpu_weight_bytes, 0.0)
        gpu_read = (model_size - experts) + gpu_experts * active
        cpu_read = placement.cpu_weight_bytes * active
        cpu_efficiency = _CPU_EXPERT_EFFICIENCY
        sync = _SPLIT_SYNC_SECONDS
    else:
        ratio = _active_ratio(model)
        gpu_read = placement.gpu_weight_bytes * ratio
        cpu_read = placement.cpu_weight_bytes * ratio
        cpu_efficiency = _CPU_DECODE_EFFICIENCY
        sync = _GGUF_TOKEN_SECONDS if _is_dense_gguf(model, variant) else 0.0
    if model.is_moe:
        gpu_read = max(gpu_read, placement.gpu_weight_bytes * _moe_read_floor(gpu))

    seconds = (
        gpu_read / gpu_bytes_per_sec
        + cpu_read
        / _cpu_decode_bytes_per_sec(model, variant, ram_bandwidth_gbps, cpu_efficiency)
        + sync
    )
    return 1.0 / seconds if seconds > 0 else 0.0


def _dense_gguf_confidence(gpu: GPUInfo) -> tuple[str, str | None]:
    """How well published runs back a dense GGUF estimate on this GPU."""
    if _JETSON_NAME_RE.search(gpu.name) or (
        gpu.shared_memory and gpu.vendor != "apple"
    ):
        return "medium", None
    if _MOBILE_MARKER_RE.search(gpu.name) or _AMD_MOBILE_NAME_RE.search(gpu.name):
        return "low", "Laptop GPUs vary with their power limit, and few runs exist."
    if _HBM_NAME_RE.search(gpu.name):
        return "low", "HBM GPUs read well below their listed bandwidth, and it varies."
    if gpu.vendor == "apple":
        return "high", _PUBLISHED_RUNS_NOTE
    if gpu.vendor == "intel":
        return "low", "Intel speed varies with driver and backend, and few runs exist."
    if gpu.vendor == "amd":
        if _AMD_RDNA_NAME_RE.search(gpu.name):
            return "high", _PUBLISHED_RUNS_NOTE
        if _AMD_GCN_NAME_RE.search(gpu.name):
            return "low", "GCN Radeon cards vary too much between runs to narrow this."
    capability = gpu.compute_capability
    if gpu.vendor == "nvidia" and capability:
        if capability >= _PRE_PASCAL_CAPABILITY:
            return "high", _PUBLISHED_RUNS_NOTE
        return "low", "NVIDIA cards before Pascal vary too much to narrow this."
    return "medium", None


def _lower_speed_confidence(current: str, candidate: str) -> str:
    if _SPEED_CONFIDENCE_ORDER[candidate] < _SPEED_CONFIDENCE_ORDER[current]:
        return candidate
    return current


def _looks_synthetic_gguf(model: ModelInfo, variant: GGUFVariant | None) -> bool:
    if variant is None:
        return False
    if not variant.filename:
        return False
    expected = f"{model.name}.{variant.quant_type}.gguf"
    return variant.filename == expected


def estimate_speed_uncertainty(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str,
    estimated_tok_per_sec: float | None,
    *,
    placement: Placement | None = None,
    ram_bandwidth_source: str = "default",
    uses_multi_gpu: bool = False,
    identical_gpus: bool = False,
) -> tuple[str, tuple[float, float] | None, list[str]]:
    """Return confidence metadata for the speed point estimate.

    The tok/s estimator is intentionally hardware/model-metadata based; it
    does not know the user's exact llama.cpp, Vulkan, ROCm, Metal, MLX, or
    runtime kernel versions. This helper keeps that uncertainty visible
    without mixing it into the ranking score itself.
    """
    notes = [
        "Speed is estimated from memory bandwidth, quantization, backend, and fit type."
    ]
    confidence = "medium"

    if estimated_tok_per_sec is None or estimated_tok_per_sec <= 0:
        return (
            "low",
            None,
            notes + ["No usable bandwidth estimate was available for this setup."],
        )

    ram_known = ram_bandwidth_source in ("detected", "unified", "override")
    if gpu is None or fit_type == "cpu_only":
        confidence = "medium" if ram_known else "low"
        notes.append(
            "CPU-only speed is bound by system RAM bandwidth"
            + ("" if ram_known else ", which is estimated here")
            + "."
        )
    else:
        if not gpu.memory_bandwidth_gbps:
            confidence = "low"
            notes.append(
                "GPU memory bandwidth is unknown, so speed is especially uncertain."
            )

        if placement is not None and placement.kind == "expert_offload":
            if not ram_known:
                confidence = "low"
            notes.append(
                "MoE experts in system RAM run on the CPU; speed depends on RAM "
                "bandwidth"
                + ("" if ram_known else " (estimated here; set --ram-bandwidth)")
                + "."
            )
        elif fit_type == "partial_offload":
            confidence = "low"
            if gpu.vendor == "apple" or gpu.shared_memory:
                notes.append(
                    "Partial offload on unified memory is backend-sensitive but avoids a PCIe cliff."
                )
            else:
                notes.append(
                    "Layers offloaded from a discrete GPU run on the CPU at RAM bandwidth."
                )

        if model.is_moe:
            notes.append(
                "MoE speed uses active parameters plus a bandwidth-scaled dispatch/read floor."
            )
            if gpu.vendor == "apple":
                confidence = _lower_speed_confidence(confidence, "low")
                notes.append(
                    "Apple Silicon MoE throughput is especially sensitive to Metal/MLX runtime kernels."
                )
            elif gpu.vendor == "amd" and gpu.shared_memory:
                confidence = _lower_speed_confidence(confidence, "medium")
                notes.append(
                    "AMD shared-memory APU estimates are calibrated by bandwidth, but ROCm/Vulkan kernels can differ."
                )
        elif (
            fit_type == "full_gpu"
            and gpu.memory_bandwidth_gbps
            and _is_dense_gguf(model, variant)
        ):
            confidence, note = _dense_gguf_confidence(gpu)
            if note:
                notes.append(note)

    if variant is not None and variant.is_estimated:
        confidence = _lower_speed_confidence(confidence, "medium")
        if _looks_synthetic_gguf(model, variant):
            notes.append(
                "This is a synthetic GGUF estimate for an official repo, not a measured GGUF file."
            )
        else:
            notes.append("The size of this GGUF file is estimated.")

    if uses_multi_gpu:
        if identical_gpus and _is_dense_gguf(model, variant):
            confidence = _lower_speed_confidence(confidence, "medium")
            notes.append(
                "Published layer splits across identical cards stay close to this estimate."
            )
        else:
            confidence = "low"
            notes.append(
                "Multi-GPU speed depends on layer/tensor split mode, "
                "PCIe/NVLink bandwidth, and backend support; this estimate "
                "does not assume ideal scaling."
            )

    low_factor, high_factor = _SPEED_CONFIDENCE_RANGE_FACTORS[confidence]
    speed_range = (
        round(estimated_tok_per_sec * low_factor, 1),
        round(estimated_tok_per_sec * high_factor, 1),
    )
    return confidence, speed_range, notes


def estimate_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str = "full_gpu",
    *,
    placement: Placement | None = None,
    ram_bandwidth_gbps: float | None = None,
) -> float:
    """Estimate tokens per second for inference.

    Model: throughput is bounded by the time it takes to read all weights
    needed per token at the backend's share of memory bandwidth, plus a fixed
    per-token overhead for dense GGUF models.

    With a ``placement`` on a discrete GPU, the part of the weights kept in
    system RAM is read at ``ram_bandwidth_gbps`` by the CPU. CPU-only runs
    read everything from RAM.
    """
    if gpu is None or fit_type == "cpu_only":
        read = estimate_weight_bytes(model, variant) * _active_ratio(model)
        if read <= 0:
            return 0.0
        return _cpu_decode_bytes_per_sec(model, variant, ram_bandwidth_gbps) / read

    model_size = estimate_weight_bytes(model, variant)

    # MoE: use a speed-specific effective read ratio. VRAM fit still uses
    # total stored weights elsewhere; this only estimates per-token reads.
    if model.is_moe and model.parameter_count_active:
        effective_read = model_size * _moe_effective_read_ratio(model, gpu)
    else:
        effective_read = model_size

    bandwidth = gpu.memory_bandwidth_gbps * 1e9 if gpu.memory_bandwidth_gbps else 0
    if bandwidth == 0:
        return 0.0

    dense_gguf = _is_dense_gguf(model, variant)
    if dense_gguf:
        bandwidth = _gguf_read_bandwidth(gpu, bandwidth)
        efficiency = _GGUF_READ_EFFICIENCY * _gguf_backend_factor(gpu)
    else:
        efficiency = _quant_efficiency(model, variant) * _backend_factor(gpu)

    unified_memory = gpu.vendor == "apple" or gpu.shared_memory
    if (
        fit_type == "partial_offload"
        and placement is not None
        and placement.kind in ("expert_offload", "layer_offload")
        and not unified_memory
    ):
        return _split_tok_per_sec(
            model,
            variant,
            gpu,
            placement,
            bandwidth * efficiency,
            ram_bandwidth_gbps,
        )

    # Partial offload penalty depends on the memory architecture:
    #
    # - Discrete GPU without a placement (the plan command's GPU table):
    #   spilled weights are read from CPU RAM at a fraction of VRAM
    #   bandwidth. With ~40% of the model offloaded the blended throughput
    #   lands near 0.45x.
    # - Apple Silicon: GPU and CPU share one physical unified-memory pool.
    #   AMD shared-memory APUs such as Strix Halo have the same no-PCIe-cliff
    #   shape for model weights, even though their backend factor remains AMD.
    #   "Exceeding VRAM" only means exceeding the recommended working set;
    #   the bytes are still read from the same high-bandwidth unified RAM,
    #   so there is no PCIe cliff — only mild OS/cache contention. Using
    #   the discrete 0.45x here was the bug that made DeepSeek-R1-class
    #   models on M2/M3 Ultra report ~1.7 t/s when real-world is 4-15.
    if fit_type == "partial_offload":
        if gpu.vendor == "apple" or gpu.shared_memory:
            efficiency *= 0.85
        else:
            efficiency *= 0.45

    seconds = effective_read / (bandwidth * efficiency)
    if dense_gguf:
        seconds += _GGUF_TOKEN_SECONDS
    return 1.0 / seconds
