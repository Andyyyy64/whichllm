"""Token generation speed estimation."""

from __future__ import annotations

from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.engine.quantization import effective_quant_type
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import GGUFVariant, ModelInfo


# Per-quant efficiency factors applied to the theoretical bandwidth-bound
# tok/s. These reflect empirical llama.cpp / vLLM measurements: 4-bit GGUFs
# achieve the highest fraction of memory-bandwidth-limited theoretical
# throughput because the dequantization kernel is fast and weight reads
# dominate; 8-bit and FP16 drop because more compute is required per byte.
_QUANT_EFFICIENCY: dict[str, float] = {
    "F32": 0.30,
    "F16": 0.40,
    "BF16": 0.40,
    "Q8_0": 0.45,
    "Q6_K": 0.50,
    "Q5_K_M": 0.52,
    "Q5_K_S": 0.52,
    "Q5_0": 0.50,
    "Q4_K_M": 0.55,
    "Q4_K_S": 0.55,
    "Q4_0": 0.53,
    # 4-bit microscaling floats decode through native FP4 tensor-core paths
    # (e.g. Blackwell) where weight reads dominate, so they land in the same
    # high-efficiency band as the best 4-bit GGUF kernels.
    "NVFP4": 0.56,
    "MXFP4": 0.55,
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
    "IQ2_XXS": 0.38,
    "IQ1_M": 0.35,
    "IQ1_S": 0.35,
    "Q2_0": 0.38,
    "Q1_0": 0.32,
    "TQ2_0": 0.35,
    "TQ1_0": 0.32,
}

_DEFAULT_QUANT_EFFICIENCY = 0.45

# Vendor / backend multiplier applied on top of quant efficiency. CUDA on
# modern data-center GPUs is the reference (1.0); Apple's Metal kernel is
# behind on dequantization; ROCm trails further; older CUDA generations
# also drop.
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

_SPEED_CONFIDENCE_RANGE_FACTORS: dict[str, tuple[float, float]] = {
    "high": (0.85, 1.20),
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
    return _QUANT_EFFICIENCY.get(quant.upper(), _DEFAULT_QUANT_EFFICIENCY)


def _moe_effective_read_ratio(model: ModelInfo, gpu: GPUInfo) -> float:
    """Return fraction of stored weights read per generated token for MoE."""
    if not model.is_moe or not model.parameter_count_active:
        return 1.0
    if model.parameter_count <= 0:
        return 1.0

    active_ratio = model.parameter_count_active / model.parameter_count
    if active_ratio <= 0:
        return 1.0

    bandwidth = gpu.memory_bandwidth_gbps or 0.0
    if bandwidth > 0:
        floor = _MOE_MIN_READ_RATIO_AT_REFERENCE * max(
            1.0, bandwidth / _MOE_REFERENCE_BANDWIDTH_GBPS
        )
    else:
        floor = _MOE_MAX_READ_RATIO_FLOOR
    floor = min(_MOE_MAX_READ_RATIO_FLOOR, floor)

    return min(1.0, max(active_ratio, floor))


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

    if gpu is None or fit_type == "cpu_only":
        confidence = "low"
        notes.append(
            "CPU-only speed varies heavily with memory channels and BLAS/kernel path."
        )
    else:
        if not gpu.memory_bandwidth_gbps:
            confidence = "low"
            notes.append(
                "GPU memory bandwidth is unknown, so speed is especially uncertain."
            )

        if fit_type == "partial_offload":
            confidence = "low"
            if gpu.vendor == "apple" or gpu.shared_memory:
                notes.append(
                    "Partial offload on unified memory is backend-sensitive but avoids a PCIe cliff."
                )
            else:
                notes.append(
                    "Partial offload on a discrete GPU depends strongly on PCIe and CPU RAM bandwidth."
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

    if _looks_synthetic_gguf(model, variant):
        confidence = _lower_speed_confidence(confidence, "medium")
        notes.append(
            "This is a synthetic GGUF estimate for an official repo, not a measured GGUF file."
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
) -> float:
    """Estimate tokens per second for inference.

    Model: throughput is bounded by the time it takes to read all weights
    needed per token, multiplied by quant- and backend-specific efficiency
    factors. The default 0.5 efficiency factor used earlier mixed two
    distinct losses (compute kernel quality and offload overhead) into one
    constant — this version separates them so a Q4_K_M model on CUDA scores
    differently from the same model running on Metal or with partial
    offload.
    """
    if gpu is None or fit_type == "cpu_only":
        params_b = model.parameter_count / 1e9
        if model.is_moe and model.parameter_count_active:
            params_b = model.parameter_count_active / 1e9
        if params_b <= 0:
            return 0.0
        # Modern desktop CPUs sustain roughly 4-8 GB/s effective for the
        # bandwidth-bound dequant+matmul loop on a single socket. Quantized
        # 4-bit 7B → ~3.5 GB → ~1-2 tok/s. Approximate with an inverse-size
        # heuristic that gets the right order of magnitude.
        quant_factor = _quant_efficiency(model, variant) / _DEFAULT_QUANT_EFFICIENCY
        return max(0.3, 18.0 / max(params_b, 0.5) * quant_factor)

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

    theoretical = bandwidth / effective_read

    # Real-world efficiency depends on quant kernel and backend.
    efficiency = _quant_efficiency(model, variant) * _backend_factor(gpu)

    # Partial offload penalty depends on the memory architecture:
    #
    # - Discrete GPU (NVIDIA/AMD/Intel): spilled weights live in CPU RAM
    #   and are read across PCIe at ~1/10th of VRAM bandwidth. With ~40%
    #   of the model offloaded the blended throughput lands near 0.45x.
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

    return theoretical * efficiency
