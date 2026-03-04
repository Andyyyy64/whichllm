"""Token generation speed estimation."""

from __future__ import annotations

from local_llm_checker.hardware.types import GPUInfo
from local_llm_checker.models.types import GGUFVariant, ModelInfo


def estimate_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str = "full_gpu",
) -> float:
    """Estimate tokens per second for inference."""
    if gpu is None or fit_type == "cpu_only":
        # CPU-only: very rough estimate based on param count
        params_b = model.parameter_count / 1e9
        if params_b <= 0:
            return 0.0
        # Rough: ~1-3 tok/s for 7B on modern CPU
        return max(0.5, 20.0 / params_b)

    model_size = variant.file_size_bytes if variant else model.parameter_count * 2

    # MoE: only active params need to be read per token
    if model.is_moe and model.parameter_count_active:
        active_ratio = model.parameter_count_active / model.parameter_count
        effective_read = model_size * active_ratio
    else:
        effective_read = model_size

    bandwidth = gpu.memory_bandwidth_gbps * 1e9 if gpu.memory_bandwidth_gbps else 0
    if bandwidth == 0:
        return 0.0

    # Theoretical tok/s = bandwidth / bytes_per_token
    theoretical = bandwidth / effective_read

    # Real-world efficiency: ~40-60% of theoretical
    efficiency = 0.5

    # Partial offload penalty
    if fit_type == "partial_offload":
        efficiency *= 0.5  # significant slowdown from CPU-GPU transfers

    return theoretical * efficiency
