"""Compatibility checking: can a model run on given hardware?"""

from __future__ import annotations

from whichllm.constants import _GiB
from whichllm.constants import MIN_COMPUTE_CAPABILITY_OLLAMA
from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.engine.vram import estimate_vram
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _gpu_available_memory(gpu: GPUInfo, usable_ram: int) -> int:
    if gpu.shared_memory and gpu.vram_bytes < 2 * _GiB:
        return usable_ram
    return gpu.vram_bytes


def _uses_shared_system_pool(gpu: GPUInfo) -> bool:
    return gpu.shared_memory and gpu.vram_bytes < 2 * _GiB


def _fit_candidate_gpus(gpus: list[GPUInfo]) -> list[GPUInfo]:
    has_dedicated_gpu = any(
        not _uses_shared_system_pool(gpu) and gpu.vram_bytes > 0 for gpu in gpus
    )
    if not has_dedicated_gpu:
        return gpus
    return [gpu for gpu in gpus if not _uses_shared_system_pool(gpu)]


def check_compatibility(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    context_length: int = 4096,
) -> CompatibilityResult:
    """Check if a model+variant can run on the given hardware."""
    warnings: list[str] = []

    vram_required = estimate_vram(model, variant, context_length)

    # Reserve 20% of RAM for OS and other processes
    usable_ram = int(hardware.ram_bytes * 0.80)

    # Determine best GPU
    best_gpu: GPUInfo | None = None
    best_gpu_available = 0
    total_vram = 0
    candidate_gpus = _fit_candidate_gpus(hardware.gpus)
    for gpu in candidate_gpus:
        gpu_available = _gpu_available_memory(gpu, usable_ram)
        total_vram += gpu_available
        if best_gpu is None or gpu_available > best_gpu_available:
            best_gpu = gpu
            best_gpu_available = gpu_available

    vram_available = total_vram if total_vram > 0 else 0
    offload_ram_available = (
        0
        if best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple")
        else usable_ram
    )

    # Check compute capability for NVIDIA
    if best_gpu and best_gpu.vendor == "nvidia" and best_gpu.compute_capability:
        if best_gpu.compute_capability < MIN_COMPUTE_CAPABILITY_OLLAMA:
            warnings.append(
                f"Compute capability {best_gpu.compute_capability} is below "
                f"minimum {MIN_COMPUTE_CAPABILITY_OLLAMA} for Ollama"
            )

    # Check ROCm for AMD. Windows AMD users can still use Vulkan/DirectML
    # backends, so do not label the GPU path as unavailable there.
    if (
        best_gpu
        and best_gpu.vendor == "amd"
        and hardware.os not in ("linux", "windows")
    ):
        warnings.append("ROCm requires Linux for AMD GPU inference")

    # Check Metal for Apple
    if best_gpu and best_gpu.vendor == "apple" and hardware.os != "darwin":
        warnings.append("Metal requires macOS for Apple Silicon inference")

    # Determine fit type
    if vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
        offload_ratio = 0.0
    elif (
        vram_available > 0 and (vram_available + offload_ram_available) >= vram_required
    ):
        fit_type = "partial_offload"
        can_run = True
        offload_ratio = (
            (vram_required - vram_available) / vram_required
            if vram_required > 0
            else 0.0
        )
        offload_pct = offload_ratio * 100
        if best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple"):
            warnings.append("Will use shared system memory")
        else:
            warnings.append(
                f"~{offload_pct:.0f}% of layers will be offloaded to CPU RAM"
            )
    elif usable_ram >= vram_required:
        fit_type = "cpu_only"
        can_run = True
        offload_ratio = 0.0
        warnings.append("Will run on CPU only (much slower)")
    else:
        fit_type = "cpu_only"
        can_run = False
        offload_ratio = 0.0
        warnings.append("Insufficient memory (GPU VRAM + RAM) to run this model")

    # Context length warning
    context_fits = not (
        model.context_length is not None and model.context_length < context_length
    )
    if not context_fits:
        warnings.append(
            f"Model max context {model.context_length} < requested "
            f"{context_length}; runtime will truncate or reject"
        )
    elif (
        context_length > 8192
        and model.context_length
        and model.context_length >= context_length
    ):
        warnings.append(
            f"Large context ({context_length}) increases VRAM usage significantly"
        )

    # File size vs disk space
    file_size = estimate_weight_bytes(model, variant)
    if hardware.disk_free_bytes > 0 and file_size > hardware.disk_free_bytes:
        warnings.append("Insufficient disk space to download this model")
        can_run = False

    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=can_run,
        vram_required_bytes=vram_required,
        vram_available_bytes=vram_available,
        offload_ratio=offload_ratio,
        warnings=warnings,
        fit_type=fit_type,
        context_fits=context_fits,
    )
