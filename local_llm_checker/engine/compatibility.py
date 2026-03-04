"""Compatibility checking: can a model run on given hardware?"""

from __future__ import annotations

from local_llm_checker.constants import MIN_COMPUTE_CAPABILITY_OLLAMA
from local_llm_checker.engine.types import CompatibilityResult
from local_llm_checker.engine.vram import estimate_vram
from local_llm_checker.hardware.types import GPUInfo, HardwareInfo
from local_llm_checker.models.types import GGUFVariant, ModelInfo


def check_compatibility(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    context_length: int = 4096,
) -> CompatibilityResult:
    """Check if a model+variant can run on the given hardware."""
    warnings: list[str] = []

    vram_required = estimate_vram(model, variant, context_length)

    # Determine best GPU
    best_gpu: GPUInfo | None = None
    total_vram = 0
    for gpu in hardware.gpus:
        total_vram += gpu.vram_bytes
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu

    vram_available = total_vram if total_vram > 0 else 0

    # Check compute capability for NVIDIA
    if best_gpu and best_gpu.vendor == "nvidia" and best_gpu.compute_capability:
        if best_gpu.compute_capability < MIN_COMPUTE_CAPABILITY_OLLAMA:
            warnings.append(
                f"Compute capability {best_gpu.compute_capability} is below "
                f"minimum {MIN_COMPUTE_CAPABILITY_OLLAMA} for Ollama"
            )

    # Check ROCm for AMD
    if best_gpu and best_gpu.vendor == "amd" and hardware.os != "linux":
        warnings.append("ROCm requires Linux for AMD GPU inference")

    # Check Metal for Apple
    if best_gpu and best_gpu.vendor == "apple" and hardware.os != "darwin":
        warnings.append("Metal requires macOS for Apple Silicon inference")

    # Reserve 20% of RAM for OS and other processes
    usable_ram = int(hardware.ram_bytes * 0.80)

    # Determine fit type
    if vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
    elif vram_available > 0 and (vram_available + usable_ram) >= vram_required:
        fit_type = "partial_offload"
        can_run = True
        offload_pct = (
            (vram_required - vram_available) / vram_required * 100
            if vram_required > 0
            else 0
        )
        warnings.append(f"~{offload_pct:.0f}% of layers will be offloaded to CPU RAM")
    elif usable_ram >= vram_required:
        fit_type = "cpu_only"
        can_run = True
        warnings.append("Will run on CPU only (much slower)")
    else:
        fit_type = "cpu_only"
        can_run = False
        warnings.append("Insufficient memory (GPU VRAM + RAM) to run this model")

    # Context length warning
    if context_length > 8192 and model.context_length and model.context_length >= context_length:
        warnings.append(f"Large context ({context_length}) increases VRAM usage significantly")

    # File size vs disk space
    file_size = variant.file_size_bytes if variant else model.parameter_count * 2
    if hardware.disk_free_bytes > 0 and file_size > hardware.disk_free_bytes:
        warnings.append("Insufficient disk space to download this model")
        can_run = False

    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=can_run,
        vram_required_bytes=vram_required,
        vram_available_bytes=vram_available,
        warnings=warnings,
        fit_type=fit_type,
    )
