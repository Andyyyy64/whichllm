"""Compatibility checking: can a model run on given hardware?"""

from __future__ import annotations

from dataclasses import dataclass, replace

from whichllm.constants import _GiB
from whichllm.constants import MIN_COMPUTE_CAPABILITY_OLLAMA
from whichllm.constants import VULKAN_ONLY_GPUS
from whichllm.engine.multi_gpu import (
    PER_DEVICE_RUNTIME_BUFFER_BYTES,
    PlacementPlan,
    plan_device_split,
)
from whichllm.engine.placement import (
    Placement,
    cpu_placement,
    gpu_placement,
    plan_expert_offload,
    plan_layer_offload,
)
from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.engine.vram import MemoryBreakdown, estimate_memory_breakdown
from whichllm.hardware.memory import effective_usable_ram
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_MULTI_GPU_HOMOGENEOUS_UTILIZATION = 0.95
_MULTI_GPU_HETEROGENEOUS_UTILIZATION = 0.90
# A layer split hands each token's activations from card to card; the cost is
# small next to reading the weights. Used when a card's bandwidth is unknown,
# the old flat factor applies instead.
_MULTI_GPU_SYNC_FACTOR = 0.95
_MULTI_GPU_UNKNOWN_SPEED_FACTOR = 0.70


def _gpu_available_memory(
    gpu: GPUInfo, usable_ram: int, *, ram_budget_active: bool = False
) -> int:
    vram_bytes = (
        gpu.usable_vram_bytes if gpu.usable_vram_bytes is not None else gpu.vram_bytes
    )
    if gpu.shared_memory and vram_bytes < 2 * _GiB and not gpu.vram_overridden:
        return usable_ram
    if gpu.shared_memory and ram_budget_active:
        return min(vram_bytes, usable_ram)
    return vram_bytes


def _uses_shared_system_pool(gpu: GPUInfo) -> bool:
    return gpu.shared_memory and gpu.vram_bytes < 2 * _GiB and not gpu.vram_overridden


def _is_vulkan_only_gpu(gpu: GPUInfo) -> bool:
    """Return True for legacy NVIDIA GPUs with no modern CUDA support.

    Kepler cards (compute capability 3.x) were dropped by CUDA 12 and current
    llama.cpp CUDA builds, so they only run through the Vulkan backend. Matches
    ``VULKAN_ONLY_GPUS`` entries as case-insensitive substrings of the GPU name,
    the same convention used by the bandwidth/compute-capability lookups.
    """
    if gpu.vendor != "nvidia":
        return False
    name_upper = gpu.name.upper()
    return any(marker.upper() in name_upper for marker in VULKAN_ONLY_GPUS)


def _fit_candidate_gpus(gpus: list[GPUInfo]) -> list[GPUInfo]:
    has_dedicated_gpu = any(
        not _uses_shared_system_pool(gpu) and gpu.vram_bytes > 0 for gpu in gpus
    )
    if not has_dedicated_gpu:
        return gpus
    return [gpu for gpu in gpus if not _uses_shared_system_pool(gpu)]


def _gpu_identity(gpu: GPUInfo) -> str:
    name = gpu.name.lower().replace("(simulated)", "")
    return " ".join(name.split())


def _is_homogeneous_gpu_set(gpus: list[GPUInfo], available: list[int]) -> bool:
    if not gpus:
        return True
    first = gpus[0]
    first_identity = _gpu_identity(first)
    first_available = available[0]
    vram_tolerance = max(256 * 1024**2, int(first_available * 0.02))
    return all(
        gpu.vendor == first.vendor
        and _gpu_identity(gpu) == first_identity
        and abs(gpu_available - first_available) <= vram_tolerance
        for gpu, gpu_available in zip(gpus, available, strict=True)
    )


def _usable_devices(
    gpus: list[GPUInfo], available: list[int]
) -> tuple[list[GPUInfo], list[int], list[str]]:
    """Devices that can pay their own runtime buffer, and the ones that cannot.

    A card below the buffer holds nothing in a split, so it must not weigh on
    the homogeneity check, the speed factor or the pool it stays out of.
    """
    usable_gpus: list[GPUInfo] = []
    usable_available: list[int] = []
    excluded: list[str] = []
    for gpu, gpu_available in zip(gpus, available, strict=False):
        if gpu_available - PER_DEVICE_RUNTIME_BUFFER_BYTES <= 0:
            excluded.append(gpu.name)
            continue
        usable_gpus.append(gpu)
        usable_available.append(gpu_available)
    return usable_gpus, usable_available, excluded


@dataclass(frozen=True)
class _MultiGpuFit:
    """What the fit check uses when more than one dedicated GPU is pooled."""

    fit_vram_available: int
    uses_multi_gpu: bool
    effective_vram: int | None
    plan: PlacementPlan | None
    gpus: list[GPUInfo]
    available: list[int]
    utilization: float

    def gpu_resident_plan(
        self,
        *,
        gpu_weight_bytes: int,
        kv_bytes: int,
        nonsplittable_bytes: int,
    ) -> PlacementPlan | None:
        """Devices the GPU-resident part of a placement is split across."""
        if self.plan is None or not self.plan.devices:
            return None
        return plan_device_split(
            self.gpus,
            self.available,
            gpu_weight_bytes=gpu_weight_bytes,
            kv_bytes=kv_bytes,
            nonsplittable_bytes=nonsplittable_bytes,
            utilization=self.utilization,
        )


def _single_device_fit(
    gpus: list[GPUInfo], available: list[int], utilization: float
) -> _MultiGpuFit:
    """One card carries the GPU part: a plain single-GPU fit, no buffer taken."""
    budget = max(available, default=0)
    return _MultiGpuFit(
        fit_vram_available=budget,
        uses_multi_gpu=False,
        effective_vram=None,
        plan=None,
        gpus=list(gpus),
        available=list(available),
        utilization=utilization,
    )


def _multi_gpu_fit_budget(
    gpus: list[GPUInfo],
    available: list[int],
    warnings: list[str],
    *,
    weights_bytes: int,
    kv_bytes: int,
    nonsplittable_bytes: int,
) -> _MultiGpuFit:
    """Pooled budget for the placement choice, plus the per-device plan."""
    raw_total = sum(available)
    if len(gpus) <= 1:
        return _single_device_fit(gpus, available, 1.0)

    if any(gpu.shared_memory or gpu.vendor == "apple" for gpu in gpus):
        warnings.append(
            "Multiple shared-memory GPUs are not pooled; using the largest "
            "reported memory pool for fit checks"
        )
        return _single_device_fit(gpus, [max(available)], 1.0)

    usable_gpus, usable_available, excluded = _usable_devices(gpus, available)
    if excluded:
        warnings.append(
            "Devices without room for their own runtime buffer stay out of the "
            f"multi-GPU pool: {', '.join(excluded)}"
        )
    if len(usable_gpus) < 2:
        # Nothing left to split across: the fit is the remaining card's own, at
        # its full available memory, exactly as a single-GPU run would be.
        warnings.append(
            "Multi-GPU fit skipped: fewer than two devices have room left after "
            "their own runtime buffer"
        )
        return _single_device_fit(
            usable_gpus or gpus, usable_available or [max(available)], 1.0
        )

    homogeneous = _is_homogeneous_gpu_set(usable_gpus, usable_available)
    utilization = (
        _MULTI_GPU_HOMOGENEOUS_UTILIZATION
        if homogeneous
        else _MULTI_GPU_HETEROGENEOUS_UTILIZATION
    )
    plan = plan_device_split(
        usable_gpus,
        usable_available,
        gpu_weight_bytes=weights_bytes,
        kv_bytes=kv_bytes,
        nonsplittable_bytes=nonsplittable_bytes,
        utilization=utilization,
    )
    if not plan.reserve_fits:
        # Activation buffers and framework overhead cannot be divided by a layer
        # split, and no single device can host them: only one card can carry the
        # GPU part of a partial offload, at its full available memory.
        warnings.append(
            "Multi-GPU placement rejected: the runtime-only part of the estimate "
            f"({nonsplittable_bytes / _GiB:.2f} GB of activation buffers and "
            "framework overhead) does not fit any device after its runtime "
            "buffer, so no layer split can carry it; the GPU part runs on the "
            "largest card alone"
        )
        return _single_device_fit(usable_gpus, usable_available, utilization)

    warnings.append(
        "Multi-GPU fit uses a conservative layer-split budget: "
        f"{plan.pooled_bytes / _GiB:.1f} GB effective from "
        f"{raw_total / _GiB:.1f} GB raw VRAM"
    )
    if not homogeneous:
        warnings.append(
            "Heterogeneous multi-GPU setup: fit assumes uneven layer placement; "
            "speed depends on backend split mode and interconnect"
        )
    return _MultiGpuFit(
        fit_vram_available=plan.pooled_bytes,
        uses_multi_gpu=True,
        effective_vram=plan.pooled_bytes,
        plan=plan,
        gpus=usable_gpus,
        available=usable_available,
        utilization=utilization,
    )


def _attach_device_split(
    placement: Placement | None,
    multi_gpu: _MultiGpuFit,
    memory: MemoryBreakdown,
    warnings: list[str],
) -> Placement | None:
    """Split the GPU-resident part of a placement across the pooled devices."""
    if placement is None or placement.gpu_weight_bytes <= 0:
        return placement
    plan = multi_gpu.gpu_resident_plan(
        gpu_weight_bytes=placement.gpu_weight_bytes,
        kv_bytes=memory.kv_cache,
        nonsplittable_bytes=memory.activation + memory.overhead,
    )
    if plan is None:
        return placement
    return replace(placement, devices=plan.devices)


def _multi_gpu_speed_factor(gpus: list[GPUInfo], available: list[int]) -> float:
    """Decode speed of a layer split relative to reading it all on the best card.

    llama.cpp splits layers in proportion to each card's free memory, and a
    token reads every card's share in turn, so the time is the sum of
    share / bandwidth. Two identical cards run about as fast as one would if
    the model fit; a slower second card drags in proportion to its share.
    """
    bandwidths = [gpu.memory_bandwidth_gbps for gpu in gpus]
    total = sum(available)
    if total <= 0 or not all(bandwidths):
        return _MULTI_GPU_UNKNOWN_SPEED_FACTOR
    split_time = sum(
        share / total / bw for share, bw in zip(available, bandwidths, strict=True)
    )
    return min(1.0, (1.0 / max(bandwidths)) / split_time) * _MULTI_GPU_SYNC_FACTOR


def _expert_offload_note(placement: Placement) -> str:
    if placement.n_cpu_moe is None or placement.n_cpu_moe >= (
        placement.num_layers or 0
    ):
        return "MoE experts stay in system RAM (llama.cpp --cpu-moe)"
    return (
        f"MoE experts of {placement.n_cpu_moe}/{placement.num_layers} layers stay "
        f"in system RAM (llama.cpp --n-cpu-moe {placement.n_cpu_moe})"
    )


def check_compatibility(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    context_length: int = 4096,
) -> CompatibilityResult:
    """Check if a model+variant can run on the given hardware."""
    warnings: list[str] = []

    memory = estimate_memory_breakdown(model, variant, context_length)
    vram_required = memory.total

    usable_ram = effective_usable_ram(hardware.ram_bytes, hardware.ram_budget_bytes)

    # Determine best GPU
    best_gpu: GPUInfo | None = None
    best_gpu_available = 0
    gpu_available_values: list[int] = []
    candidate_gpus = _fit_candidate_gpus(hardware.gpus)
    ram_budget_active = hardware.ram_budget_bytes is not None
    for gpu in candidate_gpus:
        gpu_available = _gpu_available_memory(
            gpu, usable_ram, ram_budget_active=ram_budget_active
        )
        gpu_available_values.append(gpu_available)
        if best_gpu is None or gpu_available > best_gpu_available:
            best_gpu = gpu
            best_gpu_available = gpu_available

    vram_available = sum(gpu_available_values) if gpu_available_values else 0
    multi_gpu_speed_factor = 1.0
    nonsplittable_bytes = memory.activation + memory.overhead
    if len(candidate_gpus) > 1 and best_gpu_available >= vram_required:
        # One card holds the whole model: no split, no split overhead.
        multi_gpu = _MultiGpuFit(
            fit_vram_available=best_gpu_available,
            uses_multi_gpu=False,
            effective_vram=None,
            plan=None,
            gpus=candidate_gpus,
            available=gpu_available_values,
            utilization=1.0,
        )
    else:
        multi_gpu = _multi_gpu_fit_budget(
            candidate_gpus,
            gpu_available_values,
            warnings,
            weights_bytes=memory.weights,
            kv_bytes=memory.kv_cache,
            nonsplittable_bytes=nonsplittable_bytes,
        )
        if multi_gpu.uses_multi_gpu:
            multi_gpu_speed_factor = _multi_gpu_speed_factor(
                multi_gpu.gpus, multi_gpu.available
            )
    fit_vram_available = multi_gpu.fit_vram_available
    uses_multi_gpu = multi_gpu.uses_multi_gpu
    multi_gpu_effective_vram = multi_gpu.effective_vram
    if (
        len(candidate_gpus) > 1
        and not uses_multi_gpu
        and any(gpu.shared_memory or gpu.vendor == "apple" for gpu in candidate_gpus)
    ):
        vram_available = fit_vram_available
    if best_gpu and best_gpu.vendor == "apple":
        # Past Metal's working set, llama.cpp keeps the remaining layers on the
        # CPU; they live in the same unified memory the GPU share comes from.
        offload_ram_available = max(0, usable_ram - fit_vram_available)
    elif best_gpu and best_gpu.shared_memory:
        offload_ram_available = 0
    else:
        offload_ram_available = usable_ram

    # Check compute capability for NVIDIA
    if best_gpu and best_gpu.vendor == "nvidia" and best_gpu.compute_capability:
        if best_gpu.compute_capability < MIN_COMPUTE_CAPABILITY_OLLAMA:
            warnings.append(
                f"Compute capability {best_gpu.compute_capability} is below "
                f"minimum {MIN_COMPUTE_CAPABILITY_OLLAMA} for Ollama"
            )

    # Flag legacy Kepler GPUs that have no CUDA support in modern llama.cpp.
    # They can still run, but only through the Vulkan backend on Linux.
    if best_gpu and _is_vulkan_only_gpu(best_gpu):
        warnings.append(
            "Legacy Kepler GPU: no CUDA support in modern llama.cpp; "
            "use the Vulkan backend (Linux) instead"
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
    unified_memory = bool(
        best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple")
    )
    placement: Placement | None = None
    if fit_vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
        offload_ratio = 0.0
        placement = gpu_placement(model, memory)
    else:
        # On a discrete GPU, a MoE first moves routed experts to RAM: each
        # token then reads only its active experts from there.
        if fit_vram_available > 0 and not unified_memory:
            placement = plan_expert_offload(
                model, memory, fit_vram_available, offload_ram_available
            )
        if placement is not None:
            fit_type = "partial_offload"
            can_run = True
            offload_ratio = placement.cpu_weight_bytes / vram_required
            warnings.append(_expert_offload_note(placement))
        elif (
            fit_vram_available > 0
            and (fit_vram_available + offload_ram_available) >= vram_required
        ):
            fit_type = "partial_offload"
            can_run = True
            placement = plan_layer_offload(model, memory, fit_vram_available)
            offload_ratio = (
                (vram_required - fit_vram_available) / vram_required
                if vram_required > 0
                else 0.0
            )
            offload_pct = offload_ratio * 100
            if unified_memory:
                warnings.append("Will use shared system memory")
            else:
                warnings.append(
                    f"~{offload_pct:.0f}% of layers will be offloaded to CPU RAM"
                )
        elif usable_ram >= vram_required:
            fit_type = "cpu_only"
            can_run = True
            offload_ratio = 0.0
            placement = cpu_placement(model, memory)
            warnings.append("Will run on CPU only (much slower)")
        else:
            fit_type = "cpu_only"
            can_run = False
            offload_ratio = 0.0
            warnings.append("Insufficient memory (GPU VRAM + RAM) to run this model")

    placement = _attach_device_split(placement, multi_gpu, memory, warnings)

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
        uses_multi_gpu=uses_multi_gpu,
        multi_gpu_effective_vram_bytes=multi_gpu_effective_vram,
        multi_gpu_speed_factor=multi_gpu_speed_factor,
        multi_gpu_identical=uses_multi_gpu
        and _is_homogeneous_gpu_set(multi_gpu.gpus, multi_gpu.available),
        warnings=warnings,
        fit_type=fit_type,
        placement=placement,
        context_fits=context_fits,
    )
