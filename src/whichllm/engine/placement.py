"""Where a model's weights live when it does not fit in VRAM.

Two layouts are possible on a discrete GPU, and they differ a lot in speed:

- **Expert offload** (llama.cpp ``--n-cpu-moe N``): attention, shared
  experts, the KV cache and the routed experts of the later layers stay on
  the GPU; the routed experts of the first ``N`` layers stay in system RAM
  and run on the CPU. Each token reads only its few active experts from RAM,
  so a 120B MoE stays usable on a 16 GB card.
- **Layer offload** (``-ngl``): whole layers run on the CPU and every one of
  their weights is read from RAM for each token.
"""

from __future__ import annotations

from dataclasses import dataclass

from whichllm.engine.moe import moe_layer_count, routed_expert_params
from whichllm.engine.multi_gpu import DevicePlacement
from whichllm.engine.vram import MemoryBreakdown
from whichllm.models.types import ModelInfo


@dataclass(frozen=True)
class Placement:
    kind: str  # "gpu" | "expert_offload" | "layer_offload" | "cpu"
    gpu_weight_bytes: int
    cpu_weight_bytes: int
    # llama.cpp --n-cpu-moe: the first N layers keep their routed experts in
    # RAM. None with expert offload means every expert (--cpu-moe).
    n_cpu_moe: int | None = None
    # llama.cpp -ngl; None when the layer count is unknown.
    n_gpu_layers: int | None = None
    num_layers: int | None = None
    # Per-device share of the GPU-resident part. Empty unless the fit pooled
    # more than one dedicated GPU; see whichllm.engine.multi_gpu.
    devices: tuple[DevicePlacement, ...] = ()


def _num_layers(model: ModelInfo) -> int | None:
    return model.shape.num_layers if model.shape else None


def gpu_placement(model: ModelInfo, memory: MemoryBreakdown) -> Placement:
    return Placement(
        kind="gpu",
        gpu_weight_bytes=memory.weights,
        cpu_weight_bytes=0,
        n_gpu_layers=_num_layers(model),
        num_layers=_num_layers(model),
    )


def cpu_placement(model: ModelInfo, memory: MemoryBreakdown) -> Placement:
    return Placement(
        kind="cpu",
        gpu_weight_bytes=0,
        cpu_weight_bytes=memory.weights,
        n_gpu_layers=0,
        num_layers=_num_layers(model),
    )


def plan_expert_offload(
    model: ModelInfo,
    memory: MemoryBreakdown,
    gpu_budget: int,
    ram_budget: int,
) -> Placement | None:
    """Fewest MoE layers whose experts must move to RAM for the rest to fit.

    None when the model has no routed experts, when the dense weights, KV
    cache and buffers alone overflow the GPU, or when the experts that must
    move do not fit in RAM.
    """
    if memory.expert_weights <= 0:
        return None
    resident = memory.total - memory.expert_weights
    if resident > gpu_budget:
        return None
    room = gpu_budget - resident

    shape = model.shape
    layers = moe_layer_count(shape) if shape and routed_expert_params(model) else 0
    if shape and layers:
        per_layer = memory.expert_weights / layers
        cpu_layers = layers - min(int(room // per_layer), layers)
        if cpu_layers <= 0:
            return None
        cpu_bytes = int(per_layer * cpu_layers)
        n_cpu_moe: int | None = shape.first_dense_layers + cpu_layers
    else:
        # Without the layer layout the split point is unknown, so every
        # expert goes to RAM, which is also what the recipe will suggest.
        cpu_bytes = memory.expert_weights
        n_cpu_moe = None

    if cpu_bytes > ram_budget:
        return None
    return Placement(
        kind="expert_offload",
        gpu_weight_bytes=memory.weights - cpu_bytes,
        cpu_weight_bytes=cpu_bytes,
        n_cpu_moe=n_cpu_moe,
        n_gpu_layers=_num_layers(model),
        num_layers=_num_layers(model),
    )


def plan_layer_offload(
    model: ModelInfo, memory: MemoryBreakdown, gpu_budget: int
) -> Placement:
    """Whole layers on the CPU: the overflow is spilled weight bytes."""
    cpu_bytes = max(0, min(memory.total - gpu_budget, memory.weights))
    gpu_bytes = memory.weights - cpu_bytes
    num_layers = _num_layers(model)
    n_gpu_layers = (
        int(num_layers * gpu_bytes / memory.weights)
        if num_layers and memory.weights > 0
        else None
    )
    return Placement(
        kind="layer_offload",
        gpu_weight_bytes=gpu_bytes,
        cpu_weight_bytes=cpu_bytes,
        n_gpu_layers=n_gpu_layers,
        num_layers=num_layers,
    )
