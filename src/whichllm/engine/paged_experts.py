"""SGLang's Paged Experts: an MoE served from one GPU with its experts in RAM.

Each MoE layer keeps K of its E routed experts in a K-slot table on the GPU and
holds all E in pinned host memory. A step copies in the experts it routes to
that are not resident, and the GPU computes them (SGLang 0.5.23,
docs/advanced_features/paged_experts.mdx). Paging moves only those experts,
where ``--cpu-offload-gb`` streams every offloaded weight on every token.
"""

from __future__ import annotations

from whichllm.engine.moe import moe_layer_count
from whichllm.engine.vram import MemoryBreakdown
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import ModelInfo

# The docs: "Under WSL2, pinned memory is capped at about half of system RAM."
_WSL_PINNED_RAM_SHARE = 0.5
# SGLang's static budget for the weights, the K slots and the KV cache, as a
# fraction of the GPU's memory (--mem-fraction-static); the rest is left to
# activations and CUDA graphs. The docs' example passes 0.88, and so does the
# command whichllm writes, so K fits what the server reserves.
STATIC_MEMORY_FRACTION = 0.88


def pinned_ram_bytes(hardware: HardwareInfo, usable_ram: int) -> int:
    """RAM the host store can pin: the usable RAM, at most about half under WSL."""
    if hardware.wsl:
        return min(usable_ram, int(hardware.ram_bytes * _WSL_PINNED_RAM_SHARE))
    return usable_ram


def resident_experts(
    model: ModelInfo,
    memory: MemoryBreakdown,
    gpu_budget: int,
    pinned_ram: int,
) -> int | None:
    """K, the routed experts per MoE layer kept on the GPU, or None.

    One resident expert costs its bytes in every MoE layer. As the docs'
    "Choosing K" works it out, K takes the room the static budget leaves
    after the rest of the weights and the KV cache, here also after the CUDA
    context, and must be at least the router's top-k. The host store holds
    every expert, so all of them must fit in pinned RAM.
    """
    shape = model.shape
    if shape is None or not shape.num_experts or not shape.experts_per_token:
        return None
    if memory.expert_weights <= 0 or not moe_layer_count(shape):
        return None
    if memory.expert_weights > pinned_ram:
        return None
    room = (
        gpu_budget * STATIC_MEMORY_FRACTION
        - memory.overhead
        - (memory.weights - memory.expert_weights)
        - memory.kv_cache
    )
    if room <= 0:
        return None
    per_expert = memory.expert_weights / shape.num_experts
    resident = min(int(room // per_expert), shape.num_experts)
    if resident < shape.experts_per_token:
        return None
    return resident


def has_shared_experts(model: ModelInfo) -> bool:
    """Whether the model has shared experts, which SGLang may fuse."""
    shape = model.shape
    return bool(shape and shape.shared_expert_intermediate_size)


__all__ = [
    "STATIC_MEMORY_FRACTION",
    "has_shared_experts",
    "pinned_ram_bytes",
    "resident_experts",
]
