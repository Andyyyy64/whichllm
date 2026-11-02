"""Per-device planning for a multi-GPU layer split.

A pooled-VRAM check answers "is there enough memory in total". It cannot answer
"does every device hold its own share", which is what decides whether a layer
split loads at all: a placement that overflows one device fails even when the
pooled budget would have been enough.

The split plans only the part that stays on the GPUs. On a discrete card the
``Placement`` from :mod:`whichllm.engine.placement` already says which weights
run in system RAM (MoE experts or whole layers); this module divides what is
left — the GPU-resident weights plus the KV cache — across the devices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from whichllm.constants import _GiB
from whichllm.hardware.types import GPUInfo

# Per-device runtime buffer: driver context, graph buffers, and allocator slack
# that every participating device needs before a single layer can live on it.
PER_DEVICE_RUNTIME_BUFFER_BYTES = int(0.3 * _GiB)


@dataclass(frozen=True)
class DevicePlacement:
    """One device's share of a layer-split placement."""

    name: str
    available_bytes: int
    budget_bytes: int
    weight_bytes: int
    kv_bytes: int
    nonsplittable_bytes: int
    required_bytes: int
    fits: bool


@dataclass(frozen=True)
class PlacementPlan:
    """Placement of the GPU-resident part across the participating devices.

    ``pooled_bytes`` is the budget the pooled fit check uses: the usable
    capacity of every device after its own runtime buffer, times the
    utilization factor. ``reserve_fits`` says whether the runtime-only part fits
    the largest device's split budget — when it does not, no layer split can
    carry it and the caller falls back to a single card.

    Shares are proportional to each device's remaining capacity, so a device
    cannot overflow once the reserve fits: the only way a split fails is the
    runtime-only part.
    """

    devices: tuple[DevicePlacement, ...]
    pooled_bytes: int
    reserve_fits: bool


def plan_device_split(
    gpus: Sequence[GPUInfo],
    available: Sequence[int],
    *,
    gpu_weight_bytes: int,
    kv_bytes: int,
    nonsplittable_bytes: int,
    utilization: float,
) -> PlacementPlan:
    """Split the GPU-resident part across devices and check each device.

    Weights and KV cache are assigned in proportion to the capacity each device
    has left after its own runtime buffer. Activation buffers and framework
    overhead cannot be divided by a layer split, so they are reserved on the
    device with the most capacity. A device that cannot hold its share makes the
    split unusable, which is what a pooled-total check misses.

    The caller drops devices that cannot pay their own runtime buffer first, so
    ``available`` holds the participating devices only.
    """
    budgets: list[tuple[int, GPUInfo, int]] = []
    for index, (gpu, gpu_available) in enumerate(zip(gpus, available, strict=False)):
        budget = max(0, gpu_available - PER_DEVICE_RUNTIME_BUFFER_BYTES)
        if budget == 0:
            continue
        budgets.append((index, gpu, budget))

    pooled_bytes = int(max(0, sum(budget for _, _, budget in budgets)) * utilization)

    if len(budgets) < 2:
        # Defensive: a split needs two devices that can pay their buffer.
        return PlacementPlan(devices=(), pooled_bytes=pooled_bytes, reserve_fits=False)

    reserve_position = max(range(len(budgets)), key=lambda pos: budgets[pos][2])
    capacities = [budget for _, _, budget in budgets]
    reserve_fits = capacities[reserve_position] >= nonsplittable_bytes
    if reserve_fits:
        capacities[reserve_position] -= nonsplittable_bytes
        splittable_capacity = sum(capacities)
    else:
        # The runtime-only part fits no device in the pool: nothing can be
        # placed even though the pooled budget is larger.
        capacities = [max(0, capacity) for capacity in capacities]
        splittable_capacity = 0

    weights = max(0, gpu_weight_bytes)
    kv = max(0, kv_bytes)
    placements: list[DevicePlacement] = []
    for position, (index, gpu, budget) in enumerate(budgets):
        capacity = capacities[position]
        share = capacity / splittable_capacity if splittable_capacity else 0.0
        weight_share = int(weights * share)
        kv_share = int(kv * share)
        nonsplittable_share = nonsplittable_bytes if position == reserve_position else 0
        required = weight_share + kv_share + nonsplittable_share
        placements.append(
            DevicePlacement(
                name=gpu.name,
                available_bytes=available[index],
                budget_bytes=budget,
                weight_bytes=weight_share,
                kv_bytes=kv_share,
                nonsplittable_bytes=nonsplittable_share,
                required_bytes=required,
                fits=required <= budget,
            )
        )

    return PlacementPlan(
        devices=tuple(placements),
        pooled_bytes=pooled_bytes,
        reserve_fits=reserve_fits,
    )
