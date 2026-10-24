"""Per-device placement planning for multi-GPU fit checks.

A combined-VRAM check answers "is there enough memory in total". It cannot answer
"does every device hold its own share", which is what decides whether a layer
split loads at all: a placement that overflows one device fails even when the
combined budget would have been enough.
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

    @property
    def overflow_bytes(self) -> int:
        """Bytes this device is asked to hold beyond its own budget."""
        return max(0, self.required_bytes - self.budget_bytes)


@dataclass(frozen=True)
class PlacementPlan:
    """Placement of weights and KV cache across the participating devices."""

    devices: tuple[DevicePlacement, ...]
    excluded: tuple[str, ...]
    effective_bytes: int
    fits: bool


def _empty_plan(excluded: tuple[str, ...]) -> PlacementPlan:
    return PlacementPlan(devices=(), excluded=excluded, effective_bytes=0, fits=False)


def plan_multi_gpu_placement(
    gpus: Sequence[GPUInfo],
    available: Sequence[int],
    *,
    weights_bytes: int,
    kv_bytes: int,
    nonsplittable_bytes: int,
    utilization: float,
) -> PlacementPlan:
    """Split weights and KV cache across devices and check each device.

    Weights and KV cache are assigned in proportion to the capacity each device
    has left after its own runtime buffer. Activation buffers and framework
    overhead cannot be split in a layer split, so they are reserved on the
    device with the most capacity. A device that cannot hold its share makes the
    whole placement unusable, which is what a combined-total check misses.
    """
    budgets: list[tuple[int, GPUInfo, int]] = []
    excluded: list[str] = []
    for index, (gpu, gpu_available) in enumerate(zip(gpus, available, strict=False)):
        budget = max(0, gpu_available - PER_DEVICE_RUNTIME_BUFFER_BYTES)
        if budget == 0:
            excluded.append(gpu.name)
            continue
        budgets.append((index, gpu, budget))

    if len(budgets) < 2:
        # Fewer than two usable devices is a single-device fit, not a split.
        return _empty_plan(tuple(excluded))

    reserve_position = max(range(len(budgets)), key=lambda pos: budgets[pos][2])
    capacities = [budget for _, _, budget in budgets]
    capacities[reserve_position] -= nonsplittable_bytes

    reserved_overflow = capacities[reserve_position] < 0
    if reserved_overflow:
        # The runtime-only part fits no device in the pool: nothing can be
        # placed even though the combined budget is larger.
        capacities = [max(0, capacity) for capacity in capacities]
        splittable_capacity = 0
    else:
        splittable_capacity = sum(capacities)

    weights = max(0, weights_bytes)
    kv = max(0, kv_bytes)

    placements: list[DevicePlacement] = []
    for position, (_, gpu, budget) in enumerate(budgets):
        capacity = capacities[position]
        share = capacity / splittable_capacity if splittable_capacity else 0.0
        weight_share = int(weights * share)
        kv_share = int(kv * share)
        nonsplittable_share = nonsplittable_bytes if position == reserve_position else 0
        required = weight_share + kv_share + nonsplittable_share
        placements.append(
            DevicePlacement(
                name=gpu.name,
                available_bytes=available[budgets[position][0]],
                budget_bytes=budget,
                weight_bytes=weight_share,
                kv_bytes=kv_share,
                nonsplittable_bytes=nonsplittable_share,
                required_bytes=required,
                fits=required <= budget,
            )
        )

    fits = not reserved_overflow and all(placement.fits for placement in placements)
    plannable = splittable_capacity + (0 if reserved_overflow else nonsplittable_bytes)
    effective = int(plannable * utilization) if fits else 0

    return PlacementPlan(
        devices=tuple(placements),
        excluded=tuple(excluded),
        effective_bytes=max(0, effective),
        fits=fits,
    )
