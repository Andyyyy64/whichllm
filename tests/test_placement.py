"""Tests for per-device layer-split placement planning."""

from whichllm.constants import _GiB
from whichllm.engine.placement import (
    PER_DEVICE_RUNTIME_BUFFER_BYTES,
    plan_multi_gpu_placement,
)
from whichllm.hardware.types import GPUInfo
from whichllm.output.json_output import _placement_rows


def _gpu(name: str, vram_bytes: int) -> GPUInfo:
    return GPUInfo(
        name=name,
        vendor="nvidia",
        vram_bytes=vram_bytes,
        compute_capability=(8, 6),
        memory_bandwidth_gbps=500.0,
    )


def test_each_device_pays_its_own_runtime_buffer():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]
    available = [6 * _GiB, 16 * _GiB]

    plan = plan_multi_gpu_placement(
        gpus,
        available,
        weights_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )

    assert [device.budget_bytes for device in plan.devices] == [
        6 * _GiB - PER_DEVICE_RUNTIME_BUFFER_BYTES,
        16 * _GiB - PER_DEVICE_RUNTIME_BUFFER_BYTES,
    ]
    # The runtime-only part is reserved once, on the largest device.
    assert plan.devices[0].nonsplittable_bytes == 0
    assert plan.devices[1].nonsplittable_bytes == 1 * _GiB
    assert plan.fits is True


def test_shares_follow_each_device_budget():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]
    available = [6 * _GiB, 16 * _GiB]

    plan = plan_multi_gpu_placement(
        gpus,
        available,
        weights_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )

    small, large = plan.devices
    assert small.weight_bytes < large.weight_bytes
    assert small.kv_bytes < large.kv_bytes
    assert small.required_bytes <= small.budget_bytes
    assert large.required_bytes <= large.budget_bytes
    total_budget = 22 * _GiB - 2 * PER_DEVICE_RUNTIME_BUFFER_BYTES
    assert plan.effective_bytes == int(total_budget * 0.9)
    # Shares cover the whole splittable requirement.
    assert abs(
        sum(device.weight_bytes + device.kv_bytes for device in plan.devices)
        - 16 * _GiB
    ) <= 1 * len(plan.devices)


def test_device_below_its_runtime_buffer_is_excluded():
    gpus = [_gpu("Tiny", 256 * 1024**2), _gpu("RTX 5060 Ti", 16 * _GiB)]

    plan = plan_multi_gpu_placement(
        gpus,
        [256 * 1024**2, 16 * _GiB],
        weights_bytes=8 * _GiB,
        kv_bytes=1 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.95,
    )

    # One usable device left: this is a single-device fit, not a split.
    assert plan.devices == ()
    assert plan.excluded == ("Tiny",)
    assert plan.fits is False
    assert plan.effective_bytes == 0


def test_runtime_part_that_fits_no_device_rejects_the_placement():
    gpus = [_gpu("GT 1030", 1 * _GiB), _gpu("GT 1030", 1 * _GiB)]

    plan = plan_multi_gpu_placement(
        gpus,
        [1 * _GiB, 1 * _GiB],
        weights_bytes=64 * 1024**2,
        kv_bytes=1024**2,
        nonsplittable_bytes=1060 * 1024**2,
        utilization=0.95,
    )

    assert plan.fits is False
    assert plan.effective_bytes == 0
    # The device that was asked to host the runtime-only part cannot hold it.
    reserved = [device for device in plan.devices if device.nonsplittable_bytes]
    assert reserved and not reserved[0].fits
    assert reserved[0].overflow_bytes > 0


def test_placement_rows_serialize_every_device():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]
    plan = plan_multi_gpu_placement(
        gpus,
        [6 * _GiB, 16 * _GiB],
        weights_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )

    rows = _placement_rows(list(plan.devices))

    assert rows is not None
    assert [row["name"] for row in rows] == ["GTX 1660 SUPER", "RTX 5060 Ti"]
    assert set(rows[0]) == {
        "name",
        "available_bytes",
        "budget_bytes",
        "weight_bytes",
        "kv_bytes",
        "nonsplittable_bytes",
        "required_bytes",
        "fits",
    }
    assert _placement_rows(None) is None
