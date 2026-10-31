"""Tests for per-device planning of a multi-GPU layer split."""

from whichllm.constants import _GiB
from whichllm.engine.multi_gpu import (
    PER_DEVICE_RUNTIME_BUFFER_BYTES,
    plan_device_split,
)
from whichllm.engine.placement import Placement
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import ModelInfo
from whichllm.output.json_output import _placement_json, _placement_rows


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

    plan = plan_device_split(
        gpus,
        [6 * _GiB, 16 * _GiB],
        gpu_weight_bytes=14 * _GiB,
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
    assert plan.reserve_fits is True


def test_shares_follow_each_device_budget():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]

    plan = plan_device_split(
        gpus,
        [6 * _GiB, 16 * _GiB],
        gpu_weight_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )

    small, large = plan.devices
    assert small.weight_bytes < large.weight_bytes
    assert small.kv_bytes < large.kv_bytes
    assert small.required_bytes <= small.budget_bytes
    assert large.required_bytes <= large.budget_bytes
    budget_total = 22 * _GiB - 2 * PER_DEVICE_RUNTIME_BUFFER_BYTES
    assert plan.pooled_bytes == int(budget_total * 0.9)
    # Shares cover the whole splittable requirement.
    assert abs(
        sum(device.weight_bytes + device.kv_bytes for device in plan.devices)
        - 16 * _GiB
    ) <= len(plan.devices)


def test_device_below_its_runtime_buffer_leaves_one_card():
    gpus = [_gpu("Tiny", 256 * 1024**2), _gpu("RTX 5060 Ti", 16 * _GiB)]

    plan = plan_device_split(
        gpus,
        [256 * 1024**2, 16 * _GiB],
        gpu_weight_bytes=8 * _GiB,
        kv_bytes=1 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.95,
    )

    # The caller drops the card that cannot pay its buffer, so a plan built from
    # one usable device is not a split.
    assert plan.devices == ()
    assert plan.reserve_fits is False


def test_runtime_part_that_fits_no_device_rejects_the_split():
    gpus = [_gpu("GT 1030", 1 * _GiB), _gpu("GT 1030", 1 * _GiB)]

    plan = plan_device_split(
        gpus,
        [1 * _GiB, 1 * _GiB],
        gpu_weight_bytes=64 * 1024**2,
        kv_bytes=1024**2,
        nonsplittable_bytes=1060 * 1024**2,
        utilization=0.95,
    )

    assert plan.reserve_fits is False
    # The device that was asked to host the runtime-only part cannot hold it.
    reserved = [device for device in plan.devices if device.nonsplittable_bytes]
    assert reserved and not reserved[0].fits


def test_placement_json_reports_the_device_split():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]
    plan = plan_device_split(
        gpus,
        [6 * _GiB, 16 * _GiB],
        gpu_weight_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )
    result = CompatibilityResult(
        model=ModelInfo(
            id="org/Test-7B-GGUF", family_id="test", name="Test-7B", parameter_count=7e9
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=17 * _GiB,
        vram_available_bytes=22 * _GiB,
        fit_type="full_gpu",
        placement=Placement(
            kind="gpu",
            gpu_weight_bytes=14 * _GiB,
            cpu_weight_bytes=0,
            devices=plan.devices,
        ),
    )

    payload = _placement_json(result)

    assert payload is not None
    assert payload["kind"] == "gpu"
    assert [device["name"] for device in payload["devices"]] == [
        "GTX 1660 SUPER",
        "RTX 5060 Ti",
    ]
    assert payload["devices"][0]["budget_bytes"] == 6 * _GiB - (
        PER_DEVICE_RUNTIME_BUFFER_BYTES
    )


def test_placement_rows_serialize_every_device():
    gpus = [_gpu("GTX 1660 SUPER", 6 * _GiB), _gpu("RTX 5060 Ti", 16 * _GiB)]
    plan = plan_device_split(
        gpus,
        [6 * _GiB, 16 * _GiB],
        gpu_weight_bytes=14 * _GiB,
        kv_bytes=2 * _GiB,
        nonsplittable_bytes=1 * _GiB,
        utilization=0.9,
    )

    rows = _placement_rows(plan.devices)

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
    assert _placement_rows(()) is None
    assert _placement_rows(None) is None
