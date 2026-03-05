"""Tests for CLI helper logic."""

from whichllm.cli import (
    _auto_min_params_for_profile,
    _fill_missing_published_at,
    _include_vision_candidates,
)
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import ModelInfo


def _hw_with_gpu(vram_gb: int) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                memory_bandwidth_gbps=1.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=1,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )


def test_auto_min_params_general_by_vram():
    assert _auto_min_params_for_profile(_hw_with_gpu(8), "general") == 7.0
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "general") == 10.0
    assert _auto_min_params_for_profile(_hw_with_gpu(32), "general") == 12.0


def test_auto_min_params_non_general_disabled():
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "coding") is None


def test_include_vision_candidates_by_profile():
    assert _include_vision_candidates("vision") is True
    assert _include_vision_candidates("any") is True
    assert _include_vision_candidates("general") is False
    assert _include_vision_candidates("coding") is False


def test_fill_missing_published_at_updates_models():
    model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
    )

    async def _fake_fetch(ids: list[str]) -> dict[str, str]:
        assert ids == ["Qwen/Qwen3-8B-AWQ"]
        return {"Qwen/Qwen3-8B-AWQ": "2026-03-05T08:00:00.000Z"}

    updated = _fill_missing_published_at([model], [result], _fake_fetch)
    assert updated is True
    assert model.published_at == "2026-03-05T08:00:00.000Z"
