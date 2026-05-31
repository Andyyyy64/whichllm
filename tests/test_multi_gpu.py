"""Tests for multi-GPU support: parsing, VRAM pooling, and speed estimation."""

import pytest

from whichllm.constants import _GiB
from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.performance import (
    estimate_tok_per_sec,
    estimate_tok_per_sec_multi_gpu,
)
from whichllm.hardware.gpu_simulator import create_synthetic_gpu, parse_multi_gpu_spec
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _make_gpu(name: str = "Test GPU", vram_gb: int = 24, bw: float = 500.0) -> GPUInfo:
    return GPUInfo(
        name=name,
        vendor="nvidia",
        vram_bytes=vram_gb * _GiB,
        compute_capability=(8, 6),
        memory_bandwidth_gbps=bw,
    )


def _make_model(params: int = 70_000_000_000) -> ModelInfo:
    return ModelInfo(
        id="test/large-model",
        family_id="test/large-model",
        name="large-model",
        parameter_count=params,
    )


def _make_variant(size: int = 40_000_000_000) -> GGUFVariant:
    return GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=size
    )


class TestParseMultiGpuSpec:
    def test_single_gpu(self):
        result = parse_multi_gpu_spec("RTX 4090")
        assert result == [("RTX 4090", None)]

    def test_comma_separated(self):
        result = parse_multi_gpu_spec("RTX 5080,RTX 5060 Ti")
        assert result == [("RTX 5080", None), ("RTX 5060 Ti", None)]

    def test_comma_with_spaces(self):
        result = parse_multi_gpu_spec("RTX 5080 , RTX 5060 Ti 16GB")
        assert result == [("RTX 5080", None), ("RTX 5060 Ti 16GB", None)]

    def test_count_shorthand(self):
        result = parse_multi_gpu_spec("2x RTX 4090")
        assert len(result) == 2
        assert all(name == "RTX 4090" for name, _ in result)

    def test_count_shorthand_four(self):
        result = parse_multi_gpu_spec("4x H100")
        assert len(result) == 4
        assert all(name == "H100" for name, _ in result)

    def test_count_shorthand_case_insensitive(self):
        result = parse_multi_gpu_spec("2X RTX 3090")
        assert len(result) == 2

    def test_mixed_count_and_comma(self):
        result = parse_multi_gpu_spec("2x RTX 3090,RTX 4090")
        assert len(result) == 3
        assert result[0] == ("RTX 3090", None)
        assert result[1] == ("RTX 3090", None)
        assert result[2] == ("RTX 4090", None)

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_multi_gpu_spec("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_multi_gpu_spec("   ")

    def test_count_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            parse_multi_gpu_spec("0x RTX 4090")

    def test_count_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="exceed 16"):
            parse_multi_gpu_spec("17x RTX 4090")

    def test_total_count_exceeds_max_raises(self):
        spec = ",".join(["RTX 4090"] * 17)
        with pytest.raises(ValueError, match="exceed 16"):
            parse_multi_gpu_spec(spec)


class TestMultiGpuCreateSynthetic:
    def test_dual_gpu_from_spec(self):
        specs = parse_multi_gpu_spec("2x RTX 4090")
        gpus = [create_synthetic_gpu(name) for name, _ in specs]
        assert len(gpus) == 2
        assert all(g.vram_bytes == 24 * _GiB for g in gpus)
        assert all(g.vendor == "nvidia" for g in gpus)

    def test_heterogeneous_gpus(self):
        specs = parse_multi_gpu_spec("RTX 4090,RTX 3090")
        gpus = [create_synthetic_gpu(name) for name, _ in specs]
        assert len(gpus) == 2
        assert gpus[0].vram_bytes == 24 * _GiB
        assert gpus[1].vram_bytes == 24 * _GiB


class TestMultiGpuVramPooling:
    def test_dual_gpu_vram_sums(self):
        model = _make_model(70_000_000_000)
        variant = _make_variant(40_000_000_000)
        hw = HardwareInfo(
            gpus=[_make_gpu(vram_gb=24), _make_gpu(vram_gb=24)],
            cpu_name="Test CPU",
            cpu_cores=8,
            ram_bytes=64 * _GiB,
            disk_free_bytes=200 * _GiB,
            os="linux",
        )
        result = check_compatibility(model, variant, hw)
        assert result.vram_available_bytes == 48 * _GiB
        assert result.can_run is True
        assert result.fit_type == "full_gpu"

    def test_heterogeneous_vram_sums(self):
        model = _make_model(70_000_000_000)
        variant = _make_variant(30_000_000_000)
        hw = HardwareInfo(
            gpus=[_make_gpu(vram_gb=24), _make_gpu(vram_gb=12)],
            cpu_name="Test CPU",
            cpu_cores=8,
            ram_bytes=64 * _GiB,
            disk_free_bytes=200 * _GiB,
            os="linux",
        )
        result = check_compatibility(model, variant, hw)
        assert result.vram_available_bytes == 36 * _GiB
        assert result.can_run is True
        assert result.fit_type == "full_gpu"


class TestMultiGpuSpeedEstimation:
    def test_multi_gpu_faster_than_single(self):
        model = _make_model(70_000_000_000)
        variant = _make_variant(40_000_000_000)
        gpu = _make_gpu(vram_gb=24, bw=1000.0)

        single_speed = estimate_tok_per_sec(model, variant, gpu, "full_gpu")
        multi_speed = estimate_tok_per_sec_multi_gpu(
            model, variant, [gpu, gpu], "full_gpu"
        )
        assert multi_speed > single_speed

    def test_single_gpu_passthrough(self):
        model = _make_model()
        variant = _make_variant()
        gpu = _make_gpu()

        single = estimate_tok_per_sec(model, variant, gpu, "full_gpu")
        multi = estimate_tok_per_sec_multi_gpu(model, variant, [gpu], "full_gpu")
        assert single == multi

    def test_no_gpu_passthrough(self):
        model = _make_model(7_000_000_000)
        variant = _make_variant(4_000_000_000)

        single = estimate_tok_per_sec(model, variant, None, "cpu_only")
        multi = estimate_tok_per_sec_multi_gpu(model, variant, [], "cpu_only")
        assert single == multi

    def test_heterogeneous_bottlenecked_by_slowest(self):
        model = _make_model(70_000_000_000)
        variant = _make_variant(40_000_000_000)
        fast_gpu = _make_gpu(name="Fast GPU", vram_gb=24, bw=1000.0)
        slow_gpu = _make_gpu(name="Slow GPU", vram_gb=24, bw=200.0)

        balanced_speed = estimate_tok_per_sec_multi_gpu(
            model, variant, [fast_gpu, fast_gpu], "full_gpu"
        )
        mixed_speed = estimate_tok_per_sec_multi_gpu(
            model, variant, [fast_gpu, slow_gpu], "full_gpu"
        )
        assert mixed_speed < balanced_speed

    def test_four_gpus_faster_than_two(self):
        model = _make_model(70_000_000_000)
        variant = _make_variant(40_000_000_000)
        gpu = _make_gpu(vram_gb=80, bw=2000.0)

        two_speed = estimate_tok_per_sec_multi_gpu(
            model, variant, [gpu, gpu], "full_gpu"
        )
        four_speed = estimate_tok_per_sec_multi_gpu(
            model, variant, [gpu] * 4, "full_gpu"
        )
        assert four_speed > two_speed
