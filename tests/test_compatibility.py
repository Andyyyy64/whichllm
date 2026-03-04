"""Tests for compatibility checking."""

from local_llm_checker.engine.compatibility import check_compatibility
from local_llm_checker.hardware.types import GPUInfo, HardwareInfo
from local_llm_checker.models.types import GGUFVariant, ModelInfo


def _make_model(params: int = 7_000_000_000) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=params,
    )


def _make_variant(size: int = 4_000_000_000) -> GGUFVariant:
    return GGUFVariant(filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=size)


def _make_hardware(vram: int = 0, ram: int = 16 * 1024**3, disk: int = 100 * 1024**3, **gpu_kwargs) -> HardwareInfo:
    gpus = []
    if vram > 0:
        gpus.append(
            GPUInfo(
                name="Test GPU",
                vendor=gpu_kwargs.get("vendor", "nvidia"),
                vram_bytes=vram,
                compute_capability=gpu_kwargs.get("cc", (8, 6)),
                memory_bandwidth_gbps=gpu_kwargs.get("bw", 500.0),
            )
        )
    return HardwareInfo(
        gpus=gpus,
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=ram,
        disk_free_bytes=disk,
        os="linux",
    )


def test_full_gpu_fit():
    model = _make_model()
    variant = _make_variant(4_000_000_000)
    hw = _make_hardware(vram=24 * 1024**3)  # 24GB VRAM
    result = check_compatibility(model, variant, hw)
    assert result.can_run is True
    assert result.fit_type == "full_gpu"


def test_partial_offload():
    model = _make_model()
    variant = _make_variant(20_000_000_000)  # 20GB model
    hw = _make_hardware(vram=8 * 1024**3, ram=64 * 1024**3)  # 8GB VRAM, 64GB RAM
    result = check_compatibility(model, variant, hw)
    assert result.can_run is True
    assert result.fit_type == "partial_offload"
    assert any("offload" in w.lower() for w in result.warnings)


def test_cpu_only():
    model = _make_model(1_000_000_000)
    variant = _make_variant(600_000_000)
    hw = _make_hardware(vram=0, ram=16 * 1024**3)  # No GPU
    result = check_compatibility(model, variant, hw)
    assert result.can_run is True
    assert result.fit_type == "cpu_only"


def test_insufficient_memory():
    model = _make_model(70_000_000_000)
    variant = _make_variant(40_000_000_000)
    hw = _make_hardware(vram=0, ram=8 * 1024**3)  # Only 8GB RAM, no GPU
    result = check_compatibility(model, variant, hw)
    assert result.can_run is False


def test_low_compute_capability():
    model = _make_model()
    variant = _make_variant(4_000_000_000)
    hw = _make_hardware(vram=24 * 1024**3, cc=(4, 0))  # Very old GPU
    result = check_compatibility(model, variant, hw)
    assert result.can_run is True  # Still runs, just with warning
    assert any("compute capability" in w.lower() for w in result.warnings)


def test_insufficient_disk():
    model = _make_model()
    variant = _make_variant(50_000_000_000)  # 50GB file
    hw = _make_hardware(vram=80 * 1024**3, disk=10 * 1024**3)  # Only 10GB disk
    result = check_compatibility(model, variant, hw)
    assert result.can_run is False
    assert any("disk" in w.lower() for w in result.warnings)
