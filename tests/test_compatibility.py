"""Tests for compatibility checking."""

from whichllm.engine.compatibility import check_compatibility
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _make_model(
    params: int = 7_000_000_000, context_length: int | None = None
) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=params,
        context_length=context_length,
    )


def _make_variant(size: int = 4_000_000_000) -> GGUFVariant:
    return GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=size
    )


def _make_hardware(
    vram: int = 0, ram: int = 16 * 1024**3, disk: int = 100 * 1024**3, **gpu_kwargs
) -> HardwareInfo:
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
    assert 0.0 < result.offload_ratio < 1.0
    assert any("offload" in w.lower() for w in result.warnings)


def test_shared_memory_amd_apu_uses_system_memory_pool():
    model = _make_model(120_000_000_000)
    variant = _make_variant(55_000_000_000)
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="STRXLGEN",
                vendor="amd",
                vram_bytes=512 * 1024**2,
                memory_bandwidth_gbps=256.0,
                shared_memory=True,
            )
        ],
        cpu_name="AMD Ryzen AI MAX+ 395",
        cpu_cores=16,
        ram_bytes=128 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="linux",
    )

    result = check_compatibility(model, variant, hw)

    assert result.can_run is True
    assert result.fit_type == "full_gpu"
    assert result.vram_available_bytes == int(hw.ram_bytes * 0.80)
    assert not any("offload" in w.lower() for w in result.warnings)
    assert not any("cpu only" in w.lower() for w in result.warnings)


def test_windows_shared_memory_amd_apu_does_not_emit_rocm_warning():
    model = _make_model(8_000_000_000)
    variant = _make_variant(6_000_000_000)
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
                vendor="amd",
                vram_bytes=0,
                memory_bandwidth_gbps=120.0,
                shared_memory=True,
            )
        ],
        cpu_name="AMD Ryzen AI 9 HX 370",
        cpu_cores=12,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="windows",
    )

    result = check_compatibility(model, variant, hw)

    assert result.can_run is True
    assert result.fit_type == "full_gpu"
    assert result.vram_available_bytes == int(hw.ram_bytes * 0.80)
    assert not any("rocm" in w.lower() for w in result.warnings)
    assert not any("offload" in w.lower() for w in result.warnings)


def test_shared_memory_igpu_is_not_summed_with_dedicated_gpu():
    model = _make_model(20_000_000_000)
    variant = _make_variant(14 * 1024**3)
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4060",
                vendor="nvidia",
                vram_bytes=8 * 1024**3,
                memory_bandwidth_gbps=272.0,
            ),
            GPUInfo(
                name="Intel(R) Arc(TM) Graphics",
                vendor="intel",
                vram_bytes=0,
                shared_memory=True,
            ),
        ],
        cpu_name="Intel CPU",
        cpu_cores=12,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="windows",
    )

    result = check_compatibility(model, variant, hw)

    assert result.can_run is True
    assert result.fit_type == "partial_offload"
    assert result.vram_available_bytes == 8 * 1024**3
    assert any("offloaded to CPU RAM" in w for w in result.warnings)


def test_apple_silicon_does_not_double_count_unified_memory():
    """Apple Silicon uses unified memory: vram_bytes IS the system RAM.
    The fit checker must not add a separate offload pool on top."""
    model = _make_model(70_000_000_000)
    variant = _make_variant(40_000_000_000)  # 40 GB model
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M2 Max",
                vendor="apple",
                vram_bytes=32 * 1024**3,  # 32 GB unified memory
                memory_bandwidth_gbps=400.0,
                shared_memory=True,
            )
        ],
        cpu_name="Apple M2 Max",
        cpu_cores=12,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="darwin",
    )

    result = check_compatibility(model, variant, hw)

    # Model (40 GB) exceeds unified memory (32 GB). There is no separate
    # CPU RAM pool to spill into, so this must NOT be partial_offload.
    assert result.fit_type != "partial_offload", (
        "Apple Silicon should not get partial_offload — unified memory "
        "cannot be double-counted as GPU VRAM + CPU RAM offload pool"
    )
    assert not any("offloaded to CPU RAM" in w for w in result.warnings)


def test_apple_silicon_full_gpu_fit():
    """A model that fits within unified memory should be full_gpu."""
    model = _make_model(7_000_000_000)
    variant = _make_variant(4_000_000_000)  # 4 GB model
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M4 Pro",
                vendor="apple",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=273.0,
                shared_memory=True,
            )
        ],
        cpu_name="Apple M4 Pro",
        cpu_cores=14,
        ram_bytes=24 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="darwin",
    )

    result = check_compatibility(model, variant, hw)

    assert result.can_run is True
    assert result.fit_type == "full_gpu"
    assert not any("offload" in w.lower() for w in result.warnings)


def test_apple_silicon_vendor_guard_handles_legacy_shared_memory_false():
    """Even if a cached/older GPUInfo has shared_memory=False, the
    vendor=='apple' guard should still prevent double-counting."""
    model = _make_model(70_000_000_000)
    variant = _make_variant(40_000_000_000)  # 40 GB model
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M2 Max",
                vendor="apple",
                vram_bytes=32 * 1024**3,
                memory_bandwidth_gbps=400.0,
                shared_memory=False,  # legacy/cached object
            )
        ],
        cpu_name="Apple M2 Max",
        cpu_cores=12,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="darwin",
    )

    result = check_compatibility(model, variant, hw)

    assert result.fit_type != "partial_offload", (
        "vendor='apple' guard must prevent double-counting even when "
        "shared_memory=False (cached/older GPUInfo)"
    )
    assert not any("offloaded to CPU RAM" in w for w in result.warnings)


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


def test_context_fits_true_when_model_supports():
    model = _make_model(context_length=131072)
    variant = _make_variant()
    hw = _make_hardware(vram=24 * 1024**3)
    result = check_compatibility(model, variant, hw, context_length=32768)
    assert result.context_fits is True


def test_context_fits_false_when_model_too_small():
    model = _make_model(context_length=8192)
    variant = _make_variant()
    hw = _make_hardware(vram=24 * 1024**3)
    result = check_compatibility(model, variant, hw, context_length=32768)
    assert result.context_fits is False
    assert any("max context" in w.lower() for w in result.warnings)


def test_context_fits_unknown_is_true():
    model = _make_model(context_length=None)
    variant = _make_variant()
    hw = _make_hardware(vram=24 * 1024**3)
    result = check_compatibility(model, variant, hw, context_length=32768)
    assert result.context_fits is True
    assert not any("max context" in w.lower() for w in result.warnings)
