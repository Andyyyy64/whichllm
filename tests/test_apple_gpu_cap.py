"""Metal's default GPU working set on Apple Silicon."""

from __future__ import annotations

from whichllm.engine.compatibility import check_compatibility
from whichllm.hardware.apple import default_gpu_working_set_bytes
from whichllm.hardware.gpu_simulator import create_synthetic_gpu
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3


def test_working_set_is_two_thirds_up_to_32gb_and_three_quarters_above():
    assert default_gpu_working_set_bytes(16 * _GiB) == int(16 * _GiB * 2 / 3)
    assert default_gpu_working_set_bytes(32 * _GiB) == int(32 * _GiB * 2 / 3)
    assert default_gpu_working_set_bytes(64 * _GiB) == 48 * _GiB


def test_simulated_mac_gets_the_working_set_not_all_memory():
    gpu = create_synthetic_gpu("M4", vram_override_gb=16)
    assert gpu.unified_memory_bytes == 16 * _GiB
    assert gpu.vram_bytes < 11 * _GiB


def _mac(ram_gb: int) -> HardwareInfo:
    return HardwareInfo(
        gpus=[create_synthetic_gpu("M4", vram_override_gb=ram_gb)],
        cpu_name="Apple M4",
        cpu_cores=10,
        ram_bytes=ram_gb * _GiB,
        disk_free_bytes=1024 * _GiB,
        os="darwin",
    )


def _model(size_gb: float) -> tuple[ModelInfo, GGUFVariant]:
    variant = GGUFVariant(
        filename="m-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=int(size_gb * _GiB),
    )
    model = ModelInfo(
        id="org/Model-14B",
        family_id="org/model-14b",
        name="Model-14B",
        parameter_count=14_000_000_000,
        gguf_variants=[variant],
    )
    return model, variant


def test_a_model_past_the_working_set_spills_into_unified_memory():
    model, variant = _model(8.5)
    result = check_compatibility(model, variant, _mac(16))
    assert result.can_run
    assert result.fit_type == "partial_offload"
    assert any("shared system memory" in w for w in result.warnings)


def test_a_model_within_the_working_set_stays_on_the_gpu():
    model, variant = _model(6.0)
    result = check_compatibility(model, variant, _mac(16))
    assert result.fit_type == "full_gpu"
