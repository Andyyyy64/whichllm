"""MoE expert offload: RAM bandwidth, placement, speed and scoring."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from whichllm.cli import app
from whichllm.cli_hardware import _apply_ram_bandwidth
from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.moe import active_expert_fraction, routed_expert_fraction
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.engine.placement import plan_expert_offload
from whichllm.engine.ranking_score import _partial_offload_quality_factor
from whichllm.engine.vram import estimate_memory_breakdown
from whichllm.hardware import ram_bandwidth as ram_bw
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.shape import ModelShape
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3


def _gpt_oss_120b(*, shape: bool = True) -> ModelInfo:
    return ModelInfo(
        id="openai/gpt-oss-120b",
        family_id="gpt-oss-120b",
        name="gpt-oss-120b",
        parameter_count=116_829_156_672,
        parameter_count_active=5_100_000_000,
        is_moe=True,
        gguf_variants=[
            GGUFVariant(
                filename="gpt-oss-120b-MXFP4.gguf",
                quant_type="MXFP4",
                file_size_bytes=63_387_346_464,
            )
        ],
        shape=(
            ModelShape(
                model_type="gpt_oss",
                num_layers=36,
                full_kv_bytes_per_token=18 * 2048,
                sliding_kv_bytes_per_token=18 * 2048,
                sliding_window=128,
                honors_sliding_window=True,
                hidden_size=2880,
                intermediate_size=2880,
                num_experts=128,
                experts_per_token=4,
            )
            if shape
            else None
        ),
    )


def _dense_32b() -> ModelInfo:
    return ModelInfo(
        id="example/Dense-32B",
        family_id="dense-32b",
        name="Dense-32B",
        parameter_count=32_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="dense-32b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=19 * _GiB,
            )
        ],
    )


def _pc(vram_gb: float = 16, ram_gb: float = 64, ram_gbps: float = 89.6):
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="RTX 5080",
                vendor="nvidia",
                vram_bytes=int(vram_gb * _GiB),
                memory_bandwidth_gbps=960.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=16,
        ram_bytes=int(ram_gb * _GiB),
        ram_bandwidth_gbps=ram_gbps,
        ram_bandwidth_source="detected",
        disk_free_bytes=2048 * _GiB,
        os="windows",
    )


# --- RAM bandwidth -----------------------------------------------------------


def test_dual_channel_ddr5_modules_give_bus_width_times_rate():
    modules = [
        {"ConfiguredClockSpeed": 5600, "DataWidth": 64, "SMBIOSMemoryType": 34},
        {"ConfiguredClockSpeed": 5600, "DataWidth": 64, "SMBIOSMemoryType": 34},
    ]
    result = ram_bw.bandwidth_from_modules(modules, "Intel(R) Core(TM) Ultra 9 285K")
    assert result == ram_bw.RamBandwidth(89.6, "detected", "DDR5-5600, 128-bit")


def test_four_dimms_on_a_consumer_cpu_stay_dual_channel():
    dimm = {"Speed": 3200, "DataWidth": 64, "SMBIOSMemoryType": 26}
    consumer = ram_bw.bandwidth_from_modules([dimm] * 4, "AMD Ryzen 7 5800X")
    workstation = ram_bw.bandwidth_from_modules(
        [dimm] * 4, "AMD Ryzen Threadripper 3970X"
    )
    assert consumer is not None and consumer.gbps == 51.2
    assert workstation is not None and workstation.gbps == 102.4


def test_modules_without_a_usable_rate_are_ignored():
    assert ram_bw.bandwidth_from_modules([{"Speed": 0}, "junk"], "CPU") is None


@pytest.mark.parametrize(
    ("cpu_name", "gbps"),
    [
        ("Intel(R) Core(TM) Ultra 9 285K", 89.6),
        ("Intel(R) Core(TM) Ultra 7 258V", 136.5),
        ("AMD Ryzen AI Max+ 395 w/ Radeon 8060S", 256.0),
        ("AMD Ryzen 9 7950X3D 16-Core Processor", 83.2),
        ("AMD Ryzen 7 8845HS w/ Radeon 780M Graphics", 83.2),
        # DDR4 or DDR5: left to the conservative default.
        ("13th Gen Intel(R) Core(TM) i7-13700K", None),
        ("AMD Ryzen 7 5800X 8-Core Processor", None),
    ],
)
def test_cpu_family_estimates(cpu_name, gbps):
    estimate = ram_bw.estimate_from_cpu_name(cpu_name)
    assert (estimate.gbps if estimate else None) == gbps


def test_unknown_machine_falls_back_to_dual_channel_ddr4():
    result = ram_bw.detect_ram_bandwidth("linux", "Some CPU", [])
    assert result.gbps == ram_bw.DEFAULT_RAM_BANDWIDTH_GBPS
    assert result.source == "default"


def test_apple_gpu_defines_unified_ram_bandwidth():
    m4 = GPUInfo(
        name="Apple M4 Max",
        vendor="apple",
        vram_bytes=64 * _GiB,
        memory_bandwidth_gbps=546.0,
        shared_memory=True,
    )
    result = ram_bw.detect_ram_bandwidth("darwin", "Apple M4 Max", [m4])
    assert (result.gbps, result.source) == (546.0, "unified")


def test_ram_bandwidth_flag_also_sets_shared_memory_gpus():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(name="RTX 4060", vendor="nvidia", vram_bytes=8 * _GiB),
            GPUInfo(name="Intel UHD", vendor="intel", vram_bytes=0, shared_memory=True),
        ],
        ram_bytes=32 * _GiB,
    )
    _apply_ram_bandwidth(hw, 68.0)
    assert (hw.ram_bandwidth_gbps, hw.ram_bandwidth_source) == (68.0, "override")
    assert hw.gpus[0].memory_bandwidth_gbps is None
    assert hw.gpus[1].memory_bandwidth_gbps == 68.0


def test_explicit_gpu_bandwidth_wins_over_ram_bandwidth_for_an_igpu():
    igpu = GPUInfo(
        name="Intel UHD",
        vendor="intel",
        vram_bytes=0,
        shared_memory=True,
        memory_bandwidth_gbps=60.0,
    )
    hw = HardwareInfo(gpus=[igpu], ram_bytes=32 * _GiB)
    _apply_ram_bandwidth(hw, 68.0, gpu_bandwidth_set=True)
    assert hw.ram_bandwidth_gbps == 68.0
    assert igpu.memory_bandwidth_gbps == 60.0


def test_cli_rejects_non_positive_ram_bandwidth():
    result = CliRunner().invoke(app, ["hardware", "--ram-bandwidth", "0"])
    assert result.exit_code == 1
    assert "--ram-bandwidth must be greater than 0" in result.output


# --- Expert accounting and placement ----------------------------------------


def test_routed_experts_are_counted_from_the_shape():
    model = _gpt_oss_120b()
    # 36 layers x 128 experts x 3 x 2880 x 2880 = 114.6B of 116.8B.
    assert routed_expert_fraction(model) == pytest.approx(0.97)
    assert active_expert_fraction(model) == 4 / 128


def test_without_a_shape_only_inactive_weights_count_as_experts():
    model = _gpt_oss_120b(shape=False)
    assert routed_expert_fraction(model) == pytest.approx(1 - 5.1 / 116.83, abs=1e-3)


def test_dense_models_have_no_routed_experts():
    assert routed_expert_fraction(_dense_32b()) == 0.0


def test_expert_offload_moves_the_fewest_layers_that_make_it_fit():
    model = _gpt_oss_120b()
    variant = model.gguf_variants[0]
    memory = estimate_memory_breakdown(model, variant, 8192)

    on_16gb = plan_expert_offload(model, memory, 15 * _GiB, 50 * _GiB)
    on_24gb = plan_expert_offload(model, memory, 23 * _GiB, 50 * _GiB)

    assert on_16gb is not None and on_24gb is not None
    assert 0 < on_24gb.n_cpu_moe < on_16gb.n_cpu_moe < 36
    for placement in (on_16gb, on_24gb):
        assert placement.gpu_weight_bytes + placement.cpu_weight_bytes == memory.weights
        resident = memory.total - placement.cpu_weight_bytes
        assert resident <= 23 * _GiB


def test_expert_offload_needs_room_for_the_dense_part_and_the_ram_for_experts():
    model = _gpt_oss_120b()
    memory = estimate_memory_breakdown(model, model.gguf_variants[0], 8192)
    assert plan_expert_offload(model, memory, 1 * _GiB, 50 * _GiB) is None
    assert plan_expert_offload(model, memory, 15 * _GiB, 20 * _GiB) is None


def test_expert_offload_without_a_shape_keeps_every_expert_in_ram():
    model = _gpt_oss_120b(shape=False)
    memory = estimate_memory_breakdown(model, model.gguf_variants[0], 8192)
    placement = plan_expert_offload(model, memory, 15 * _GiB, 60 * _GiB)
    assert placement is not None
    assert placement.n_cpu_moe is None
    assert placement.cpu_weight_bytes == memory.expert_weights


def test_compatibility_reports_the_llama_cpp_flag():
    model = _gpt_oss_120b()
    result = check_compatibility(model, model.gguf_variants[0], _pc(), 8192)
    assert result.fit_type == "partial_offload"
    assert result.placement is not None
    assert result.placement.kind == "expert_offload"
    assert any(
        f"--n-cpu-moe {result.placement.n_cpu_moe}" in w for w in result.warnings
    )


def test_unified_memory_never_uses_expert_offload():
    model = _gpt_oss_120b()
    mac = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M4 Max",
                vendor="apple",
                vram_bytes=48 * _GiB,
                memory_bandwidth_gbps=546.0,
                shared_memory=True,
            )
        ],
        ram_bytes=128 * _GiB,
        disk_free_bytes=2048 * _GiB,
        os="darwin",
    )
    result = check_compatibility(model, model.gguf_variants[0], mac, 8192)
    assert result.placement is not None
    assert result.placement.kind != "expert_offload"


def test_dense_overflow_uses_layer_offload():
    model = _dense_32b()
    result = check_compatibility(model, model.gguf_variants[0], _pc(), 8192)
    assert result.placement is not None
    assert result.placement.kind == "layer_offload"


# --- Speed -------------------------------------------------------------------


def _speed(model: ModelInfo, hw: HardwareInfo, context: int = 8192) -> float:
    variant = model.gguf_variants[0]
    result = check_compatibility(model, variant, hw, context)
    return estimate_tok_per_sec(
        model,
        variant,
        hw.gpus[0] if hw.gpus else None,
        result.fit_type,
        placement=result.placement,
        ram_bandwidth_gbps=hw.ram_bandwidth_gbps,
    )


def test_expert_offload_speeds_up_with_vram_and_ram_bandwidth():
    model = _gpt_oss_120b()
    base = _speed(model, _pc(vram_gb=16))
    assert _speed(model, _pc(vram_gb=24)) > base
    assert _speed(model, _pc(vram_gb=16, ram_gbps=51.2)) < base


def test_expert_offload_beats_whole_layer_offload_of_the_same_bytes():
    model = _gpt_oss_120b()
    variant = model.gguf_variants[0]
    hw = _pc()
    result = check_compatibility(model, variant, hw, 8192)
    gpu = hw.gpus[0]
    experts = estimate_tok_per_sec(
        model,
        variant,
        gpu,
        "partial_offload",
        placement=result.placement,
        ram_bandwidth_gbps=89.6,
    )
    legacy_layers = estimate_tok_per_sec(model, variant, gpu, "partial_offload")
    assert experts > legacy_layers


def test_dense_layer_offload_is_bound_by_ram_bandwidth():
    model = _dense_32b()
    fast = _speed(model, _pc(ram_gbps=120.0))
    slow = _speed(model, _pc(ram_gbps=60.0))
    assert slow < fast < 2 * slow


def test_cpu_only_speed_scales_with_ram_bandwidth():
    model = _dense_32b()
    variant = model.gguf_variants[0]
    slow = estimate_tok_per_sec(model, variant, None, "cpu_only", ram_bandwidth_gbps=50)
    fast = estimate_tok_per_sec(
        model, variant, None, "cpu_only", ram_bandwidth_gbps=100
    )
    assert fast == pytest.approx(2 * slow)


# --- Scoring -----------------------------------------------------------------


def test_expert_offload_is_judged_by_speed_not_spilled_bytes():
    moe = _gpt_oss_120b()
    assert _partial_offload_quality_factor(moe, 0.80, "expert_offload") == 0.9
    # A MoE that spills whole layers is discounted like a dense model.
    assert _partial_offload_quality_factor(moe, 0.80, "layer_offload") == 0.42
    assert _partial_offload_quality_factor(_dense_32b(), 0.80) == 0.42
