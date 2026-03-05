"""Tests for ranking behavior."""

from whichllm.engine.ranker import rank_models
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _make_hardware(vram_gb: int = 24, bandwidth_gbps: float = 80.0) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="Test GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=bandwidth_gbps,
            ),
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


def test_ranker_picks_highest_scoring_variant():
    model = ModelInfo(
        id="org/Test-8B-GGUF",
        family_id="org/Test-8B-GGUF",
        name="Test-8B-GGUF",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="test-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_500_000_000),
            GGUFVariant(filename="test-F16.gguf", quant_type="F16", file_size_bytes=5_000_000_000),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={"org/Test-8B-GGUF": 70.0},
    )
    assert results
    assert results[0].gguf_variant is not None
    assert results[0].gguf_variant.quant_type == "F16"


def test_quant_filter_applies_to_non_gguf_models():
    model = ModelInfo(
        id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        family_id="qwen2.5-14b",
        name="Qwen2.5-14B-Instruct-AWQ",
        parameter_count=14_000_000_000,
        downloads=1000,
        likes=100,
    )
    hw = _make_hardware(vram_gb=24, bandwidth_gbps=300.0)

    awq_only = rank_models([model], hw, top_n=5, quant_filter="AWQ")
    q4_only = rank_models([model], hw, top_n=5, quant_filter="Q4_K_M")

    assert len(awq_only) == 1
    assert q4_only == []


def test_popularity_has_no_effect_with_direct_benchmark():
    model_low_pop = ModelInfo(
        id="Qwen/test-8b-lowpop",
        family_id="qwen-test-8b-lowpop",
        name="test-8b-lowpop",
        parameter_count=8_000_000_000,
        downloads=100,
        likes=5,
        gguf_variants=[
            GGUFVariant(filename="test-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_500_000_000),
        ],
    )
    model_high_pop = ModelInfo(
        id="Qwen/test-8b-highpop",
        family_id="qwen-test-8b-highpop",
        name="test-8b-highpop",
        parameter_count=8_000_000_000,
        downloads=1_000_000,
        likes=10_000,
        gguf_variants=[
            GGUFVariant(filename="test-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_500_000_000),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model_low_pop, model_high_pop],
        hw,
        top_n=2,
        benchmark_scores={
            "Qwen/test-8b-lowpop": 70.0,
            "Qwen/test-8b-highpop": 70.0,
        },
    )
    assert len(results) == 2
    assert abs(results[0].quality_score - results[1].quality_score) < 1e-9


def test_general_profile_excludes_specialized_models():
    general_model = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="a-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000),
        ],
    )
    coding_model = ModelInfo(
        id="Qwen/Qwen2.5-Coder-7B-Instruct",
        family_id="qwen2.5-coder-7b",
        name="Qwen2.5-Coder-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="b-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [general_model, coding_model],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-7B-Instruct": 70.0,
            "Qwen/Qwen2.5-Coder-7B-Instruct": 75.0,
        },
        task_profile="general",
    )
    assert len(results) == 1
    assert "Coder" not in results[0].model.id


def test_require_direct_top_prioritizes_direct_benchmark():
    direct_model = ModelInfo(
        id="Qwen/direct-7b",
        family_id="qwen-direct-7b",
        name="direct-7b",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="d-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000),
        ],
    )
    estimated_model = ModelInfo(
        id="Qwen/Qwen3-9B",
        family_id="qwen3-9b",
        name="Qwen3-9B",
        parameter_count=9_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="e-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=5_000_000_000),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [direct_model, estimated_model],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/direct-7b": 65.0,
            # Qwen3のラインスコアだけ与えてestimatedを作る
            "Qwen/Qwen3-32B": 80.0,
        },
        task_profile="any",
        require_direct_top=True,
    )
    assert len(results) == 2
    assert results[0].benchmark_status == "direct"


def test_min_params_filter_excludes_small_models():
    small = ModelInfo(
        id="Qwen/Qwen2.5-3B-Instruct",
        family_id="qwen2.5-3b",
        name="Qwen2.5-3B-Instruct",
        parameter_count=3_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="s-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=1_700_000_000),
        ],
    )
    large = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="l-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [small, large],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-3B-Instruct": 90.0,
            "Qwen/Qwen2.5-7B-Instruct": 70.0,
        },
        task_profile="any",
        min_params_b=7.0,
    )
    assert len(results) == 1
    assert results[0].model.id == "Qwen/Qwen2.5-7B-Instruct"


def test_general_profile_prefers_full_gpu_when_direct_is_partial():
    partial_direct = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
    )
    full_gpu_estimated = ModelInfo(
        id="Qwen/Qwen3-9B-AWQ",
        family_id="qwen3-9b",
        name="Qwen3-9B-AWQ",
        parameter_count=9_000_000_000,
        downloads=1000,
        likes=100,
    )
    hw = _make_hardware(vram_gb=8, bandwidth_gbps=272.0)
    results = rank_models(
        [partial_direct, full_gpu_estimated],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-7B-Instruct": 80.0,  # direct
            "Qwen/Qwen3-32B": 85.0,  # line inherited for Qwen3-9B
        },
        task_profile="general",
        require_direct_top=True,
    )
    assert results
    assert results[0].fit_type == "full_gpu"
    assert results[0].model.id == "Qwen/Qwen3-9B-AWQ"


def test_family_dedup_prefers_direct_when_enabled():
    # 同一family内でfit条件が同等なら、directを優先する
    direct_base = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
    )
    estimated_variant = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct-GGUF",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(filename="x-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000),
        ],
    )
    hw = _make_hardware(vram_gb=16, bandwidth_gbps=272.0)
    results = rank_models(
        [direct_base, estimated_variant],
        hw,
        top_n=10,
        benchmark_scores={"Qwen/Qwen2.5-7B-Instruct": 75.0},
        task_profile="general",
        require_direct_top=True,
        min_params_b=7.0,
    )
    assert len(results) == 1
    assert results[0].model.id == "Qwen/Qwen2.5-7B-Instruct"
    assert results[0].benchmark_status == "direct"


def test_full_gpu_estimated_ranks_above_partial_direct():
    partial_direct = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b-a",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
    )
    full_gpu_estimated = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
    )
    hw = _make_hardware(vram_gb=8, bandwidth_gbps=272.0)
    results = rank_models(
        [partial_direct, full_gpu_estimated],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-7B-Instruct": 75.0,  # direct but partial
            "Qwen/Qwen3-32B": 85.0,  # estimated but full gpu
        },
        task_profile="general",
        require_direct_top=True,
        min_params_b=7.0,
    )
    assert len(results) == 2
    assert results[0].fit_type == "full_gpu"
    assert results[1].fit_type == "partial_offload"
