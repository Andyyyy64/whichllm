"""Tests for ranking behavior."""

from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.ranker import _partial_offload_quality_factor, rank_models
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _make_hardware(
    vram_gb: int = 24,
    bandwidth_gbps: float = 80.0,
    vendor: str = "nvidia",
    os_name: str = "linux",
    with_gpu: bool = True,
) -> HardwareInfo:
    gpus = []
    if with_gpu:
        gpus = [
            GPUInfo(
                name="Test GPU",
                vendor=vendor,
                vram_bytes=vram_gb * 1024**3,
                compute_capability=(8, 9) if vendor == "nvidia" else None,
                memory_bandwidth_gbps=bandwidth_gbps,
            ),
        ]
    return HardwareInfo(
        gpus=gpus,
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os=os_name,
    )


def test_ranker_picks_highest_scoring_variant():
    # On a fast GPU (≥800 GB/s) both quants run well above the comfort
    # threshold, so the F16 quality bonus dominates and F16 wins. On a slow
    # GPU the speed gap flips the choice — that's exercised separately.
    model = ModelInfo(
        id="org/Test-8B-GGUF",
        family_id="org/Test-8B-GGUF",
        name="Test-8B-GGUF",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="test-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_500_000_000,
            ),
            GGUFVariant(
                filename="test-F16.gguf",
                quant_type="F16",
                file_size_bytes=5_000_000_000,
            ),
        ],
    )
    hw = _make_hardware(bandwidth_gbps=900.0)
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


def test_quant_filter_matches_mxfp4_non_gguf_model():
    model = ModelInfo(
        id="openai/gpt-oss-20b-MXFP4",
        family_id="gpt-oss-20b",
        name="gpt-oss-20b-MXFP4",
        parameter_count=20_000_000_000,
        downloads=1000,
        likes=100,
    )
    # Linux + NVIDIA: non-GGUF formats are runnable, so the filter resolves.
    hw = _make_hardware(vram_gb=24, bandwidth_gbps=900.0)

    mxfp4_only = rank_models([model], hw, top_n=5, quant_filter="MXFP4")
    nvfp4_only = rank_models([model], hw, top_n=5, quant_filter="NVFP4")

    assert len(mxfp4_only) == 1
    # The label surfaced in the output table (display.py uses the same call).
    assert (
        effective_quant_type(mxfp4_only[0].model, mxfp4_only[0].gguf_variant) == "MXFP4"
    )
    assert nvfp4_only == []


def test_darwin_backend_filters_out_fp4_non_gguf_models():
    mxfp4_model = ModelInfo(
        id="openai/gpt-oss-20b-MXFP4",
        family_id="gpt-oss-20b",
        name="gpt-oss-20b-MXFP4",
        parameter_count=20_000_000_000,
        downloads=1000,
        likes=100,
    )
    hw = _make_hardware(
        vram_gb=64, bandwidth_gbps=400.0, vendor="apple", os_name="darwin"
    )
    results = rank_models([mxfp4_model], hw, top_n=10)
    assert results == []


def test_darwin_backend_filters_out_non_gguf_models():
    awq_model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b-awq",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
    )
    gguf_model = ModelInfo(
        id="Qwen/Qwen3-8B-GGUF",
        family_id="qwen3-8b-gguf",
        name="Qwen3-8B-GGUF",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="a-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    hw = _make_hardware(
        vram_gb=16, bandwidth_gbps=200.0, vendor="apple", os_name="darwin"
    )
    results = rank_models([awq_model, gguf_model], hw, top_n=10)
    assert len(results) == 1
    assert results[0].model.id == "Qwen/Qwen3-8B-GGUF"


def test_cpu_only_backend_filters_out_non_gguf_models():
    awq_model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b-awq",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
    )
    gguf_model = ModelInfo(
        id="Qwen/Qwen3-8B-GGUF",
        family_id="qwen3-8b-gguf",
        name="Qwen3-8B-GGUF",
        parameter_count=8_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="a-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    hw = _make_hardware(with_gpu=False, os_name="linux")
    results = rank_models([awq_model, gguf_model], hw, top_n=10)
    assert len(results) == 1
    assert results[0].model.id == "Qwen/Qwen3-8B-GGUF"


def _gguf_model(model_id: str, family_id: str, downloads: int) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=family_id,
        name=model_id.split("/")[-1],
        parameter_count=7_000_000_000,
        downloads=downloads,
        likes=downloads // 10,
        gguf_variants=[
            GGUFVariant(
                filename=f"{family_id}-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_500_000_000,
            ),
        ],
    )


def test_rank_models_clamps_non_positive_top_n():
    # Several distinct families so the full ranking has multiple entries; only
    # then is the slice hazard observable.
    hw = _make_hardware(vram_gb=24, bandwidth_gbps=300.0)
    models = [
        _gguf_model("org/Alpha-7B-GGUF", "alpha-7b", 1000),
        _gguf_model("org/Beta-7B-GGUF", "beta-7b", 900),
        _gguf_model("org/Gamma-7B-GGUF", "gamma-7b", 800),
    ]

    full = rank_models(models, hw, top_n=10)
    assert len(full) >= 2  # guard is only meaningful with several results

    # 0 and negative requests must yield nothing, never a slice-from-the-end
    # subset: ``results[:-1]`` would otherwise return all-but-last.
    assert rank_models(models, hw, top_n=0) == []
    assert rank_models(models, hw, top_n=-1) == []
    # Positive requests are unaffected.
    assert len(rank_models(models, hw, top_n=2)) == 2


def test_popularity_has_no_effect_with_direct_benchmark():
    model_low_pop = ModelInfo(
        id="Qwen/test-8b-lowpop",
        family_id="qwen-test-8b-lowpop",
        name="test-8b-lowpop",
        parameter_count=8_000_000_000,
        downloads=100,
        likes=5,
        gguf_variants=[
            GGUFVariant(
                filename="test-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_500_000_000,
            ),
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
            GGUFVariant(
                filename="test-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_500_000_000,
            ),
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
            GGUFVariant(
                filename="a-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
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
            GGUFVariant(
                filename="b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
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
            GGUFVariant(
                filename="d-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
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
            GGUFVariant(
                filename="e-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=5_000_000_000,
            ),
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
            GGUFVariant(
                filename="s-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=1_700_000_000,
            ),
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
            GGUFVariant(
                filename="l-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
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
    # Direct-evidence model that won't fit on 8GB even after Q4_K_M synthesis
    # (72B * 0.56 ≈ 40GB → partial_offload).
    partial_direct = ModelInfo(
        id="Qwen/Qwen2.5-72B-Instruct",
        family_id="qwen2.5-72b",
        name="Qwen2.5-72B-Instruct",
        parameter_count=72_000_000_000,
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
            "Qwen/Qwen2.5-72B-Instruct": 80.0,  # direct
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
            GGUFVariant(
                filename="x-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
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
    # Use 72B model so Q4_K_M synthesis still doesn't fit 8GB — preserves the
    # "direct evidence, but model is too big" half of this scenario.
    partial_direct = ModelInfo(
        id="Qwen/Qwen2.5-72B-Instruct",
        family_id="qwen2.5-72b",
        name="Qwen2.5-72B-Instruct",
        parameter_count=72_000_000_000,
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
            "Qwen/Qwen2.5-72B-Instruct": 75.0,  # direct but partial
            "Qwen/Qwen3-32B": 85.0,  # estimated but full gpu
        },
        task_profile="general",
        require_direct_top=True,
        min_params_b=7.0,
    )
    # The full-GPU candidate must always win over a partial-offload one of
    # comparable quality. The partial 72B may or may not be retained
    # depending on whether its sub-2 t/s estimate trips the speed floor —
    # either way the full-GPU 8B should be #1.
    assert results
    assert results[0].fit_type == "full_gpu"
    assert results[0].model.id == "Qwen/Qwen3-8B-AWQ"


def test_strong_partial_offload_not_buried_below_weaker_full_gpu():
    strong_partial = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3.6-27b",
        name="Qwen3.6-27B",
        parameter_count=27_800_000_000,
        downloads=5_300_000,
        likes=10_000,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3.6-27b-q4_k_m.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=15 * 1024**3,
            )
        ],
    )
    full_gpu_14b = ModelInfo(
        id="Qwen/Qwen3-14B",
        family_id="qwen3-14b",
        name="Qwen3-14B",
        parameter_count=14_800_000_000,
        downloads=1_600_000,
        likes=5_000,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3-14b-q5_k_m.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=9 * 1024**3,
            )
        ],
    )
    full_gpu_8b = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_200_000_000,
        downloads=11_000_000,
        likes=5_000,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3-8b-q5_k_m.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=5 * 1024**3,
            )
        ],
    )
    old_full_gpu = ModelInfo(
        id="google/gemma-2-9b-it",
        family_id="gemma-2-9b-it",
        name="gemma-2-9b-it",
        parameter_count=9_200_000_000,
        downloads=400_000,
        likes=1_000,
        gguf_variants=[
            GGUFVariant(
                filename="gemma-2-9b-q5_k_m.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=5_500_000_000,
            )
        ],
    )
    hardware = HardwareInfo(
        gpus=[
            GPUInfo(
                name="RTX 3060",
                vendor="nvidia",
                vram_bytes=12 * 1024**3,
                compute_capability=(8, 6),
                memory_bandwidth_gbps=360.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=6,
        has_avx2=True,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="windows",
    )

    results = rank_models(
        [strong_partial, full_gpu_14b, full_gpu_8b, old_full_gpu],
        hardware,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen3.6-27B": 83.5,
            "Qwen/Qwen3-14B": 66.7,
            "Qwen/Qwen3-8B": 56.1,
            "google/gemma-2-9b-it": 35.1,
        },
        task_profile="any",
    )

    ids = [r.model.id for r in results]
    assert ids.index("Qwen/Qwen3.6-27B") < ids.index("Qwen/Qwen3-8B")
    assert ids.index("Qwen/Qwen3.6-27B") < ids.index("google/gemma-2-9b-it")
    strong = next(r for r in results if r.model.id == "Qwen/Qwen3.6-27B")
    assert strong.fit_type == "partial_offload"
    assert (
        strong.quality_score
        > next(r for r in results if r.model.id == "Qwen/Qwen3-8B").quality_score
    )


def test_moe_partial_offload_penalty_uses_active_working_set():
    dense = ModelInfo(
        id="example/Dense-30B",
        family_id="dense-30b",
        name="Dense-30B",
        parameter_count=30_000_000_000,
    )
    moe = ModelInfo(
        id="example/MoE-30B-A3B",
        family_id="moe-30b-a3b",
        name="MoE-30B-A3B",
        parameter_count=30_000_000_000,
        parameter_count_active=3_000_000_000,
        is_moe=True,
    )

    assert _partial_offload_quality_factor(dense, 0.80) == 0.42
    assert _partial_offload_quality_factor(moe, 0.80) >= 0.66


def test_evidence_strict_filters_out_estimated_models():
    direct_model = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="d-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    estimated_model = ModelInfo(
        id="Qwen/Qwen3-14B-Instruct-GGUF",
        family_id="qwen3-14b",
        name="Qwen3-14B-Instruct-GGUF",
        parameter_count=14_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="e-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=8_000_000_000,
            ),
        ],
    )
    hw = _make_hardware(vram_gb=24, bandwidth_gbps=300.0)
    results = rank_models(
        [direct_model, estimated_model],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-7B-Instruct": 70.0,
            "Qwen/Qwen3-32B-Instruct": 85.0,  # Qwen3-14B には line 推定が入る
        },
        task_profile="any",
        evidence_filter="strict",
    )
    assert len(results) == 1
    assert results[0].model.id == "Qwen/Qwen2.5-7B-Instruct"
    assert results[0].benchmark_status == "direct"


def test_evidence_base_keeps_base_model_match_and_drops_line_interp():
    direct_model = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
    )
    base_match_model = ModelInfo(
        id="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        family_id="gemma-3-27b",
        name="gemma-3-27b-it-GPTQ-4b-128g",
        parameter_count=27_000_000_000,
        downloads=1000,
        likes=100,
        base_model="google/gemma-3-27b-it",
    )
    line_interp_model = ModelInfo(
        id="Qwen/Qwen3-14B-Instruct-GGUF",
        family_id="qwen3-14b",
        name="Qwen3-14B-Instruct-GGUF",
        parameter_count=14_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="f-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=8_000_000_000,
            ),
        ],
    )
    hw = _make_hardware(vram_gb=24, bandwidth_gbps=300.0)
    results = rank_models(
        [direct_model, base_match_model, line_interp_model],
        hw,
        top_n=10,
        benchmark_scores={
            "Qwen/Qwen2.5-7B-Instruct": 70.0,
            "google/gemma-3-27b-it": 82.0,
            "Qwen/Qwen3-32B-Instruct": 85.0,
        },
        task_profile="any",
        evidence_filter="base",
    )
    ids = {r.model.id for r in results}
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    assert "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g" in ids
    assert "Qwen/Qwen3-14B-Instruct-GGUF" not in ids


def test_unknown_speed_heavy_partial_offload_does_not_top_rank():
    heavy_partial = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3.6-27b",
        name="Qwen3.6-27B",
        parameter_count=27_800_000_000,
        downloads=1_000_000,
        likes=10_000,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3.6-27b-q8_0.gguf",
                quant_type="Q8_0",
                file_size_bytes=29_500_000_000,
            )
        ],
    )
    full_gpu = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        downloads=500_000,
        likes=5_000,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3-8b-q4_k_m.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    hardware = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Unknown 6GB NVIDIA GPU",
                vendor="nvidia",
                vram_bytes=6 * 1024**3,
                compute_capability=(8, 6),
                memory_bandwidth_gbps=None,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="windows",
    )

    results = rank_models(
        [heavy_partial, full_gpu],
        hardware,
        top_n=2,
        benchmark_scores={
            "Qwen/Qwen3.6-27B": 84.0,
            "Qwen/Qwen3-8B": 62.0,
        },
    )

    assert results
    assert results[0].model.id == "Qwen/Qwen3-8B"
    assert results[0].fit_type == "full_gpu"
    heavy = next((r for r in results if r.model.id == "Qwen/Qwen3.6-27B"), None)
    if heavy is not None:
        assert heavy.fit_type == "partial_offload"
        assert heavy.offload_ratio >= 0.70
        assert heavy.estimated_tok_per_sec == 0.0


def test_fit_filter_full_gpu_excludes_partial_offload_and_cpu_only():
    partial = ModelInfo(
        id="org/Test-30B-GGUF",
        family_id="test-30b",
        name="Test-30B-GGUF",
        parameter_count=30_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="test-30b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    full = ModelInfo(
        id="org/Test-7B-GGUF",
        family_id="test-7b",
        name="Test-7B-GGUF",
        parameter_count=7_000_000_000,
        downloads=900,
        likes=90,
        gguf_variants=[
            GGUFVariant(
                filename="test-7b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    cpu_only = ModelInfo(
        id="org/Test-60B-GGUF",
        family_id="test-60b",
        name="Test-60B-GGUF",
        parameter_count=60_000_000_000,
        downloads=800,
        likes=80,
        gguf_variants=[
            GGUFVariant(
                filename="test-60b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=36_000_000_000,
            )
        ],
    )
    hw = _make_hardware(vram_gb=8, bandwidth_gbps=300.0)
    results = rank_models(
        [partial, full, cpu_only],
        hw,
        top_n=10,
        fit_filter="full_gpu",
        task_profile="any",
        require_direct_top=False,
    )

    assert [r.model.id for r in results] == ["org/Test-7B-GGUF"]
    assert results[0].fit_type == "full_gpu"


def test_fit_filter_full_gpu_returns_empty_when_no_full_gpu_candidate():
    partial = ModelInfo(
        id="org/Test-30B-GGUF",
        family_id="test-30b",
        name="Test-30B-GGUF",
        parameter_count=30_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="test-30b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    hw = _make_hardware(vram_gb=8, bandwidth_gbps=300.0)
    results = rank_models(
        [partial],
        hw,
        top_n=10,
        fit_filter="full_gpu",
        task_profile="any",
        require_direct_top=False,
    )

    assert results == []


def test_multi_gpu_speed_confidence_is_low():
    from whichllm.engine.performance import estimate_tok_per_sec

    model = ModelInfo(
        id="org/Test-34B-GGUF",
        family_id="org/Test-34B-GGUF",
        name="Test-34B-GGUF",
        parameter_count=34_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="test-34b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=22 * 1024**3,
            )
        ],
    )
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1008.0,
            ),
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1008.0,
            ),
        ],
        cpu_name="Test CPU",
        cpu_cores=16,
        has_avx2=True,
        ram_bytes=128 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )

    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={"org/Test-34B-GGUF": 70.0},
    )

    assert results
    assert results[0].fit_type == "full_gpu"
    assert results[0].uses_multi_gpu is True
    assert results[0].speed_confidence == "low"
    single_gpu_speed = estimate_tok_per_sec(
        model,
        model.gguf_variants[0],
        hw.gpus[0],
        "full_gpu",
    )
    assert results[0].estimated_tok_per_sec == single_gpu_speed * 0.70
    assert any("Multi-GPU speed depends" in note for note in results[0].speed_notes)


def test_benchmark_source_and_confidence_exposed_for_direct():
    model = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="a-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={"Qwen/Qwen2.5-7B-Instruct": 75.0},
        task_profile="any",
    )
    assert results
    assert results[0].benchmark_status == "direct"
    assert results[0].benchmark_source == "direct"
    assert results[0].benchmark_confidence == 1.0


def test_benchmark_source_and_confidence_exposed_for_estimated():
    model = ModelInfo(
        id="Qwen/Qwen3-14B-Instruct-GGUF",
        family_id="qwen3-14b",
        name="Qwen3-14B-Instruct-GGUF",
        parameter_count=14_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="e-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=8_000_000_000,
            ),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={"Qwen/Qwen3-32B-Instruct": 85.0},
        task_profile="any",
    )
    assert results
    assert results[0].benchmark_status == "estimated"
    assert results[0].benchmark_source == "line_interp"
    assert 0.0 < results[0].benchmark_confidence < 1.0


def test_benchmark_source_and_confidence_exposed_for_self_reported():
    model = ModelInfo(
        id="someorg/mystery-7B",
        family_id="mystery-7b",
        name="mystery-7B",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        benchmark_scores={"hf_eval": 72.0},
        gguf_variants=[
            GGUFVariant(
                filename="m-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={},
        task_profile="any",
    )
    assert results
    assert results[0].benchmark_status == "self_reported"
    assert results[0].benchmark_source == "self_reported"
    assert results[0].benchmark_confidence > 0.0


def test_benchmark_source_and_confidence_exposed_for_none():
    model = ModelInfo(
        id="someorg/unknown-7B",
        family_id="unknown-7b",
        name="unknown-7B",
        parameter_count=7_000_000_000,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="u-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            ),
        ],
    )
    hw = _make_hardware()
    results = rank_models(
        [model],
        hw,
        top_n=1,
        benchmark_scores={},
        task_profile="any",
    )
    assert results
    assert results[0].benchmark_status == "none"
    assert results[0].benchmark_source == "none"
    assert results[0].benchmark_confidence == 0.0


def test_ctx_penalty_demotes_non_fitting():
    models = [
        ModelInfo(
            id="org/LongCtx-8B",
            family_id="longctx-8b",
            name="LongCtx-8B",
            parameter_count=8_000_000_000,
            context_length=131072,
            downloads=900,
            likes=90,
            gguf_variants=[
                GGUFVariant(
                    filename="long-Q4_K_M.gguf",
                    quant_type="Q4_K_M",
                    file_size_bytes=4_500_000_000,
                ),
            ],
        ),
        ModelInfo(
            id="org/ShortCtx-8B",
            family_id="shortctx-8b",
            name="ShortCtx-8B",
            parameter_count=8_000_000_000,
            context_length=8192,
            downloads=1000,
            likes=100,
            gguf_variants=[
                GGUFVariant(
                    filename="short-Q4_K_M.gguf",
                    quant_type="Q4_K_M",
                    file_size_bytes=4_500_000_000,
                ),
            ],
        ),
    ]
    scores = {
        "org/LongCtx-8B": 74.0,
        "org/ShortCtx-8B": 76.0,
    }
    hw = _make_hardware(bandwidth_gbps=900.0)

    results = rank_models(
        models,
        hw,
        context_length=32768,
        top_n=2,
        benchmark_scores=scores,
        require_direct_top=False,
        task_profile="any",
    )

    assert len(results) == 2
    assert results[0].model.family_id == "longctx-8b"
    assert results[0].context_fits is True
    assert results[1].context_fits is False
