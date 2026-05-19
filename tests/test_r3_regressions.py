"""Regression tests for the Round 3 fixes (whichllm 0.5.1).

Each test reproduces a specific bug class that the 0.5.1-dev ranker
exhibited and that was found by stress-testing previously unexercised
axes (Apple --gpu, family inheritance order, grouper base selection,
reasoning-model surfacing):

- R3-1: ``--gpu "M1"`` fuzzy-matched the 1997 ATI Rage Mobility-M1
  (vendor=amd); ``--gpu "M3 Max"`` fell through to vendor=nvidia.
- R3-2: a 6.6B "imatrix-aligned" / MTP-head fork inherited its 158B
  base's benchmark via family/base_model lookup.
- R3-3: a high-download downstream fork overrode the official upstream
  model as the family base, corrupting ``family_id``.
- R3-4/R3-5: reasoning lines (QwQ-32B, DeepSeek-R1-Distill) had no
  curated benchmark entry and never surfaced in the ranking.
"""

from __future__ import annotations

from whichllm.engine.ranker import rank_models
from whichllm.hardware.gpu_simulator import create_synthetic_gpu
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.benchmark import _params_compatible
from whichllm.models.benchmark_sources.aa_index import AA_INDEX_FALLBACK_2026_05_14
from whichllm.models.benchmark_sources.livebench import LIVEBENCH_RAW_DATA
from whichllm.models.benchmark_sources.vision import VISION_FALLBACK_2026_05
from whichllm.models.grouper import group_models
from whichllm.models.types import GGUFVariant, ModelInfo


def _hw(
    vram_gb: int = 24,
    bandwidth_gbps: float = 1000.0,
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
            )
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


def _gguf(quant: str, size_gb: float) -> GGUFVariant:
    return GGUFVariant(
        filename=f"model-{quant}.gguf",
        quant_type=quant,
        file_size_bytes=int(size_gb * 1e9),
    )


# ------------------------------------------------------------------ R3-1


class TestAppleSiliconSimulator:
    """``--gpu`` must recognize Apple Silicon instead of fuzzy-matching
    discrete-GPU database entries."""

    def test_m1_default_is_apple_not_ati_rage_mobility(self):
        gpu = create_synthetic_gpu("M1")
        assert gpu.vendor == "apple", (
            f"M1 must be Apple, got vendor={gpu.vendor!r} "
            f"name={gpu.name!r} (regression: fuzzy-matched ATI Rage "
            "Mobility-M1)"
        )
        assert "rage" not in gpu.name.lower()
        # Default unified memory for the base M1 is 8 GB.
        assert gpu.vram_bytes == 8 * 1024**3
        assert gpu.memory_bandwidth_gbps == 68.25

    def test_m3_max_vram_override_apple_400gbps(self):
        gpu = create_synthetic_gpu("M3 Max", vram_override_gb=64)
        assert gpu.vendor == "apple", (
            f"M3 Max must be Apple, got vendor={gpu.vendor!r} "
            "(regression: fell through to nvidia default)"
        )
        assert gpu.vram_bytes == 64 * 1024**3
        assert gpu.memory_bandwidth_gbps == 400.0

    def test_m2_ultra_192gb_apple_800gbps(self):
        gpu = create_synthetic_gpu("M2 Ultra", vram_override_gb=192)
        assert gpu.vendor == "apple"
        assert gpu.vram_bytes == 192 * 1024**3
        assert gpu.memory_bandwidth_gbps == 800.0

    def test_apple_chip_compact_form_is_recognized(self):
        # Users type "M2Max" / "m2 max" / "M2 MAX" interchangeably.
        for name in ("M2Max", "m2 max", "M2 MAX"):
            gpu = create_synthetic_gpu(name, vram_override_gb=32)
            assert gpu.vendor == "apple", f"{name!r} not recognized as Apple"
            assert gpu.memory_bandwidth_gbps == 400.0

    def test_longest_match_wins_m2_ultra_not_m2(self):
        # "M2 Ultra" must not be swallowed by the "M2" entry (100 GB/s).
        gpu = create_synthetic_gpu("M2 Ultra", vram_override_gb=128)
        assert gpu.memory_bandwidth_gbps == 800.0
        assert gpu.memory_bandwidth_gbps != 100.0


# ------------------------------------------------------------------ R3-2


class TestFamilySizeInheritance:
    """A small fork must not inherit a much larger base model's
    benchmark score."""

    def test_params_compatible_rejects_25x_mismatch(self):
        # 6.6B vs an id that encodes 158B → ratio 0.04, must reject.
        assert _params_compatible(6.6, "org/Some-Model-158B") is False

    def test_params_compatible_accepts_same_size_quant(self):
        # A Q4 repack of an 8B model is still 8B → must inherit.
        assert _params_compatible(7.8, "org/Llama-3-8B-GGUF") is True

    def test_params_compatible_permissive_when_no_actual_size(self):
        # No actual size → cannot judge, must not block (avoids
        # false-negatives that would erase legitimate inheritance).
        assert _params_compatible(None, "org/Model-70B") is True

    def test_params_compatible_permissive_when_ref_has_no_size(self):
        # Base id without a parseable size (e.g. DeepSeek-V4-Flash) →
        # the function alone cannot guard; the ranker's
        # family_dominant_params check is the backstop (next test).
        assert _params_compatible(6.6, "deepseek-ai/DeepSeek-V4-Flash") is True

    def test_ranker_drops_tiny_fork_inheriting_huge_base(self):
        """The real bug: jedisct1/DeepSeek-V4-Flash-imatrix-aligned
        (6.6B) shared family_id with the 158B base and inherited its
        leaderboard score, landing in the CPU-only top 5."""
        base = ModelInfo(
            id="org/DeepSeek-Vx-Flash",
            family_id="deepseek-vx-flash",
            name="DeepSeek-Vx-Flash",
            parameter_count=158_000_000_000,
            downloads=1_000_000,
            likes=1000,
            gguf_variants=[],  # safetensors-only official
        )
        tiny_fork = ModelInfo(
            id="fork/DeepSeek-Vx-Flash-mtp-aligned",
            family_id="deepseek-vx-flash",
            name="DeepSeek-Vx-Flash-mtp-aligned",
            parameter_count=6_600_000_000,
            downloads=0,
            likes=0,
            base_model="org/DeepSeek-Vx-Flash",
            gguf_variants=[_gguf("Q8_0", 7.0)],
        )
        # External leaderboard only knows the 158B base.
        scores = {"org/DeepSeek-Vx-Flash": 92.0}
        # Tiny VRAM: the 158B base cannot run; only the 6.6B fork fits.
        hw = _hw(vram_gb=12)
        ranked = rank_models(
            [base, tiny_fork],
            hw,
            benchmark_scores=scores,
            require_direct_top=False,
        )
        # With the guard the fork's family/base inheritance is rejected
        # entirely (status=none, score≈20). Without it the fork inherits
        # the 158B base at confidence 0.6 (status=estimated, score≈59).
        # The thresholds below sit firmly between those two regimes so
        # the test goes red the instant the guard is removed.
        tiny_res = next((r for r in ranked if r.model.id == tiny_fork.id), None)
        assert tiny_res is not None, "tiny fork should still be listed"
        assert tiny_res.benchmark_status == "none", (
            "tiny fork inherited a benchmark from its 158B base "
            f"(status={tiny_res.benchmark_status!r}); the "
            "family_dominant_params guard is not working"
        )
        assert tiny_res.quality_score < 30.0, (
            "tiny fork's score reflects inherited 158B evidence "
            f"(got {tiny_res.quality_score:.1f}, expected <30)"
        )


# ------------------------------------------------------------------ R3-3


class TestGrouperUpstreamBase:
    """Family base selection must follow the base_model graph, not raw
    download counts."""

    def test_official_upstream_wins_over_more_downloaded_fork(self):
        official = ModelInfo(
            id="Qwen/Qwen3-4B-Thinking-2507",
            family_id="",
            name="Qwen3-4B-Thinking-2507",
            parameter_count=4_000_000_000,
            downloads=494_000,
            base_model=None,
        )
        popular_fork = ModelInfo(
            id="prefeitura-rio/Rio-3.0-Open-Mini",
            family_id="",
            name="Rio-3.0-Open-Mini",
            parameter_count=4_000_000_000,
            downloads=1_300_000,  # more than the official base
            base_model="Qwen/Qwen3-4B-Thinking-2507",
        )
        gguf_fork = ModelInfo(
            id="MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF",
            family_id="",
            name="Qwen3-4B-Thinking-2507-GGUF",
            parameter_count=4_000_000_000,
            downloads=26_000,
            base_model="Qwen/Qwen3-4B-Thinking-2507",
            gguf_variants=[_gguf("Q4_K_M", 2.4)],
        )
        families = group_models([popular_fork, official, gguf_fork])
        # All three collapse into one family.
        assert len(families) == 1
        fam = families[0]
        assert fam.base_model.id == "Qwen/Qwen3-4B-Thinking-2507", (
            f"family base is {fam.base_model.id!r}; the popular fork "
            "overrode the official upstream (regression R3-3)"
        )
        # Every member must carry the upstream-derived family_id.
        all_ids = {fam.base_model.id} | {v.id for v in fam.variants}
        assert "Qwen/Qwen3-4B-Thinking-2507" in all_ids
        for m in [official, popular_fork, gguf_fork]:
            assert m.family_id == official.family_id
            assert "rio" not in m.family_id

    def test_falls_back_to_downloads_without_upstream_reference(self):
        # No member references another's base_model → keep the prior
        # "most downloads, no GGUF" behaviour.
        a = ModelInfo(
            id="orgA/Model-7B",
            family_id="",
            name="Model-7B",
            parameter_count=7_000_000_000,
            downloads=500,
        )
        b = ModelInfo(
            id="orgB/Model-7B",
            family_id="",
            name="Model-7B",
            parameter_count=7_000_000_000,
            downloads=5000,
        )
        families = group_models([a, b])
        assert len(families) == 1
        assert families[0].base_model.id == "orgB/Model-7B"


# --------------------------------------------------------------- R3-4/5


class TestReasoningSurface:
    """Reasoning/thinking lines must have curated benchmark anchors and
    surface in the ranking."""

    def test_qwq32b_has_curated_benchmarks(self):
        assert "Qwen/QwQ-32B" in LIVEBENCH_RAW_DATA
        assert "Qwen/QwQ-32B" in AA_INDEX_FALLBACK_2026_05_14

    def test_r1_distill_family_has_curated_benchmarks(self):
        for mid in (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ):
            assert mid in LIVEBENCH_RAW_DATA, f"{mid} missing in LB"
            assert mid in AA_INDEX_FALLBACK_2026_05_14, f"{mid} missing in AA"

    def test_qwq32b_surfaces_with_curated_score(self):
        qwq = ModelInfo(
            id="Qwen/QwQ-32B",
            family_id="qwq-32b",
            name="QwQ-32B",
            parameter_count=32_800_000_000,
            downloads=64_000,
            gguf_variants=[_gguf("Q4_K_M", 20.0)],
        )
        filler = ModelInfo(
            id="org/Generic-7B",
            family_id="generic-7b",
            name="Generic-7B",
            parameter_count=7_000_000_000,
            downloads=10,
            gguf_variants=[_gguf("Q4_K_M", 4.5)],
        )
        scores = {"Qwen/QwQ-32B": 57.0}
        hw = _hw(vram_gb=48)
        ranked = rank_models(
            [qwq, filler], hw, benchmark_scores=scores, require_direct_top=False
        )
        ids = [r.model.id for r in ranked]
        assert "Qwen/QwQ-32B" in ids, "QwQ-32B did not surface in ranking"
        qwq_res = next(r for r in ranked if r.model.id == "Qwen/QwQ-32B")
        assert qwq_res.benchmark_status == "direct"
        assert qwq_res.quality_score > 0


# ----------------------------------------------------------------- R3-6


class TestVisionGenerationOrder:
    """``--profile vision`` had no VLM benchmark source, so only a
    two-generations-old Qwen2-VL-7B had a direct hit and outranked the
    current Qwen3-VL-32B even on an 80 GB GPU."""

    def test_curated_vision_scores_respect_generation(self):
        v = VISION_FALLBACK_2026_05
        # The whole point: newest generation must outscore the oldest.
        assert (
            v["Qwen/Qwen3-VL-32B-Instruct"]
            > v["Qwen/Qwen2.5-VL-32B-Instruct"]
            > v["Qwen/Qwen2-VL-7B-Instruct"]
        )
        # Two-generations-old 7B must sit in the low band so it cannot
        # win a vision ranking by virtue of being the only direct hit.
        assert v["Qwen/Qwen2-VL-7B-Instruct"] <= 35.0

    def test_qwen3_vl_outranks_legacy_qwen2_vl_on_vision_profile(self):
        new_vlm = ModelInfo(
            id="Qwen/Qwen3-VL-32B-Instruct",
            family_id="qwen3-vl-32b",
            name="Qwen3-VL-32B-Instruct",
            parameter_count=33_400_000_000,
            downloads=1_500_000,
            gguf_variants=[_gguf("Q4_K_M", 20.0)],
        )
        legacy_vlm = ModelInfo(
            id="Qwen/Qwen2-VL-7B-Instruct",
            family_id="qwen2-vl-7b",
            name="Qwen2-VL-7B-Instruct",
            parameter_count=8_300_000_000,
            downloads=2_000_000,  # more downloads — must NOT win
            gguf_variants=[_gguf("Q4_K_M", 5.0)],
        )
        # Feed the curated vision scores exactly as the merge would.
        scores = {
            "Qwen/Qwen3-VL-32B-Instruct": VISION_FALLBACK_2026_05[
                "Qwen/Qwen3-VL-32B-Instruct"
            ],
            "Qwen/Qwen2-VL-7B-Instruct": VISION_FALLBACK_2026_05[
                "Qwen/Qwen2-VL-7B-Instruct"
            ],
        }
        hw = _hw(vram_gb=80)
        ranked = rank_models(
            [legacy_vlm, new_vlm],
            hw,
            benchmark_scores=scores,
            task_profile="vision",
            require_direct_top=False,
        )
        assert ranked, "no vision models ranked"
        assert ranked[0].model.id == "Qwen/Qwen3-VL-32B-Instruct", (
            "legacy Qwen2-VL-7B outranked the current Qwen3-VL-32B on "
            f"the vision profile (got {ranked[0].model.id})"
        )


# ----------------------------------------------------------------- R3-7


class TestApplePartialOffloadPenalty:
    """Partial-offload penalty must respect memory architecture: Apple
    unified memory has no PCIe cliff, so the discrete 0.45x penalty made
    DeepSeek-R1-class models on M2/M3 Ultra report ~1.7 t/s vs a
    real-world 4-15."""

    def _model_variant(self):
        m = ModelInfo(
            id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            family_id="deepseek-r1-distill-qwen-32b",
            name="DeepSeek-R1-Distill-Qwen-32B",
            parameter_count=32_800_000_000,
        )
        v = _gguf("Q4_K_M", 20.0)
        return m, v

    def test_apple_partial_offload_keeps_most_of_full_speed(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        m, v = self._model_variant()
        apple = GPUInfo(
            name="M2 Ultra",
            vendor="apple",
            vram_bytes=192 * 1024**3,
            memory_bandwidth_gbps=800.0,
        )
        full = estimate_tok_per_sec(m, v, apple, "full_gpu")
        partial = estimate_tok_per_sec(m, v, apple, "partial_offload")
        assert full > 0
        ratio = partial / full
        # 0.85 with the fix; 0.45 if the vendor branch is removed. The
        # 0.7 threshold sits between the two regimes.
        assert ratio > 0.7, (
            f"Apple partial-offload ratio {ratio:.2f} — the discrete "
            "0.45x PCIe penalty is being wrongly applied to unified memory"
        )

    def test_discrete_partial_offload_still_takes_pcie_penalty(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        m, v = self._model_variant()
        nvidia = GPUInfo(
            name="RTX 4090",
            vendor="nvidia",
            vram_bytes=24 * 1024**3,
            compute_capability=(8, 9),
            memory_bandwidth_gbps=1008.0,
        )
        full = estimate_tok_per_sec(m, v, nvidia, "full_gpu")
        partial = estimate_tok_per_sec(m, v, nvidia, "partial_offload")
        assert full > 0
        ratio = partial / full
        # Discrete GPUs must keep the steep PCIe penalty (~0.45).
        assert ratio < 0.6, (
            f"discrete partial-offload ratio {ratio:.2f} — the PCIe "
            "penalty is too weak; offload should hurt on NVIDIA/AMD"
        )

    def test_apple_partial_faster_than_discrete_partial_same_bandwidth(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        m, v = self._model_variant()
        # Same nominal bandwidth so the only difference is the
        # architecture-aware offload penalty.
        apple = GPUInfo(
            name="Apple",
            vendor="apple",
            vram_bytes=192 * 1024**3,
            memory_bandwidth_gbps=900.0,
        )
        nvidia = GPUInfo(
            name="NVIDIA",
            vendor="nvidia",
            vram_bytes=24 * 1024**3,
            compute_capability=(8, 9),
            memory_bandwidth_gbps=900.0,
        )
        a = estimate_tok_per_sec(m, v, apple, "partial_offload")
        n = estimate_tok_per_sec(m, v, nvidia, "partial_offload")
        assert a > n, (
            "Apple unified-memory partial offload should beat discrete "
            f"PCIe-bound partial offload at equal bandwidth ({a:.1f} vs "
            f"{n:.1f})"
        )


# ----------------------------------------------------------------- R3-8


class TestMoESpeedEstimation:
    """MoE speed estimation should use active params without letting
    high-bandwidth GPUs turn sparse models into unrealistic outliers."""

    def test_qwen3_next_strix_halo_matches_reported_generation_speed(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        model = ModelInfo(
            id="Qwen/Qwen3-Next-80B-A3B-Instruct",
            family_id="qwen3-next-80b-a3b",
            name="Qwen3-Next-80B-A3B-Instruct",
            parameter_count=79_670_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
        )
        variant = GGUFVariant(
            filename="qwen3next-q4_k_m.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(45.17 * 1024**3),
        )
        strix_halo = GPUInfo(
            name="STRXLGEN",
            vendor="amd",
            vram_bytes=0,
            memory_bandwidth_gbps=256.0,
            shared_memory=True,
        )

        speed = estimate_tok_per_sec(model, variant, strix_halo, "full_gpu")

        assert 40.0 <= speed <= 50.0

    def test_unknown_ultra_sparse_moe_uses_active_params_on_strix_halo(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        model = ModelInfo(
            id="unknown/Experimental-80B-A3B",
            family_id="experimental-80b-a3b",
            name="Experimental-80B-A3B",
            parameter_count=79_670_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
        )
        variant = GGUFVariant(
            filename="experimental-q4_k_m.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(45.17 * 1024**3),
        )
        strix_halo = GPUInfo(
            name="STRXLGEN",
            vendor="amd",
            vram_bytes=0,
            memory_bandwidth_gbps=256.0,
            shared_memory=True,
        )

        speed = estimate_tok_per_sec(model, variant, strix_halo, "full_gpu")

        assert 40.0 <= speed <= 50.0

    def test_qwen3_30b_a3b_strix_halo_no_longer_uses_legacy_floor(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        model = ModelInfo(
            id="Qwen/Qwen3-30B-A3B",
            family_id="qwen3-30b-a3b",
            name="Qwen3-30B-A3B",
            parameter_count=30_000_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
        )
        variant = GGUFVariant(
            filename="qwen3-30b-q4_k_m.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(17.1 * 1024**3),
        )
        strix_halo = GPUInfo(
            name="STRXLGEN",
            vendor="amd",
            vram_bytes=0,
            memory_bandwidth_gbps=256.0,
            shared_memory=True,
        )

        speed = estimate_tok_per_sec(model, variant, strix_halo, "full_gpu")

        assert 50.0 <= speed <= 70.0

    def test_high_bandwidth_gpu_keeps_moe_kernel_floor(self):
        from whichllm.engine.performance import estimate_tok_per_sec

        model = ModelInfo(
            id="Qwen/Qwen3-30B-A3B",
            family_id="qwen3-30b-a3b",
            name="Qwen3-30B-A3B",
            parameter_count=30_000_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
        )
        variant = _gguf("Q5_K_M", 20.6)
        rtx_4090 = GPUInfo(
            name="RTX 4090",
            vendor="nvidia",
            vram_bytes=24 * 1024**3,
            compute_capability=(8, 9),
            memory_bandwidth_gbps=1008.0,
        )

        speed = estimate_tok_per_sec(model, variant, rtx_4090, "full_gpu")

        assert 100.0 <= speed <= 150.0

    def test_deepseek_v4_flash_synthetic_q4_does_not_fit_strix_halo_96gb(self):
        deepseek = ModelInfo(
            id="deepseek-ai/DeepSeek-V4-Flash",
            family_id="deepseek-v4-flash",
            name="DeepSeek-V4-Flash",
            parameter_count=284_000_000_000,
            parameter_count_active=13_000_000_000,
            is_moe=True,
            downloads=1_000_000,
            gguf_variants=[],
        )
        qwen_dense = ModelInfo(
            id="Qwen/Qwen3.6-27B",
            family_id="qwen3.6-27b",
            name="Qwen3.6-27B",
            parameter_count=27_000_000_000,
            downloads=100_000,
            gguf_variants=[],
        )
        hardware = HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Strix Halo",
                    vendor="amd",
                    vram_bytes=96 * 1024**3,
                    memory_bandwidth_gbps=256.0,
                    shared_memory=True,
                )
            ],
            cpu_name="Ryzen AI MAX+ 395",
            cpu_cores=16,
            ram_bytes=128 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            os="linux",
        )

        results = rank_models(
            [deepseek, qwen_dense],
            hardware,
            top_n=5,
            quant_filter="Q4_K_M",
            benchmark_scores={
                "deepseek-ai/DeepSeek-V4-Flash": 87.0,
                "Qwen/Qwen3.6-27B": 84.0,
            },
        )

        ids = [r.model.id for r in results]
        assert "deepseek-ai/DeepSeek-V4-Flash" not in ids
        assert "Qwen/Qwen3.6-27B" in ids


class TestSpeedUncertainty:
    def test_strix_halo_moe_speed_is_medium_confidence_with_range(self):
        from whichllm.engine.performance import estimate_speed_uncertainty

        model = ModelInfo(
            id="unknown/Experimental-80B-A3B",
            family_id="experimental-80b-a3b",
            name="Experimental-80B-A3B",
            parameter_count=79_670_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
        )
        variant = GGUFVariant(
            filename="experimental-q4_k_m.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(45.17 * 1024**3),
        )
        strix_halo = GPUInfo(
            name="Strix Halo",
            vendor="amd",
            vram_bytes=96 * 1024**3,
            memory_bandwidth_gbps=256.0,
            shared_memory=True,
        )

        confidence, speed_range, notes = estimate_speed_uncertainty(
            model, variant, strix_halo, "full_gpu", 48.0
        )

        assert confidence == "medium"
        assert speed_range == (28.8, 76.8)
        assert any("shared-memory APU" in note for note in notes)

    def test_apple_silicon_moe_speed_is_low_confidence(self):
        from whichllm.engine.performance import estimate_speed_uncertainty

        model = ModelInfo(
            id="google/gemma-4-26B-A4B-it",
            family_id="gemma-4-26b-a4b-it",
            name="gemma-4-26B-A4B-it",
            parameter_count=26_000_000_000,
            parameter_count_active=3_800_000_000,
            is_moe=True,
        )
        variant = _gguf("Q4_K_M", 15.0)
        apple = GPUInfo(
            name="M3 Max",
            vendor="apple",
            vram_bytes=96 * 1024**3,
            memory_bandwidth_gbps=400.0,
            shared_memory=True,
        )

        confidence, speed_range, notes = estimate_speed_uncertainty(
            model, variant, apple, "full_gpu", 30.0
        )

        assert confidence == "low"
        assert speed_range == (10.5, 60.0)
        assert any("Metal/MLX" in note for note in notes)

    def test_synthetic_gguf_rank_result_exposes_speed_uncertainty(self):
        model = ModelInfo(
            id="Qwen/Qwen3-30B-A3B",
            family_id="qwen3-30b-a3b",
            name="Qwen3-30B-A3B",
            parameter_count=30_000_000_000,
            parameter_count_active=3_000_000_000,
            is_moe=True,
            downloads=1_000_000,
        )
        hardware = HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Strix Halo",
                    vendor="amd",
                    vram_bytes=96 * 1024**3,
                    memory_bandwidth_gbps=256.0,
                    shared_memory=True,
                )
            ],
            cpu_name="Ryzen AI MAX+ 395",
            cpu_cores=16,
            ram_bytes=128 * 1024**3,
            disk_free_bytes=500 * 1024**3,
            os="linux",
        )

        result = rank_models(
            [model],
            hardware,
            top_n=1,
            quant_filter="Q4_K_M",
            benchmark_scores={"Qwen/Qwen3-30B-A3B": 80.0},
        )[0]

        assert result.speed_confidence == "medium"
        assert result.speed_range_tok_per_sec is not None
        assert result.speed_range_tok_per_sec[0] < result.estimated_tok_per_sec
        assert result.speed_range_tok_per_sec[1] > result.estimated_tok_per_sec
        assert any("synthetic GGUF" in note for note in result.speed_notes)
