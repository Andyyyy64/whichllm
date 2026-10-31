"""Regression tests for the Phase 1-3 ranking-quality fixes.

Each test reproduces a specific bug class that the original whichllm 0.5.0
ranker exhibited on 2026-05-14:

- Q1_0 / Q2_0 derivatives must not surface as full GPU candidates by
  default (P1: extreme-quant exclusion).
- Self-reported (hf_eval) values must not beat independent leaderboard
  evidence (P3-1: self_reported tier).
- CI / research orgs (gpt2, opt-125m, tiny-Qwen2ForCausalLM) must never
  occupy a ranking slot (P2-2: excluded orgs).
- Newer generation in the same family should beat the older one when
  benchmark data is unavailable or weak (P2-1: lineage bonus).
- Heretic / abliterated / uncensored derivatives should rank below their
  clean base/converter counterparts (P2-3: derivative penalty).
"""

from __future__ import annotations

from whichllm.engine.ranker import (
    _SOURCE_WEIGHTS,
    _derivative_name_penalty,
    _generation_bonus,
    _is_excluded_model,
    rank_models,
)
from whichllm.hardware.types import GPUInfo, HardwareInfo
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


# ---------------------------------------------------------------------- P1


def test_p1_q1_0_only_repo_is_severely_penalized_when_no_quant_filter():
    """An 8B repo that only ships Q1_0 may still appear as a fallback (since
    it's the only thing the repo offers) but its quality score must be
    crushed by the combined Q1_0 (-55%) + self_reported (-45%) penalties so
    it cannot beat a normal Q4_K_M alternative."""
    bonsai = ModelInfo(
        id="prism-ml/Bonsai-8B-gguf",
        family_id="bonsai-8b",
        name="Bonsai-8B",
        parameter_count=8_000_000_000,
        downloads=48_000,
        likes=0,
        gguf_variants=[_gguf("Q1_0", 4.6)],
        benchmark_scores={"hf_eval": 88.0},
    )
    normal = ModelInfo(
        id="bartowski/Qwen3-8B-GGUF",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        downloads=58_000,
        likes=20,
        base_model="Qwen/Qwen3-8B",
        gguf_variants=[_gguf("Q4_K_M", 4.7)],
    )
    results = rank_models(
        [bonsai, normal],
        _hw(),
        top_n=5,
        benchmark_scores={"Qwen/Qwen3-8B-Instruct": 65.0},
    )
    ids = [r.model.id for r in results]
    assert "bartowski/Qwen3-8B-GGUF" in ids
    # The Q1_0 + self_reported combo must end up below the normal Q4_K_M.
    sc_bonsai = next(
        (r.quality_score for r in results if r.model.id == "prism-ml/Bonsai-8B-gguf"),
        0.0,
    )
    sc_normal = next(
        (r.quality_score for r in results if r.model.id == "bartowski/Qwen3-8B-GGUF"),
        0.0,
    )
    assert sc_normal > sc_bonsai
    # And the absolute Bonsai score must be in the "obviously broken" band.
    assert sc_bonsai < 25


def test_p1_q1_0_returned_when_explicitly_requested_via_quant_filter():
    """Users can still ask for Q1_0 with --quant Q1_0 when they really mean it."""
    model = ModelInfo(
        id="prism-ml/Bonsai-8B-gguf",
        family_id="bonsai-8b",
        name="Bonsai-8B",
        parameter_count=8_000_000_000,
        downloads=48_000,
        likes=0,
        gguf_variants=[_gguf("Q1_0", 4.6)],
    )
    results = rank_models(
        [model], _hw(), top_n=5, benchmark_scores={}, quant_filter="Q1_0"
    )
    assert len(results) == 1
    assert results[0].gguf_variant is not None
    assert results[0].gguf_variant.quant_type == "Q1_0"


def test_p1_q1_q2_quality_penalty_is_severe():
    """Sub-2-bit quants must carry 40-60% quality penalty, not the
    legacy 5% fallback. We assert the constant directly to lock this in."""
    from whichllm.constants import QUANT_QUALITY_PENALTY

    assert QUANT_QUALITY_PENALTY["Q1_0"] >= 0.50
    assert QUANT_QUALITY_PENALTY["Q2_0"] >= 0.40
    assert QUANT_QUALITY_PENALTY["TQ1_0"] >= 0.50
    assert QUANT_QUALITY_PENALTY["IQ1_S"] >= 0.50
    assert QUANT_QUALITY_PENALTY["IQ1_M"] >= 0.45
    assert QUANT_QUALITY_PENALTY["IQ2_M"] >= 0.25


# ---------------------------------------------------------------------- P2-2


def test_p2_excluded_orgs_never_rank():
    """gpt2 / opt-125m / TRL fixtures must be skipped regardless of DL."""
    gpt2 = ModelInfo(
        id="openai-community/gpt2",
        family_id="gpt2",
        name="gpt2",
        parameter_count=124_000_000,
        downloads=16_000_000,
        likes=1000,
    )
    opt = ModelInfo(
        id="facebook/opt-125m",
        family_id="opt-125m",
        name="opt-125m",
        parameter_count=125_000_000,
        downloads=9_000_000,
    )
    tiny = ModelInfo(
        id="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        family_id="tiny-q2",
        name="tiny",
        parameter_count=1_000_000,
        downloads=6_000_000,
    )
    # also a normal model so the result list is non-empty
    real = ModelInfo(
        id="bartowski/gemma-3-12b-it-GGUF",
        family_id="gemma-3-12b",
        name="gemma-3-12b",
        parameter_count=12_200_000_000,
        downloads=70_000,
        likes=10,
        gguf_variants=[_gguf("Q4_K_M", 7.0)],
    )
    results = rank_models([gpt2, opt, tiny, real], _hw(), top_n=10, benchmark_scores={})
    ids = [r.model.id for r in results]
    assert "openai-community/gpt2" not in ids
    assert "facebook/opt-125m" not in ids
    assert "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5" not in ids
    assert "bartowski/gemma-3-12b-it-GGUF" in ids


def test_p2_is_excluded_model_helper():
    assert _is_excluded_model("openai-community/gpt2")
    assert _is_excluded_model("facebook/opt-125m")
    assert _is_excluded_model("EleutherAI/pythia-70m-deduped")
    assert _is_excluded_model("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    assert _is_excluded_model("hmellor/tiny-random-LlamaForCausalLM")
    assert _is_excluded_model("llamafactory/tiny-random-Llama-3")
    assert _is_excluded_model("peft-internal-testing/tiny-dummy-qwen2")
    assert _is_excluded_model("hf-internal-testing/llama-tokenizer")
    assert not _is_excluded_model("Qwen/Qwen3-8B")
    assert not _is_excluded_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")


def test_p2_tiny_in_a_name_is_a_size_not_a_fixture():
    assert not _is_excluded_model("bartowski/Ling-3.0-tiny-GGUF")
    assert not _is_excluded_model("ibm-granite/granite-4.0-h-tiny")
    assert not _is_excluded_model("deepseek-ai/deepseek-vl2-tiny")


# ---------------------------------------------------------------------- P2-1


def test_p2_generation_bonus_newer_wins_over_legacy_in_same_family():
    new = _generation_bonus("Qwen/Qwen3.6-27B")
    mid = _generation_bonus("Qwen/Qwen3-8B")
    old = _generation_bonus("Qwen/Qwen2.5-7B-Instruct")
    ancient = _generation_bonus("Qwen/Qwen-7B")
    assert new > mid > old > ancient


def test_p2_lineage_covers_llama_deepseek_gemma_phi():
    assert _generation_bonus("meta-llama/Llama-4-Scout") > _generation_bonus(
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert _generation_bonus("deepseek-ai/DeepSeek-V4-Pro") > _generation_bonus(
        "deepseek-ai/DeepSeek-V2.5"
    )
    assert _generation_bonus("google/gemma-4-31b-it") > _generation_bonus(
        "google/gemma-2-27b-it"
    )
    assert _generation_bonus("microsoft/phi-4") > _generation_bonus(
        "microsoft/Phi-3-mini-4k-instruct"
    )


def test_p2_lineage_covers_t5_variants_without_gemma_collision():
    assert _generation_bonus("google/t5gemma-4b") > _generation_bonus(
        "google/flan-t5-xl"
    )
    assert _generation_bonus("google/t5-gemma-4b") == _generation_bonus(
        "google/t5gemma-4b"
    )
    assert _generation_bonus("google/t5_gemma-4b") == _generation_bonus(
        "google/t5gemma-4b"
    )
    assert _generation_bonus("openai/gpt5-test") == 0.0


def test_p2_unknown_family_gets_zero_bonus():
    assert _generation_bonus("random-org/random-model-7b") == 0.0


# ---------------------------------------------------------------------- P2-3


def test_p2_derivative_penalty_for_heretic_uncensored():
    assert (
        _derivative_name_penalty("diffusionmodels1254ani/gemma-3-12b-it-heretic-v2") < 0
    )
    assert (
        _derivative_name_penalty(
            "Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF"
        )
        < 0
    )
    assert _derivative_name_penalty("OBLITERATUS/gemma-4-E4B-it-OBLITERATED") < 0
    assert _derivative_name_penalty("bartowski/Qwen3-32B-GGUF") == 0.0


# ---------------------------------------------------------------------- P3-1


def test_p3_self_reported_evidence_does_not_outrank_direct_leaderboard():
    """A self-reported eval (e.g. prism-ml/Bonsai claiming 91) must NOT
    outrank an independent-leaderboard hit on a comparable model."""
    self_reported = ModelInfo(
        id="prism-ml/Self-Reported-8B",
        family_id="self-reported-8b",
        name="Self-Reported",
        parameter_count=8_000_000_000,
        downloads=20_000,
        likes=10,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 91.0},
    )
    direct_hit = ModelInfo(
        id="trusted-org/Real-Bench-8B",
        family_id="real-bench-8b",
        name="Real-Bench",
        parameter_count=8_000_000_000,
        downloads=50_000,
        likes=200,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
    )
    results = rank_models(
        [self_reported, direct_hit],
        _hw(),
        top_n=5,
        benchmark_scores={"trusted-org/Real-Bench-8B": 70.0},
    )
    assert len(results) == 2
    assert results[0].model.id == "trusted-org/Real-Bench-8B"
    assert results[0].benchmark_status == "direct"
    assert results[1].benchmark_status == "self_reported"


def test_p3_self_reported_outranks_only_when_there_is_nothing_else():
    """When no other evidence exists, self_reported should still produce a
    score so the candidate isn't filtered out — just at a lower weight."""
    only_self_reported = ModelInfo(
        id="some-org/Only-Self-Reported-8B",
        family_id="only-sr-8b",
        name="Only-SR",
        parameter_count=8_000_000_000,
        downloads=20_000,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 90.0},
    )
    results = rank_models(
        [only_self_reported],
        _hw(),
        top_n=5,
        benchmark_scores={},
    )
    assert len(results) == 1
    assert results[0].benchmark_status == "self_reported"
    # Score should be positive but well below what a 90 direct-leaderboard
    # would have produced (which would push past 60 with this size class).
    assert 0 < results[0].quality_score < 60


def test_p3_source_weights_ordering():
    """Direct must weight the most, self_reported the least among
    benchmark-producing sources."""
    assert _SOURCE_WEIGHTS["direct"] > _SOURCE_WEIGHTS["base_model"]
    assert _SOURCE_WEIGHTS["base_model"] > _SOURCE_WEIGHTS["variant"]
    assert _SOURCE_WEIGHTS["variant"] > _SOURCE_WEIGHTS["line_interp"]
    assert _SOURCE_WEIGHTS["line_interp"] > _SOURCE_WEIGHTS["self_reported"]
    assert _SOURCE_WEIGHTS["self_reported"] > 0
    assert _SOURCE_WEIGHTS["none"] == 0.0


def test_p3_strict_evidence_filter_excludes_self_reported():
    """`--evidence strict` should keep only direct hits, dropping
    self_reported (the exact bug that surfaced 5 prism-ml/squ11z1
    models as 'direct') as well as inherited evidence."""
    self_reported = ModelInfo(
        id="some-org/Self-Reported-8B",
        family_id="sr-8b",
        name="SR",
        parameter_count=8_000_000_000,
        downloads=20_000,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 95.0},
    )
    direct_hit = ModelInfo(
        id="trusted-org/Real-Bench-8B",
        family_id="real-bench-8b",
        name="Real-Bench",
        parameter_count=8_000_000_000,
        downloads=50_000,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
    )
    results = rank_models(
        [self_reported, direct_hit],
        _hw(),
        top_n=5,
        benchmark_scores={"trusted-org/Real-Bench-8B": 70.0},
        evidence_filter="strict",
    )
    ids = [r.model.id for r in results]
    assert ids == ["trusted-org/Real-Bench-8B"]


# -------------------- Rating-push regressions (Phase 2) --------------------


def test_official_org_safetensors_gets_q4km_synthesis():
    """A Qwen/Meta/Google safetensors-only repo should produce realistic
    Q4_K_M candidates rather than being scored as bf16 (which forces
    partial_offload on consumer GPUs)."""
    model = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3.6-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
        gguf_variants=[],  # safetensors only — synthesis should kick in
    )
    results = rank_models(
        [model],
        _hw(vram_gb=24),
        top_n=1,
        benchmark_scores={"Qwen/Qwen3.6-27B": 84.0},
    )
    assert results
    chosen = results[0]
    assert chosen.fit_type == "full_gpu"
    assert chosen.gguf_variant is not None
    # Synthesis offers a range of K-quants — any of them is acceptable so
    # long as a GGUF candidate is constructed (i.e. the safetensors-only
    # repo is rankable at a realistic quant).
    assert chosen.gguf_variant.quant_type in {
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    }


def test_prequantized_repo_skips_synthesis():
    """An -AWQ / -GPTQ repo must not get synthetic Q4_K_M candidates — those
    repos can't actually serve a GGUF variant."""
    model = ModelInfo(
        id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        family_id="qwen2.5-14b-awq",
        name="Qwen2.5-14B-Instruct-AWQ",
        parameter_count=14_000_000_000,
        downloads=10_000,
    )
    q4_filtered = rank_models(
        [model],
        _hw(vram_gb=24),
        top_n=5,
        benchmark_scores={"Qwen/Qwen2.5-14B-Instruct-AWQ": 70.0},
        quant_filter="Q4_K_M",
    )
    assert q4_filtered == []


def test_newer_generation_beats_older_at_same_size():
    """With the realigned AA/LB normalizers and stronger generation lineage
    bonus, a current-gen 8B model should outrank a frozen-OLLB-favored
    previous-gen 7B."""
    new_gen = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        downloads=10_000_000,
        gguf_variants=[_gguf("Q4_K_M", 4.5)],
    )
    old_gen = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=12_000_000,
        gguf_variants=[_gguf("Q4_K_M", 4.0)],
    )
    results = rank_models(
        [new_gen, old_gen],
        _hw(vram_gb=24),
        top_n=2,
        benchmark_scores={
            "Qwen/Qwen3-8B": 56.0,  # current AA/LB-derived
            "Qwen/Qwen2.5-7B-Instruct": 35.0,  # current source supersedes OLLB
        },
    )
    assert [r.model.id for r in results][0] == "Qwen/Qwen3-8B"


def test_speed_estimator_differs_by_quant_and_backend():
    """Q4_K_M should run faster than F16 on the same hardware, and CUDA
    should be faster than Metal at equal bandwidth."""
    from whichllm.engine.performance import estimate_tok_per_sec
    from whichllm.hardware.types import GPUInfo

    model = ModelInfo(
        id="t/x",
        family_id="t/x",
        name="x",
        parameter_count=8_000_000_000,
        downloads=0,
    )
    q4 = GGUFVariant(
        filename="x.Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=int(8e9 * 0.5625)
    )
    f16 = GGUFVariant(
        filename="x.F16.gguf", quant_type="F16", file_size_bytes=int(8e9 * 2.0)
    )
    cuda = GPUInfo(
        name="t-nv",
        vendor="nvidia",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1000.0,
    )
    metal = GPUInfo(
        name="t-apple",
        vendor="apple",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1000.0,
    )
    q4_cuda = estimate_tok_per_sec(model, q4, cuda, "full_gpu")
    f16_cuda = estimate_tok_per_sec(model, f16, cuda, "full_gpu")
    q4_metal = estimate_tok_per_sec(model, q4, metal, "full_gpu")
    assert q4_cuda > f16_cuda  # Q4 dequant is faster than F16 matmul read
    assert q4_cuda > q4_metal  # CUDA beats Metal at equal bandwidth


def test_vram_kv_cache_scales_with_context():
    """KV-cache contribution should grow linearly with context length so
    longer contexts surface as a real VRAM cost in `plan`."""
    from whichllm.engine.vram import estimate_kv_cache

    model = ModelInfo(
        id="t/x",
        family_id="t/x",
        name="x",
        parameter_count=32_000_000_000,
        downloads=0,
    )
    kv_4k = estimate_kv_cache(model, 4096)
    kv_32k = estimate_kv_cache(model, 32768)
    assert kv_32k > kv_4k * 7  # near-linear in context length
    # and the absolute size at 32K should be in the gigabyte range
    assert kv_32k > 2 * 1024**3
