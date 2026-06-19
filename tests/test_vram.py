"""Tests for VRAM estimation."""

from whichllm.engine.vram import estimate_kv_cache, estimate_vram
from whichllm.models.types import GGUFVariant, ModelInfo


def _make_model(params: int, **kwargs) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=params,
        **kwargs,
    )


def test_estimate_vram_gguf_variant():
    model = _make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)
    # Should be: 4GB weights + KV cache + activation + framework overhead
    assert vram > 4_000_000_000
    assert vram < 7_000_000_000  # should be well under FP16 size


def test_estimate_vram_fp16_fallback():
    model = _make_model(7_000_000_000)
    vram = estimate_vram(model, None, context_length=4096)
    # FP16: 7B * 2 = 14GB + overhead
    assert vram > 14_000_000_000
    assert vram < 20_000_000_000


def test_estimate_vram_increases_with_context():
    model = _make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram_4k = estimate_vram(model, variant, context_length=4096)
    vram_32k = estimate_vram(model, variant, context_length=32768)
    assert vram_32k > vram_4k


def test_estimate_kv_cache_scales_with_params():
    small = _make_model(1_000_000_000)
    large = _make_model(70_000_000_000)
    kv_small = estimate_kv_cache(small, 4096)
    kv_large = estimate_kv_cache(large, 4096)
    assert kv_large > kv_small


def test_estimate_vram_small_model():
    model = _make_model(500_000_000)  # 0.5B
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=300_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)
    # Should be reasonable for a tiny model
    assert vram > 300_000_000
    assert vram < 3_000_000_000


def test_kv_cache_unchanged_when_no_sliding_window():
    """Models without an honored sliding window keep the full-context KV figure.

    This is the conservative-default guarantee: the SWA change must be a no-op
    for every model that does not advertise an honored window. Expected values
    are pinned literals (the current 3.5 MB/B/Kctx coefficient) so the test
    fails if either the formula or the coefficient drifts.
    """
    dense = _make_model(7_000_000_000)
    expected = {
        4096: 102_760_448,
        32768: 822_083_584,
        131072: 3_288_334_336,
    }
    for ctx, want in expected.items():
        assert estimate_kv_cache(dense, ctx) == want


def test_kv_cache_mistral_window_in_config_is_ignored():
    """A declared window is NOT honored unless the architecture is allowlisted.

    Mistral-7B-v0.1 ships sliding_window=4096 in its config but mainline
    runtimes ignore it, so whichllm must stay at full-context KV. We model this
    by leaving sliding_window=None on the model; the estimate must equal dense.
    """
    mistral = _make_model(7_000_000_000, architecture="mistral")
    dense = _make_model(7_000_000_000)
    for ctx in (4096, 131072):
        assert estimate_kv_cache(mistral, ctx) == estimate_kv_cache(dense, ctx)


def test_kv_cache_pure_swa_plateaus_beyond_window():
    """Pure sliding-window models (global_ratio=0) plateau at the window size."""
    swa = _make_model(
        7_000_000_000, sliding_window=4096, sliding_window_global_ratio=0.0
    )
    at_window = estimate_kv_cache(swa, 4096)
    far_beyond = estimate_kv_cache(swa, 131072)
    # KV is flat once context exceeds the window.
    assert far_beyond == at_window
    # And it is far below the dense estimate at the same long context.
    dense = estimate_kv_cache(_make_model(7_000_000_000), 131072)
    assert far_beyond < dense / 10


def test_kv_cache_hybrid_grows_slower_than_dense():
    """Hybrid SWA models grow with context but much slower than dense."""
    # Gemma-3-like: 1/6 global layers, 1024-token window.
    hybrid = _make_model(
        27_000_000_000,
        sliding_window=1024,
        sliding_window_global_ratio=1.0 / 6.0,
    )
    dense = _make_model(27_000_000_000)
    short = estimate_kv_cache(hybrid, 4096)
    long_hybrid = estimate_kv_cache(hybrid, 131072)
    long_dense = estimate_kv_cache(dense, 131072)
    # Still grows with context (global layers keep scaling)...
    assert long_hybrid > short
    # ...but well under the dense estimate (roughly the global ratio).
    assert long_hybrid < long_dense
    assert long_hybrid < long_dense * 0.30


def test_kv_cache_never_exceeds_dense_estimate():
    """The SWA reduction can only ever lower the estimate, never raise it."""
    for ratio in (0.0, 1.0 / 6.0, 0.25, 0.5, 1.0):
        swa = _make_model(
            13_000_000_000,
            sliding_window=2048,
            sliding_window_global_ratio=ratio,
        )
        dense = _make_model(13_000_000_000)
        for ctx in (1024, 4096, 65536):
            assert estimate_kv_cache(swa, ctx) <= estimate_kv_cache(dense, ctx)


def test_kv_cache_below_window_matches_dense():
    """When context fits inside the window, SWA and dense agree."""
    swa = _make_model(
        7_000_000_000, sliding_window=8192, sliding_window_global_ratio=0.0
    )
    dense = _make_model(7_000_000_000)
    assert estimate_kv_cache(swa, 4096) == estimate_kv_cache(dense, 4096)
