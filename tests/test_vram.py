"""Tests for VRAM estimation."""

from local_llm_checker.engine.vram import estimate_kv_cache, estimate_vram
from local_llm_checker.models.types import GGUFVariant, ModelInfo


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
    variant = GGUFVariant(filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000)
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
    variant = GGUFVariant(filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000)
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
    variant = GGUFVariant(filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=300_000_000)
    vram = estimate_vram(model, variant, context_length=4096)
    # Should be reasonable for a tiny model
    assert vram > 300_000_000
    assert vram < 3_000_000_000
