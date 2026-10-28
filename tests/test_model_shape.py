"""Model shape parsing from real config.json files and exact KV cache sizes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whichllm.engine.vram import estimate_kv_cache
from whichllm.models.fetcher import dicts_to_models, models_to_dicts
from whichllm.models.shape import ModelShape, parse_model_shape
from whichllm.models.types import ModelInfo

CONFIGS = json.loads(
    (Path(__file__).parent / "fixtures" / "model_configs.json").read_text(
        encoding="utf-8"
    )
)
GiB = 1024**3


def _shape(name: str) -> ModelShape:
    shape = parse_model_shape(CONFIGS[name])
    assert shape is not None, name
    return shape


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_fixture_parses(name):
    shape = _shape(name)
    assert shape.num_layers > 0
    assert shape.full_kv_bytes_per_token > 0


@pytest.mark.parametrize(
    "name,ctx,expected_gib",
    [
        # 36 layers x 8 KV heads x 128 dims x K+V x FP16.
        ("Qwen/Qwen3-8B", 32768, 4.5),
        ("meta-llama/Llama-3.1-8B-Instruct", 131072, 16.0),
        # Hybrid Gated DeltaNet: only every 4th of 64 layers keeps K/V.
        ("Qwen/Qwen3.6-27B", 131072, 8.0 + 0.146),
        # MLA: 61 layers x (512 latent + 64 RoPE) x FP16.
        ("deepseek-ai/DeepSeek-V3.2", 131072, 8.578),
        # Gemma 3: 10 global layers at full context, 52 local at 1024.
        ("google/gemma-3-27b-it", 131072, 10.406),
        # gpt-oss: 12 full layers; the 12 sliding layers stop at 128 tokens.
        ("openai/gpt-oss-20b", 131072, 3.0),
    ],
)
def test_kv_cache_matches_hand_calculation(name, ctx, expected_gib):
    assert _shape(name).kv_cache_bytes(ctx) / GiB == pytest.approx(
        expected_gib, abs=0.01
    )


def test_sliding_layers_cap_at_window_only_when_honoured():
    gemma = _shape("google/gemma-3-27b-it")
    assert gemma.honors_sliding_window
    assert gemma.sliding_window == 1024
    growth = gemma.kv_cache_bytes(8192) - gemma.kv_cache_bytes(4096)
    assert growth == gemma.full_kv_bytes_per_token * 4096

    # Mistral v0.1 declares a window that llama.cpp ignores: all full layers.
    mistral = _shape("mistralai/Mistral-7B-v0.1")
    assert mistral.sliding_kv_bytes_per_token == 0
    assert mistral.full_kv_bytes_per_token == 32 * 2 * 8 * 128 * 2


def test_gemma4_global_layers_use_global_heads():
    shape = _shape("google/gemma-4-31B-it")
    assert shape.honors_sliding_window
    # 10 global layers: 4 KV heads x 512 dims; 50 local: 16 x 256.
    assert shape.full_kv_bytes_per_token == 10 * 2 * 4 * 512 * 2
    assert shape.sliding_kv_bytes_per_token == 50 * 2 * 16 * 256 * 2


def test_recurrent_layers_add_fixed_state():
    for name in (
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3.6-35B-A3B",
        "LiquidAI/LFM2-8B-A1B",
        "ibm-granite/granite-4.0-h-small",
    ):
        shape = _shape(name)
        assert shape.recurrent_state_bytes > 0, name
        assert shape.kv_cache_bytes(0) == shape.recurrent_state_bytes


def test_moe_fields():
    qwen = _shape("Qwen/Qwen3-30B-A3B")
    assert (qwen.num_experts, qwen.experts_per_token) == (128, 8)
    assert qwen.moe_intermediate_size == 768

    deepseek = _shape("deepseek-ai/DeepSeek-V3.2")
    assert deepseek.num_experts == 256
    assert deepseek.first_dense_layers == 3
    assert deepseek.shared_expert_intermediate_size == 2048

    dense = _shape("Qwen/Qwen3-8B")
    assert dense.num_experts is None
    assert dense.experts_per_token is None


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "llama"},
        {"model_type": "llama", "num_hidden_layers": 0},
        # Unknown layer kind.
        {
            "model_type": "x",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 64,
            "layer_types": ["full_attention", "rwkv"],
        },
        # layer_types length mismatch.
        {
            "model_type": "x",
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "hidden_size": 64,
            "layer_types": ["full_attention"],
        },
        # Linear attention without its dimensions.
        {
            "model_type": "x",
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "hidden_size": 64,
            "full_attention_interval": 4,
        },
        # More KV heads than query heads.
        {
            "model_type": "x",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 8,
            "hidden_size": 64,
        },
    ],
)
def test_unparseable_layouts_return_none(config):
    assert parse_model_shape(config) is None


def _model(**kwargs) -> ModelInfo:
    return ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_190_735_360,
        **kwargs,
    )


def test_estimate_kv_cache_prefers_shape_over_heuristic():
    heuristic = estimate_kv_cache(_model(), 32768)
    exact = estimate_kv_cache(_model(shape=_shape("Qwen/Qwen3-8B")), 32768)
    assert exact == 36 * 2 * 8 * 128 * 2 * 32768
    assert exact > 4 * heuristic


def test_shape_survives_cache_roundtrip():
    model = _model(shape=_shape("google/gemma-4-31B-it"))
    restored = dicts_to_models(models_to_dicts([model]))[0]
    assert restored.shape == model.shape


def test_malformed_cached_shape_is_dropped():
    data = models_to_dicts([_model()])
    data[0]["shape"] = {"model_type": "qwen3", "num_layers": "36"}
    assert dicts_to_models(data)[0].shape is None
