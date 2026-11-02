"""MoE fields filled from config.json shapes when the Hub parser missed them."""

from __future__ import annotations

from whichllm.models.moe_fields import fill_moe_from_shape
from whichllm.models.serialization import dicts_to_models, models_to_dicts
from whichllm.models.shape import ModelShape
from whichllm.models.types import ModelInfo


def _shape(**kw) -> ModelShape:
    base = dict(
        model_type="qwen4_exp_text",
        num_layers=48,
        full_kv_bytes_per_token=24576,
        hidden_size=2560,
        num_experts=512,
        experts_per_token=10,
        moe_intermediate_size=640,
    )
    base.update(kw)
    return ModelShape(**base)


def _model(**kw) -> ModelInfo:
    base = dict(
        id="Qwen/Qwen3.8-Flash-Next",
        family_id="qwen/qwen3.8-flash-next",
        name="Qwen3.8-Flash-Next",
        parameter_count=180_000_000_000,
    )
    base.update(kw)
    return ModelInfo(**base)


def test_nested_expert_config_marks_the_model_moe():
    model = _model(shape=_shape())
    assert fill_moe_from_shape(model)
    assert model.is_moe
    routed = 48 * 512 * 3 * 2560 * 640
    expected = 180_000_000_000 - routed + routed * 10 / 512
    assert abs(model.parameter_count_active - expected) < 1e6


def test_known_moe_fields_are_kept():
    model = _model(is_moe=True, parameter_count_active=3_000_000_000, shape=_shape())
    assert not fill_moe_from_shape(model)
    assert model.parameter_count_active == 3_000_000_000


def test_dense_shapes_are_left_alone():
    model = _model(shape=_shape(num_experts=None, experts_per_token=None))
    assert not fill_moe_from_shape(model)
    assert not model.is_moe


def test_index_models_get_moe_fields_on_load():
    stored = models_to_dicts([_model(shape=_shape())])
    stored[0]["is_moe"] = False
    stored[0]["parameter_count_active"] = None
    (loaded,) = dicts_to_models(stored)
    assert loaded.is_moe
    assert loaded.parameter_count_active < loaded.parameter_count
