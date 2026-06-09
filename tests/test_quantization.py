"""Tests for non-GGUF quantization inference."""

from whichllm.engine.quantization import (
    effective_quant_type,
    estimate_weight_bytes,
    infer_non_gguf_quant_type,
)
from whichllm.engine.vram import estimate_vram
from whichllm.models.types import ModelInfo


def _make_model(model_id: str, params: int = 14_000_000_000) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name=model_id.split("/")[-1],
        parameter_count=params,
    )


def test_infer_non_gguf_awq():
    model = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    assert infer_non_gguf_quant_type(model.id) == "AWQ"
    assert effective_quant_type(model, None) == "AWQ"


def test_estimate_weight_bytes_for_awq():
    model = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ", params=10_000_000_000)
    assert estimate_weight_bytes(model, None) == 5_000_000_000


def test_awq_vram_is_lower_than_fp16_fallback():
    awq = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    fp16 = _make_model("Qwen/Qwen2.5-14B-Instruct")
    assert estimate_vram(awq, None, context_length=4096) < estimate_vram(
        fp16, None, context_length=4096
    )


def test_infer_mxfp4():
    model = _make_model("openai/gpt-oss-20b-MXFP4")
    assert infer_non_gguf_quant_type(model.id) == "MXFP4"
    assert effective_quant_type(model, None) == "MXFP4"


def test_infer_nvfp4():
    model = _make_model("nvidia/Llama-3.1-8B-Instruct-NVFP4")
    assert infer_non_gguf_quant_type(model.id) == "NVFP4"
    assert effective_quant_type(model, None) == "NVFP4"


def test_fp4_patterns_do_not_false_match_plain_ids():
    # A bare id with no fp4 token must not be mislabeled as a microscaling float.
    plain = _make_model("meta-llama/Llama-3.1-8B-Instruct")
    assert infer_non_gguf_quant_type(plain.id) == "FP16"


def test_estimate_weight_bytes_for_fp4_formats():
    mxfp4 = _make_model("openai/gpt-oss-20b-MXFP4", params=20_000_000_000)
    nvfp4 = _make_model("nvidia/model-NVFP4", params=20_000_000_000)
    assert estimate_weight_bytes(mxfp4, None) == int(20_000_000_000 * 0.53125)
    assert estimate_weight_bytes(nvfp4, None) == int(20_000_000_000 * 0.5625)


def test_fp4_vram_is_lower_than_fp16_fallback():
    mxfp4 = _make_model("openai/gpt-oss-20b-MXFP4")
    fp16 = _make_model("openai/gpt-oss-20b")
    assert estimate_vram(mxfp4, None, context_length=4096) < estimate_vram(
        fp16, None, context_length=4096
    )


def test_extract_quant_type_parses_fp4_gguf_filenames():
    from whichllm.models.fetcher import _extract_quant_type

    assert _extract_quant_type("gpt-oss-20b-MXFP4.gguf") == "MXFP4"
    assert _extract_quant_type("model.NVFP4.gguf") == "NVFP4"
