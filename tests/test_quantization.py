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


def test_extract_quant_type_canonicalizes_full_precision_aliases():
    # llama.cpp publishes full-precision GGUFs as *-FP16/*-FP32; the byte and
    # penalty tables key these as F16/F32, so the extractor must canonicalize.
    from whichllm.models.fetcher import _extract_quant_type

    assert _extract_quant_type("Meta-Llama-3-8B-FP16.gguf") == "F16"
    assert _extract_quant_type("model.FP32.gguf") == "F32"
    # Canonical spellings still pass through unchanged.
    assert _extract_quant_type("model-F16.gguf") == "F16"
    assert _extract_quant_type("model.BF16.gguf") == "BF16"


def test_extract_quant_type_recognizes_ternary_gguf():
    # BitNet-class ternary GGUFs (TQ1_0/TQ2_0) are fully priced in the tables
    # but were previously extracted as "unknown" and dropped at fetch.
    from whichllm.models.fetcher import _extract_quant_type

    assert _extract_quant_type("BitNet-b1.58-2B-4T-TQ1_0.gguf") == "TQ1_0"
    assert _extract_quant_type("model.TQ2_0.gguf") == "TQ2_0"


def test_estimate_gguf_size_does_not_undersize_fp16():
    # An FP16 GGUF must size at full precision (2.0 bytes/weight), not collapse
    # to the Q4_K_M 0.5625 default that an unrecognized token falls back to.
    from whichllm.models.fetcher import _estimate_gguf_size, _extract_quant_type

    params = 7_000_000_000
    quant = _extract_quant_type("model-FP16.gguf")
    size = _estimate_gguf_size(params, quant)
    assert size == params * 2  # 14 GB, not the ~3.94 GB default
    assert size == _estimate_gguf_size(params, "F16")


def test_extract_quant_type_keys_resolve_in_byte_table():
    # Drift guard: every quant the extractor surfaces from a real GGUF filename
    # must resolve in QUANT_BYTES_PER_WEIGHT, otherwise it is silently mis-sized
    # by the default or dropped at fetch. Keeps the extractor and tables aligned.
    from whichllm.data.quantization import QUANT_BYTES_PER_WEIGHT
    from whichllm.models.fetcher import _extract_quant_type

    filenames = [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
        "model-Q6_K.gguf",
        "model-IQ4_NL.gguf",
        "model-IQ3_XXS.gguf",
        "model-TQ1_0.gguf",
        "model-TQ2_0.gguf",
        "model-F16.gguf",
        "model-FP16.gguf",
        "model-BF16.gguf",
        "model-F32.gguf",
        "model-FP32.gguf",
        "model-MXFP4.gguf",
        "model-NVFP4.gguf",
    ]
    for fname in filenames:
        quant = _extract_quant_type(fname)
        assert quant != "unknown", f"{fname} not recognized by extractor"
        assert quant in QUANT_BYTES_PER_WEIGHT, (
            f"{fname} -> {quant!r} missing from QUANT_BYTES_PER_WEIGHT"
        )
