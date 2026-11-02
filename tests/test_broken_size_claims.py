"""Repos whose size claims cannot be the model they describe."""

from __future__ import annotations

import pytest

from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _understates_its_base
from whichllm.engine.ranking_variants import _iter_candidate_variants
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.benchmark_index import _extract_params_b_from_id
from whichllm.models.gguf import _extract_gguf_variants, holds_the_whole_model
from whichllm.models.parameters import _extract_size_hint_from_id
from whichllm.models.types import GGUFVariant, ModelInfo

_GB = 10**9


def _model(
    model_id: str,
    params_b: float,
    base: str | None = None,
    variants: list[GGUFVariant] | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=int(params_b * 1e9),
        downloads=1000,
        base_model=base,
        gguf_variants=variants or [],
    )


@pytest.mark.parametrize(
    ("claimed_b", "expected"),
    [
        # kaushikvira/DeepSeek-V4.1-Flash-1017-GSQ-agentic-strata
        (14.7, True),
        # AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GGUF
        (0.115, True),
        # staccs/DeepSeek-V4.1-Flash-1017-lecore-vq10s: a 2-bit checkpoint
        (102.0, False),
        (763.0, False),
    ],
)
def test_a_derivative_cannot_claim_a_tiny_share_of_its_base(claimed_b, expected):
    base = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    model = _model("org/Derivative-GGUF", claimed_b, base)

    assert _understates_its_base(model, {base.casefold(): int(763e9)}) is expected


def test_a_base_outside_the_catalog_says_nothing():
    model = _model("org/Derivative-GGUF", 0.1, "other/Unknown-Model")

    assert _understates_its_base(model, {}) is False


def test_ranking_drops_the_fragment_and_keeps_the_real_derivative():
    variant = GGUFVariant("model-Q4_K_M.gguf", "Q4_K_M", 5 * _GB)
    base = _model("org/Base-100B", 100)
    fragment = _model("someone/Base-Fragment-GGUF", 1, base.id, [variant])
    real = _model("someone/Base-Small-GGUF", 30, base.id, [variant])
    hardware = HardwareInfo(
        gpus=[
            GPUInfo(
                name="RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1010.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=16,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )

    results = rank_models([base, fragment, real], hardware, top_n=10)

    ids = [r.model.id for r in results]
    assert real.id in ids
    assert fragment.id not in ids


@pytest.mark.parametrize(
    ("size_gb", "quant", "params_b", "expected"),
    [
        # Humanlike-Chat-2.0-LoRA-r320-F16.gguf
        (4.7, "F16", 26.9, False),
        # mtp-Qwen3.8-27B-Uncensored-Cyber-BF16.gguf
        (5.9, "BF16", 26.9, False),
        # Below one bit per weight, whatever the label.
        (1.7, "IQ1_S", 27.0, False),
        # gpt-oss-120b-F16.gguf: mixed precision, small for its label.
        (65.4, "F16", 116.83, True),
        # Qwythos-27B-MTP-Q4_K_M.gguf: a whole model with "MTP" in its name.
        (17.6, "Q4_K_M", 27.0, True),
    ],
)
def test_a_file_must_hold_the_model_its_label_describes(
    size_gb, quant, params_b, expected
):
    assert (
        holds_the_whole_model(int(size_gb * _GB), quant, int(params_b * 1e9))
        is expected
    )


def test_the_parser_skips_a_lora_published_next_to_the_weights():
    data = {
        "siblings": [
            {"rfilename": "Model-27B-LoRA-r320-F16.gguf", "size": int(4.7 * _GB)},
            {"rfilename": "Model-27B-Q4_K_M.gguf", "size": int(16.8 * _GB)},
        ]
    }

    variants = _extract_gguf_variants(data, int(26.9e9))

    assert [v.quant_type for v in variants] == ["Q4_K_M"]


def test_ranking_skips_such_a_file_from_an_older_index():
    lora = GGUFVariant("Model-LoRA-F16.gguf", "F16", int(4.7 * _GB))
    weights = GGUFVariant("Model-Q4_K_M.gguf", "Q4_K_M", int(16.8 * _GB))

    with_weights = _model("org/Model-GGUF", 26.9, variants=[lora, weights])
    only_lora = _model("org/Lora-GGUF", 26.9, variants=[lora])

    assert _iter_candidate_variants(with_weights) == [weights]
    assert _iter_candidate_variants(only_lora) == []


@pytest.mark.parametrize(
    ("model_id", "params_b"),
    [
        ("LiquidAI/LFM3-1.7B-MLX-8bit", 1.7),
        ("Genrius-AI/Qwen3-0.6B-Pingala8-2bit", 0.6),
        ("BlackwellBoy/Model-EXL3-2.9bpw", None),
        ("Qwen/Qwen3.6-35B-A3B", 35.0),
        ("org/Llama-3.1-8B-Instruct", 8.0),
    ],
)
def test_bit_widths_are_not_sizes(model_id, params_b):
    assert _extract_params_b_from_id(model_id) == params_b
    hint = _extract_size_hint_from_id(model_id)
    assert hint == (int(params_b * 1e9) if params_b else None)
