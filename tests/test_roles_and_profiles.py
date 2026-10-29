"""Base-checkpoint exclusion and task profile eligibility."""

from __future__ import annotations

from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _matches_profile
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.roles import find_base_checkpoints
from whichllm.models.types import GGUFVariant, ModelInfo


def _model(model_id: str, *, tags=(), relation=None, params=8_000_000_000, **kw):
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=params,
        tags=tuple(tags),
        base_model_relation=relation,
        **kw,
    )


def _finetune_of(parent: str) -> tuple[str, ...]:
    return (f"base_model:{parent}", f"base_model:finetune:{parent}")


def test_official_instruct_marks_its_parent_as_base():
    models = [
        _model("Qwen/Qwen2.5-7B"),
        _model("Qwen/Qwen2.5-7B-Instruct", tags=_finetune_of("Qwen/Qwen2.5-7B")),
        _model("google/gemma-3-27b-pt"),
        _model("google/gemma-3-27b-it", tags=_finetune_of("google/gemma-3-27b-pt")),
    ]
    assert find_base_checkpoints(models) == {
        "Qwen/Qwen2.5-7B",
        "google/gemma-3-27b-pt",
    }


def test_named_instruct_sibling_marks_base_but_dated_refresh_does_not():
    models = [
        _model("meta-llama/Llama-3.2-1B"),
        _model("meta-llama/Llama-3.2-1B-Instruct"),
        # Qwen3-4B is the post-trained hybrid model, fine-tuned from -Base.
        _model("Qwen/Qwen3-4B", tags=_finetune_of("Qwen/Qwen3-4B-Base")),
        _model("Qwen/Qwen3-4B-Instruct-2507"),
        _model("Qwen/Qwen3-4B-Base"),
    ]
    assert find_base_checkpoints(models) == {
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen3-4B-Base",
    }


def test_post_trained_model_is_not_a_base_even_if_an_instruct_names_it():
    models = [
        _model("Qwen/Qwen3-8B", tags=_finetune_of("Qwen/Qwen3-8B-Base")),
        _model("Qwen/Qwen3-8B-Instruct", tags=_finetune_of("Qwen/Qwen3-8B")),
    ]
    assert find_base_checkpoints(models) == set()


def test_quantized_uploads_follow_the_base_by_name_not_by_tag():
    models = [
        _model("Qwen/Qwen2.5-Coder-7B"),
        _model(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            tags=_finetune_of("Qwen/Qwen2.5-Coder-7B"),
        ),
        # Base GGUF whose uploader tagged the wrong parent.
        _model(
            "QuantFactory/Qwen2.5-Coder-7B-GGUF",
            tags=("base_model:quantized:Qwen/Qwen2.5-7B",),
            relation="quantized",
        ),
        # Instruct quant tagged with a -Base parent: the name decides.
        _model(
            "casperhansen/llama-3.3-70b-instruct-awq",
            tags=("base_model:quantized:meta-llama/Llama-3.3-70B-Base",),
            relation="quantized",
        ),
        _model(
            "deepseek-ai/DeepSeek-V3.1",
            tags=("base_model:quantized:deepseek-ai/DeepSeek-V3.1-Base",),
            relation="quantized",
        ),
    ]
    assert find_base_checkpoints(models) == {
        "Qwen/Qwen2.5-Coder-7B",
        "QuantFactory/Qwen2.5-Coder-7B-GGUF",
    }


def test_coding_and_math_admit_general_models_but_not_other_specialists():
    general = _model("openai/gpt-oss-20b")
    coder = _model("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    vl = _model("Qwen/Qwen3-VL-8B-Instruct")
    math = _model("Qwen/Qwen2.5-Math-7B-Instruct")

    assert [_matches_profile(m, "coding") for m in (general, coder, vl, math)] == [
        True,
        True,
        False,
        False,
    ]
    assert [_matches_profile(m, "math") for m in (general, coder, vl, math)] == [
        True,
        False,
        False,
        True,
    ]


def test_vision_admits_natively_multimodal_models_by_hub_tag():
    native = _model("Qwen/Qwen3.6-27B", tags=("image-text-to-text",))
    named = _model("Qwen/Qwen3-VL-8B-Instruct")
    text_only = _model("openai/gpt-oss-20b", tags=("text-generation",))

    assert _matches_profile(native, "vision")
    assert _matches_profile(named, "vision")
    assert not _matches_profile(text_only, "vision")


def _hw() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=1000.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=1024**4,
        os="linux",
    )


def _ranked(model_id: str, **kw) -> ModelInfo:
    model = _model(model_id, downloads=1000, **kw)
    model.gguf_variants = [
        GGUFVariant(
            filename=f"{model.name}-Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=5 * 1024**3,
        )
    ]
    return model


def test_ranking_never_returns_base_checkpoints():
    base = _ranked("Qwen/Qwen2.5-7B")
    instruct = _ranked("Qwen/Qwen2.5-7B-Instruct", tags=_finetune_of(base.id))

    results = rank_models(
        [base, instruct],
        _hw(),
        benchmark_scores={base.id: 80.0, instruct.id: 60.0},
        task_profile="any",
    )

    assert [r.model.id for r in results] == ["Qwen/Qwen2.5-7B-Instruct"]


def test_coding_profile_prefers_code_tuned_variant_on_equal_evidence():
    general = _ranked("Qwen/Qwen3-30B-A3B")
    coder = _ranked("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    scores = {general.id: 70.0, coder.id: 70.0}

    coding = rank_models(
        [general, coder], _hw(), benchmark_scores=scores, task_profile="coding"
    )
    anything = rank_models(
        [general, coder], _hw(), benchmark_scores=scores, task_profile="any"
    )

    assert coding[0].model.id == coder.id
    assert coding[0].quality_score - coding[1].quality_score == 2.0
    assert anything[0].quality_score == anything[1].quality_score
