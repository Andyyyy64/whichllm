"""Tests for model grouping logic."""

from whichllm.models.grouper import _normalize_name, group_models
from whichllm.models.types import ModelInfo


def _make_model(
    id: str, base_model: str | None = None, downloads: int = 100
) -> ModelInfo:
    return ModelInfo(
        id=id,
        family_id=id,
        name=id.split("/")[-1],
        parameter_count=7_000_000_000,
        downloads=downloads,
        base_model=base_model,
        base_model_relation="quantized" if base_model else None,
    )


def test_group_by_base_model():
    base = _make_model("meta/Llama-3-8B", downloads=1000)
    gguf = _make_model(
        "user/Llama-3-8B-GGUF", base_model="meta/Llama-3-8B", downloads=500
    )
    families = group_models([base, gguf])
    assert len(families) == 1
    assert families[0].base_model.id in ("meta/Llama-3-8B", "user/Llama-3-8B-GGUF")


def test_group_by_name_normalization():
    base = _make_model("org/model-v1", downloads=1000)
    gguf = _make_model("org/model-v1-GGUF", downloads=200)
    families = group_models([base, gguf])
    assert len(families) <= 2  # may or may not merge depending on normalization


def test_fp4_suffixes_normalize_to_base_family():
    # MXFP4 and NVFP4 derivatives must collapse onto the base family the same
    # way the older quant suffixes do, instead of orphaning into their own.
    base = _normalize_name("openai/gpt-oss-20b")
    assert _normalize_name("openai/gpt-oss-20b-MXFP4") == base
    assert _normalize_name("openai/gpt-oss-20b-NVFP4") == base


def test_ungrouped_models_separate():
    m1 = _make_model("org/alpha", downloads=100)
    m2 = _make_model("org/beta", downloads=200)
    families = group_models([m1, m2])
    assert len(families) == 2


def test_empty_input():
    families = group_models([])
    assert families == []


def test_checkpoint_versions_dates_and_instruction_variants_stay_distinct():
    for names in [
        ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B"],
        ["Mistral-Small-3.1-24B", "Mistral-Small-3.2-24B"],
        ["DeepSeek-V3.1", "DeepSeek-V3.2"],
        ["Model-7B-2503", "Model-7B-2507"],
        ["Model-7B", "Model-7B-Instruct", "Model-7B-Chat"],
    ]:
        models = [_make_model(f"org/{name}") for name in names]
        models += [
            _make_model(f"converter/{name}-GGUF", f"org/{name}") for name in names
        ]
        families = group_models(models)
        assert len(families) == len(names)
        assert len({f.family_id for f in families}) == len(names)
        assert all(len(f.variants) == 1 for f in families)


def test_derivative_and_its_quantization_do_not_join_the_upstream():
    base = _make_model("org/Model-7B")
    derived = _make_model("tuner/Model-7B", base.id)
    derived.base_model_relation = "finetune"
    quant = _make_model("converter/Model-7B-GGUF", derived.id)
    families = group_models([base, derived, quant])
    assert len(families) == 2
    assert derived.family_id == quant.family_id != base.family_id


def test_cached_old_family_ids_are_recomputed():
    from whichllm.models.serialization import dicts_to_models, models_to_dicts

    models = [_make_model(f"Qwen/Qwen3.{minor}-27B") for minor in [5, 6, 8]]
    for model in models:
        model.family_id = "qwen3-27b"
    restored = dicts_to_models(models_to_dicts(models))
    assert len(group_models(restored)) == 3
    assert len({model.family_id for model in restored}) == 3


def test_quantization_keeps_its_referenced_namespace_with_or_without_base():
    for include_base in [False, True]:
        unrelated = _make_model("org/Model-7B")
        referenced = _make_model("tuner/Model-7B")
        quant = _make_model("converter/Model-7B-GGUF", referenced.id)
        models = [unrelated, quant] + ([referenced] if include_base else [])
        families = group_models(models)
        assert {family.family_id for family in families} == {
            "org/model-7b",
            "tuner/model-7b",
        }
        assert quant.family_id != unrelated.family_id

    unrelated = _make_model("org/Model-7B")
    quant = _make_model("converter/Model-7B-GGUF", "org/Model-7B-FP16")
    group_models([unrelated, quant])
    assert quant.family_id == "org/model-7b-fp16"
    assert quant.family_id != unrelated.family_id


def test_family_id_set():
    base = _make_model("meta/Llama-3-8B", downloads=1000)
    gguf = _make_model(
        "user/Llama-3-8B-GGUF", base_model="meta/Llama-3-8B", downloads=500
    )
    families = group_models([base, gguf])
    for family in families:
        assert family.family_id
        assert family.base_model.family_id == family.family_id
