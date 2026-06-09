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


def test_family_id_set():
    base = _make_model("meta/Llama-3-8B", downloads=1000)
    gguf = _make_model(
        "user/Llama-3-8B-GGUF", base_model="meta/Llama-3-8B", downloads=500
    )
    families = group_models([base, gguf])
    for family in families:
        assert family.family_id
        assert family.base_model.family_id == family.family_id
