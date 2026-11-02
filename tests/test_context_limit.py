"""The Context column: the model's limit and the longest context a row holds."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.golden import harness as H
from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.context import (
    context_cell,
    format_context,
    max_context_tokens,
    model_context_limit,
)
from whichllm.models.shape import parse_model_shape

_ARCHETYPES = {a.id: a for a in H.load_spec()[0]}


def _placement(result) -> tuple:
    return result.fit_type, result.placement.kind if result.placement else None


@pytest.mark.parametrize(
    "archetype",
    ["rtx5080-16gb-win", "rtx3060-12gb-win", "m3max-36gb-mac", "cpu-32gb-linux"],
)
def test_the_answer_is_the_edge_of_the_same_placement(archetype):
    a = _ARCHETYPES[archetype]
    hardware = a.hardware()
    for row in H.rank_case(a, "general")[:5]:
        tokens = max_context_tokens(row, hardware, a.context_length)
        at = check_compatibility(row.model, row.gguf_variant, hardware, tokens)
        assert at.can_run and _placement(at) == _placement(row), row.model.id
        limit = model_context_limit(row.model) or 128 * 1024
        if tokens < limit:
            past = check_compatibility(
                row.model, row.gguf_variant, hardware, tokens + 1024
            )
            assert not past.can_run or _placement(past) != _placement(row), row.model.id


def test_a_model_that_fits_whole_shows_its_limit():
    a = _ARCHETYPES["rtx5090-32gb-linux"]
    row = next(
        r
        for r in H.rank_case(a, "general")
        if r.fit_type == "full_gpu" and r.vram_required_bytes < 16 * 1024**3
    )
    small = replace(row, model=replace(row.model, context_length=8192))

    assert max_context_tokens(small, a.hardware(), 4096) == 8192
    small.max_context_tokens = 8192
    assert context_cell(small) == "8k"


def test_cells():
    a = _ARCHETYPES["rtx5080-16gb-win"]
    row = H.rank_case(a, "general")[0]
    known = replace(row, model=replace(row.model, context_length=262144))
    known.max_context_tokens = 18432
    assert context_cell(known) == "256k→18k"
    unknown = replace(row, model=replace(row.model, context_length=None, shape=None))
    unknown.max_context_tokens = 131072
    assert context_cell(unknown) == "128k+"
    assert format_context(1048576) == "1M"


def test_the_limit_comes_from_config_when_the_hub_has_none():
    shape = parse_model_shape(
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "hidden_size": 1024,
                "max_position_embeddings": 262144,
            },
        }
    )

    assert shape is not None and shape.max_context == 262144
    model = H.catalog_models()[0]
    assert (
        model_context_limit(replace(model, context_length=None, shape=shape)) == 262144
    )
