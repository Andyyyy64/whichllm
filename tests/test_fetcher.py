"""Tests for model metadata normalization in fetcher."""

from whichllm.models.fetcher import (
    _extract_published_at,
    _normalize_param_count,
    dicts_to_models,
    models_to_dicts,
)
from whichllm.models.types import ModelInfo


def test_normalize_param_count_for_quantized_repo_uses_size_hint():
    corrected = _normalize_param_count(
        extracted=5_233_828_308,
        model_id="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        base_model="google/gemma-3-27b-it",
    )
    assert corrected == 27_000_000_000


def test_normalize_param_count_keeps_reasonable_value():
    kept = _normalize_param_count(
        extracted=11_765_788_416,
        model_id="MaziyarPanahi/gemma-3-12b-it-GGUF",
        base_model="google/gemma-3-12b-it",
    )
    assert kept == 11_765_788_416


def test_normalize_param_count_with_no_hint_keeps_original():
    kept = _normalize_param_count(
        extracted=3_820_000_000,
        model_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        base_model=None,
    )
    assert kept == 3_820_000_000


def test_dicts_to_models_normalizes_cached_parameter_count():
    models = dicts_to_models(
        [
            {
                "id": "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
                "family_id": "gemma-3-27b",
                "name": "gemma-3-27b-it-GPTQ-4b-128g",
                "parameter_count": 5_233_828_308,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
                "base_model": "google/gemma-3-27b-it",
            }
        ]
    )
    assert len(models) == 1
    assert models[0].parameter_count == 27_000_000_000


def test_models_cache_roundtrip_keeps_published_at():
    models = [
        ModelInfo(
            id="Qwen/Qwen3-8B-AWQ",
            family_id="qwen3-8b",
            name="Qwen3-8B-AWQ",
            parameter_count=8_000_000_000,
            published_at="2025-09-17T12:34:56.000Z",
            downloads=123_456,
            likes=789,
        )
    ]
    cached = models_to_dicts(models)
    restored = dicts_to_models(cached)
    assert len(restored) == 1
    assert restored[0].published_at == "2025-09-17T12:34:56.000Z"
    assert restored[0].downloads == 123_456


def test_extract_published_at_prefers_created_at():
    value = _extract_published_at(
        {
            "createdAt": "2025-01-01T00:00:00.000Z",
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2025-01-01T00:00:00.000Z"


def test_extract_published_at_falls_back_to_last_modified():
    value = _extract_published_at(
        {
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2026-01-01T00:00:00.000Z"
