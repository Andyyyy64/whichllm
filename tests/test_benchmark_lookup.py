"""Tests for benchmark lookup direct/inherited semantics."""

import asyncio

import whichllm.models.benchmark_sources as benchmark_sources
from whichllm.models.benchmark import (
    _lineage_recency_factor,
    build_line_bucket_index,
    build_score_index,
    fetch_benchmark_scores,
    lookup_benchmark,
    lookup_benchmark_evidence,
)


def test_fetch_benchmark_scores_disables_brotli_accept_encoding(monkeypatch):
    encodings: list[str] = []

    async def fake_source(client):
        encodings.append(client.headers["accept-encoding"])
        return {}

    monkeypatch.setattr(
        benchmark_sources, "fetch_leaderboard_with_fallback", fake_source
    )
    monkeypatch.setattr(benchmark_sources, "fetch_arena_scores", fake_source)
    monkeypatch.setattr(benchmark_sources, "fetch_aa_index_scores", fake_source)
    monkeypatch.setattr(benchmark_sources, "fetch_aider_polyglot_scores", fake_source)
    monkeypatch.setattr(benchmark_sources, "fetch_vision_scores", fake_source)
    monkeypatch.setattr(benchmark_sources, "get_livebench_data", lambda: {})

    assert asyncio.run(fetch_benchmark_scores()) == {}
    assert set(encodings) == {"gzip, deflate"}


def test_lookup_benchmark_model_id_match_is_direct():
    scores = {"Qwen/Qwen2.5-7B-Instruct": 70.0}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "Qwen/Qwen2.5-7B-Instruct",
        None,
        scores,
        ci,
        line,
    )
    assert result == (70.0, True)


def test_lookup_benchmark_base_model_match_is_inherited():
    scores = {"google/gemma-3-27b-it": 82.2}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        "google/gemma-3-27b-it",
        scores,
        ci,
        line,
    )
    assert result == (82.2, False)


def test_lookup_benchmark_gguf_suffix_match_is_inherited():
    scores = {"Qwen/Qwen2.5-7B-Instruct": 70.0}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        None,
        scores,
        ci,
        line,
    )
    assert result == (70.0, False)


def test_lookup_benchmark_community_gguf_without_base_model_matches_official_id():
    scores = {"Qwen/Qwen3.6-27B": 83.5}
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "unsloth/Qwen3.6-27B-GGUF",
        None,
        scores,
        ci,
        line,
        buckets,
    )
    assert result.source == "variant"
    assert result.score == 83.5


def test_lookup_benchmark_community_gguf_underscore_name_matches_official_id():
    scores = {"Qwen/Qwen3.6-35B-A3B": 86.0}
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "unsloth/Qwen_Qwen3.6-35B-A3B-GGUF",
        None,
        scores,
        ci,
        line,
        buckets,
    )
    assert result.source == "variant"
    assert result.score == 86.0


def test_lookup_benchmark_community_gguf_keeps_params_guard():
    scores = {"Qwen/Qwen3.6-27B": 83.5}
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "unsloth/Qwen3.6-27B-GGUF",
        None,
        scores,
        ci,
        line,
        buckets,
        actual_params_b=6.6,
    )
    assert result.source != "variant"


def test_lookup_benchmark_community_gguf_beats_self_reported_score():
    scores = {"Qwen/Qwen3.6-27B": 83.5}
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "unsloth/Qwen3.6-27B-GGUF",
        None,
        scores,
        ci,
        line,
        buckets,
        self_reported_score=12.0,
    )
    assert result.source == "variant"
    assert result.score == 83.5


def test_lookup_benchmark_evidence_direct_has_full_confidence():
    scores = {"Qwen/Qwen2.5-7B-Instruct": 70.0}
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "Qwen/Qwen2.5-7B-Instruct",
        None,
        scores,
        ci,
        line,
        buckets,
    )
    assert result.source == "direct"
    assert result.confidence == 1.0
    assert result.score == 70.0


def test_lookup_benchmark_evidence_line_uses_size_aware_interpolation():
    scores = {
        "Qwen/Qwen3-8B-Instruct": 65.0,
        "Qwen/Qwen3-32B-Instruct": 85.0,
    }
    ci, line = build_score_index(scores)
    buckets = build_line_bucket_index(scores)
    result = lookup_benchmark_evidence(
        "Qwen/Qwen3-14B-Instruct-GGUF",
        None,
        scores,
        ci,
        line,
        buckets,
    )
    assert result.source == "line_interp"
    assert result.score is not None
    assert 65.0 < result.score < 85.0
    assert result.confidence > 0.2


def test_lineage_recency_t5gemma_variants_are_not_demoted_as_old_gemma():
    assert _lineage_recency_factor("google/t5gemma-4b") == 1.0
    assert _lineage_recency_factor("google/t5-gemma-4b") == 1.0
    assert _lineage_recency_factor("google/t5_gemma-4b") == 1.0
    assert _lineage_recency_factor("google/gemma-2-2b") < 1.0
