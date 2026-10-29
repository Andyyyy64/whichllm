"""Index building, encoding and publishing gates."""

from __future__ import annotations

import asyncio
import gzip
import json

import pytest

import whichllm.models.index_build as index_build
from whichllm.models.benchmark_fetch import BenchmarkData
from whichllm.models.fetcher import ModelFetchReport
from whichllm.models.hf import _FRONTIER_MODEL_IDS
from whichllm.models.index_format import (
    INDEX_SCHEMA_VERSION,
    decode_index,
    encode_index,
)
from whichllm.models.types import ModelInfo


def _model(model_id: str, published_at: str | None = None) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=8_000_000_000,
        published_at=published_at,
    )


@pytest.fixture
def sources(monkeypatch):
    """Stub every network source the builder reads."""
    state = {
        "models": [_model("org/A-8B"), _model("org/B-8B", "2026-01-01T00:00:00.000Z")],
        "scores": {"org/A-8B": 70.0},
        "tasks": {"coding": {"org/A-8B": 74.0}},
        "benchmark_ids": [],
        "date_requests": [],
        "report_failed": [],
    }

    async def fake_fetch_models(*, include_vision, report):
        assert include_vision
        report.failed.extend(state["report_failed"])
        return state["models"]

    async def fake_benchmarks(model_ids):
        state["benchmark_ids"].extend(model_ids)
        return BenchmarkData(general=state["scores"], tasks=state["tasks"])

    async def fake_shapes(models, families, *, max_families, cache_file):
        assert max_families == len(families)
        return 0

    async def fake_dates(model_ids):
        state["date_requests"].extend(model_ids)
        return {i: "2026-02-02T00:00:00.000Z" for i in model_ids}

    monkeypatch.setattr(index_build, "fetch_models", fake_fetch_models)
    monkeypatch.setattr(index_build, "fetch_benchmark_data", fake_benchmarks)
    monkeypatch.setattr(index_build, "fetch_family_shapes", fake_shapes)
    monkeypatch.setattr(index_build, "fetch_model_published_at", fake_dates)
    return state


def test_build_produces_a_decodable_index(sources):
    payload, report = asyncio.run(
        index_build.build_index(whichllm_version="0.6.0", git_commit="abc123")
    )

    assert report.complete
    assert payload["schema_version"] == INDEX_SCHEMA_VERSION
    assert payload["builder"] == {"whichllm_version": "0.6.0", "git_commit": "abc123"}
    assert payload["stats"]["models"] == 2
    assert payload["benchmark_scores"] == {"org/A-8B": 70.0}
    assert payload["task_scores"] == {"coding": {"org/A-8B": 74.0}}
    assert payload["stats"]["task_scores"] == {"coding": 1}
    # AA display names are matched against the catalog's repo ids.
    assert sources["benchmark_ids"] == ["org/A-8B", "org/B-8B"]
    assert decode_index(encode_index(payload)) == payload


def test_publish_dates_come_from_previous_index_first(sources):
    previous = {
        "models": [{"id": "org/A-8B", "published_at": "2025-12-12T00:00:00.000Z"}]
    }

    payload, _ = asyncio.run(index_build.build_index(previous=previous))

    dates = {m["id"]: m["published_at"] for m in payload["models"]}
    assert dates["org/A-8B"] == "2025-12-12T00:00:00.000Z"
    assert dates["org/B-8B"] == "2026-01-01T00:00:00.000Z"
    assert sources["date_requests"] == []


def test_encoding_is_deterministic():
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "built_at": "2026-10-29T00:00:00Z",
        "models": [{"id": "org/A"}],
        "benchmark_scores": {"b": 1.0, "a": 2.0},
    }
    assert encode_index(payload) == encode_index(dict(reversed(payload.items())))


@pytest.mark.parametrize(
    "raw",
    [
        b"not gzip",
        gzip.compress(b"[]"),
        gzip.compress(json.dumps({"schema_version": 99}).encode()),
    ],
)
def test_decode_rejects_invalid_files(raw):
    with pytest.raises(ValueError):
        decode_index(raw)


def _healthy_payload(
    models: int = 900, benchmarks: int = 3000, tasks: int = 60
) -> dict:
    ids = list(_FRONTIER_MODEL_IDS) + [f"org/M{i}" for i in range(models)]
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "built_at": "2026-10-29T00:00:00Z",
        "stats": {"shapes": len(ids)},
        "models": [{"id": i} for i in ids],
        "benchmark_scores": {f"org/S{i}": 50.0 for i in range(benchmarks)},
        "task_scores": {
            "coding": {f"org/S{i}": 50.0 for i in range(tasks)},
            "vision": {f"org/V{i}": 50.0 for i in range(tasks)},
        },
    }


def test_healthy_build_passes_gates():
    assert index_build.quality_problems(_healthy_payload(), ModelFetchReport()) == []


def test_gates_catch_broken_builds():
    report = ModelFetchReport(failed=["Qwen/Qwen3.8-27B"])
    payload = _healthy_payload(models=10, benchmarks=10)
    payload["models"] = payload["models"][len(_FRONTIER_MODEL_IDS) :]
    payload["stats"]["shapes"] = 0

    problems = index_build.quality_problems(payload, report)

    text = "\n".join(problems)
    assert "requests failed" in text
    assert "only 10 models" in text
    assert "only 10 benchmark ids" in text
    assert "curated recent models" in text
    assert "shapes for 0" in text


def test_unavailable_curated_repos_do_not_count_against_coverage():
    payload = _healthy_payload()
    payload["models"] = payload["models"][len(_FRONTIER_MODEL_IDS) :]
    report = ModelFetchReport(unavailable=list(_FRONTIER_MODEL_IDS))
    assert index_build.quality_problems(payload, report) == []


def test_large_drop_from_previous_index_is_refused():
    previous = _healthy_payload(models=2000)
    problems = index_build.quality_problems(
        _healthy_payload(models=900), ModelFetchReport(), previous
    )
    assert any("models dropped" in p for p in problems)


def test_index_without_task_scores_still_decodes():
    payload = _healthy_payload()
    del payload["task_scores"]
    assert decode_index(encode_index(payload))["benchmark_scores"]


def test_malformed_task_scores_are_rejected():
    payload = _healthy_payload()
    payload["task_scores"] = {"coding": {"org/S1": "high"}}
    with pytest.raises(ValueError, match="task_scores"):
        decode_index(encode_index(payload))


def test_missing_task_scores_block_publishing():
    problems = index_build.quality_problems(
        _healthy_payload(tasks=3), ModelFetchReport()
    )
    assert any("only 3 coding task scores" in p for p in problems)
    assert any("only 3 vision task scores" in p for p in problems)


def test_large_task_score_drop_is_refused():
    problems = index_build.quality_problems(
        _healthy_payload(tasks=25), ModelFetchReport(), _healthy_payload(tasks=80)
    )
    assert any("coding task scores dropped from 80 to 25" in p for p in problems)
