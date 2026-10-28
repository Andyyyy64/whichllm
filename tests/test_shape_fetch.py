"""Fetching, caching and sharing model shapes from config.json."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

import whichllm.models.shape_fetch as shape_fetch
from whichllm.models.shape import parse_model_shape
from whichllm.models.types import ModelInfo

CONFIGS = json.loads(
    (Path(__file__).parent / "fixtures" / "model_configs.json").read_text(
        encoding="utf-8"
    )
)
QWEN_SHAPE = parse_model_shape(CONFIGS["Qwen/Qwen3-8B"])


def _model(model_id, family="qwen3-8b", params=8_190_735_360, **kwargs):
    return ModelInfo(
        id=model_id,
        family_id=family,
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=params,
        **kwargs,
    )


@pytest.fixture
def hub(monkeypatch):
    """Serve config.json responses from a dict: repo -> status or config."""
    responses: dict[str, object] = {}
    requested: list[str] = []
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    async def fake_get(client, url, **kwargs):
        repo = url.split("https://huggingface.co/", 1)[1].split("/resolve/")[0]
        requested.append(repo)
        request = httpx.Request("GET", url)
        answer = responses.get(repo, 404)
        if isinstance(answer, int):
            return httpx.Response(answer, request=request)
        return httpx.Response(200, json=answer, request=request)

    monkeypatch.setattr(shape_fetch, "get_with_retries", fake_get)
    # Undo the offline stub from conftest so the real fetch path runs.
    monkeypatch.setattr(shape_fetch, "_fetch_config_shape", _REAL_FETCH, raising=True)
    return responses, requested


_REAL_FETCH = shape_fetch._fetch_config_shape


def test_fetches_caches_and_attaches(hub):
    responses, requested = hub
    responses["Qwen/Qwen3-8B"] = CONFIGS["Qwen/Qwen3-8B"]
    model = _model("Qwen/Qwen3-8B")

    attached = asyncio.run(shape_fetch.fetch_family_shapes([model], ["qwen3-8b"]))

    assert attached == 1
    assert model.shape == QWEN_SHAPE
    assert requested == ["Qwen/Qwen3-8B"]

    # A later run reads the disk cache without any request.
    fresh = _model("Qwen/Qwen3-8B")
    assert shape_fetch.attach_cached_shapes([fresh]) == 1
    assert fresh.shape == QWEN_SHAPE
    asyncio.run(shape_fetch.fetch_family_shapes([fresh], ["qwen3-8b"]))
    assert requested == ["Qwen/Qwen3-8B"]


def test_quantized_upload_reads_base_config_first(hub):
    responses, requested = hub
    responses["Qwen/Qwen3-8B"] = CONFIGS["Qwen/Qwen3-8B"]
    gguf = _model(
        "unsloth/Qwen3-8B-GGUF",
        base_model="Qwen/Qwen3-8B",
        base_model_relation="quantized",
    )

    asyncio.run(shape_fetch.fetch_family_shapes([gguf], ["qwen3-8b"]))

    assert requested == ["Qwen/Qwen3-8B"]
    assert gguf.shape == QWEN_SHAPE


def test_gated_official_repo_borrows_same_size_family_member(hub):
    responses, requested = hub
    responses["google/gemma-3-27b-it"] = 401
    responses["unsloth/gemma-3-27b-it"] = CONFIGS["google/gemma-3-27b-it"]
    official = _model("google/gemma-3-27b-it", family="gemma3", params=27_432_406_640)
    mirror = _model(
        "unsloth/gemma-3-27b-it",
        family="gemma3",
        params=27_432_406_640,
        downloads=10,
        base_model="google/gemma-3-27b-it",
        base_model_relation="finetune",
    )
    draft = _model(
        "someone/gemma-3-27b-it-draft", family="gemma3", params=1_000_000_000
    )

    asyncio.run(shape_fetch.fetch_family_shapes([official, mirror, draft], ["gemma3"]))

    assert official.shape is not None
    assert official.shape == mirror.shape
    # A much smaller upload in the same family does not inherit the layers.
    assert draft.shape is None


def test_permanent_miss_is_cached_but_transient_failure_is_not(hub, monkeypatch):
    responses, requested = hub
    responses["org/Gated-7B"] = 401
    responses["org/Flaky-7B"] = 503

    gated = _model("org/Gated-7B", family="gated")
    flaky = _model("org/Flaky-7B", family="flaky")
    asyncio.run(shape_fetch.fetch_family_shapes([gated, flaky], ["gated", "flaky"]))
    assert gated.shape is None and flaky.shape is None

    requested.clear()
    asyncio.run(shape_fetch.fetch_family_shapes([gated, flaky], ["gated", "flaky"]))
    assert requested == ["org/Flaky-7B"]

    # After a day the miss is retried.
    entries = json.loads(shape_fetch.SHAPE_CACHE_FILE.read_text())["entries"]
    entries["org/Gated-7B"]["fetched_at"] = (
        time.time() - shape_fetch.MISSING_TTL_SECONDS - 1
    )
    shape_fetch._save_entries(entries)
    requested.clear()
    asyncio.run(shape_fetch.fetch_family_shapes([gated], ["gated"]))
    assert requested == ["org/Gated-7B"]


def test_family_and_request_limits(hub, monkeypatch):
    responses, requested = hub
    monkeypatch.setattr(shape_fetch, "MAX_FAMILIES_PER_RUN", 2)
    models = [
        _model(f"org/M{i}-{j}", family=f"f{i}", downloads=10 - j)
        for i in range(4)
        for j in range(5)
    ]

    asyncio.run(shape_fetch.fetch_family_shapes(models, ["f0", "f1", "f2", "f3"]))

    assert {r.split("-")[0] for r in requested} == {"org/M0", "org/M1"}
    assert len(requested) == 2 * shape_fetch.MAX_SOURCES_PER_FAMILY


def test_hf_endpoint_is_honoured(monkeypatch):
    from whichllm.models.hf import _hf_file_url

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")
    assert _hf_file_url("Qwen/Qwen3-8B", "config.json") == (
        "https://hf-mirror.example/Qwen/Qwen3-8B/resolve/main/config.json"
    )
