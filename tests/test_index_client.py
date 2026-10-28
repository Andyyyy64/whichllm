"""Downloading, revalidating and falling back for the prebuilt index."""

from __future__ import annotations

import json
import time

import httpx
import pytest

import whichllm.models.index_client as index_client
from whichllm.models.index_format import DEFAULT_INDEX_URL, encode_index


def _payload(model_id: str = "org/A-8B", built_at: str = "2026-10-29T03:17:00Z"):
    return {
        "schema_version": 1,
        "built_at": built_at,
        "models": [
            {
                "id": model_id,
                "family_id": model_id.lower(),
                "name": model_id.rsplit("/", 1)[-1],
                "parameter_count": 8_000_000_000,
            }
        ],
        "benchmark_scores": {model_id: 70.0},
    }


@pytest.fixture
def server(monkeypatch):
    """Answer index requests from a queue; record each request's ETag."""
    state = {"responses": [], "requests": []}

    def fake_fetch(url, etag):
        state["requests"].append((url, etag))
        answer = state["responses"].pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body, headers = answer
        return httpx.Response(
            status, content=body, headers=headers, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(index_client, "_fetch", fake_fetch)
    return state


def _ok(payload, etag='"v1"'):
    return (200, encode_index(payload), {"ETag": etag})


def _age_state(seconds: float) -> None:
    state = json.loads(index_client.INDEX_STATE_FILE.read_text())
    state["checked_at"] = time.time() - seconds
    index_client.INDEX_STATE_FILE.write_text(json.dumps(state))


def test_downloads_caches_and_reuses_without_a_request(server):
    server["responses"] = [_ok(_payload())]

    first = index_client.load_index()
    second = index_client.load_index()

    assert [m.id for m in first.models] == ["org/A-8B"]
    assert first.benchmark_scores == {"org/A-8B": 70.0}
    assert not first.offline
    assert [m.id for m in second.models] == ["org/A-8B"]
    assert server["requests"] == [(DEFAULT_INDEX_URL, None)]


def test_revalidates_with_etag_after_the_window(server):
    server["responses"] = [_ok(_payload()), (304, b"", {})]
    index_client.load_index()
    _age_state(index_client.REVALIDATE_SECONDS + 1)

    catalog = index_client.load_index()

    assert [m.id for m in catalog.models] == ["org/A-8B"]
    assert server["requests"][1] == (DEFAULT_INDEX_URL, '"v1"')
    # A 304 restarts the window.
    index_client.load_index()
    assert len(server["requests"]) == 2


def test_refresh_downloads_a_new_index(server):
    server["responses"] = [_ok(_payload()), _ok(_payload("org/B-8B"), '"v2"')]
    index_client.load_index()

    catalog = index_client.load_index(refresh=True)

    assert [m.id for m in catalog.models] == ["org/B-8B"]


def test_offline_uses_cached_copy_of_any_age(server):
    server["responses"] = [_ok(_payload()), httpx.ConnectError("down")]
    index_client.load_index()
    _age_state(30 * 24 * 3600)

    catalog = index_client.load_index()

    assert catalog.offline
    assert [m.id for m in catalog.models] == ["org/A-8B"]


def test_no_index_and_no_cache_means_live_fetch(server):
    server["responses"] = [httpx.ConnectError("down")]
    assert index_client.load_index() is None


def test_invalid_download_keeps_the_cached_index(server):
    server["responses"] = [_ok(_payload()), (200, b"<html>maintenance</html>", {})]
    index_client.load_index()

    catalog = index_client.load_index(refresh=True)

    assert catalog.offline
    assert [m.id for m in catalog.models] == ["org/A-8B"]


def test_http_error_without_cache_means_live_fetch(server):
    server["responses"] = [(404, b"", {})]
    assert index_client.load_index() is None


def test_index_url_can_be_overridden_or_disabled(server, monkeypatch):
    server["responses"] = [_ok(_payload())]
    monkeypatch.setenv("WHICHLLM_INDEX_URL", "https://mirror.example/index.json.gz")
    index_client.load_index()
    assert server["requests"][0][0] == "https://mirror.example/index.json.gz"

    # A copy cached from another URL is not reused.
    server["responses"] = [httpx.ConnectError("down")]
    monkeypatch.setenv("WHICHLLM_INDEX_URL", "https://other.example/index.json.gz")
    assert index_client.load_index() is None

    monkeypatch.setenv("WHICHLLM_INDEX_URL", "off")
    assert index_client.load_index() is None


def test_age_from_build_time():
    catalog = index_client.IndexCatalog(
        models=[], benchmark_scores={}, built_at="2026-01-01T00:00:00Z"
    )
    assert catalog.age_seconds > 0
    assert index_client.IndexCatalog([], {}, "not a date").age_seconds is None
