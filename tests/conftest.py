"""Shared test isolation."""

from __future__ import annotations

import pytest

import whichllm.models.shape_fetch as shape_fetch


@pytest.fixture(autouse=True)
def _isolate_shape_lookups(monkeypatch, tmp_path):
    """Keep model-shape lookups off the network and the user's cache.

    Commands read config.json for top-ranked families. Tests that exercise
    that path patch ``_fetch_config_shape`` themselves.
    """
    monkeypatch.setattr(shape_fetch, "CACHE_DIR", tmp_path / "shape-cache")
    monkeypatch.setattr(
        shape_fetch, "SHAPE_CACHE_FILE", tmp_path / "shape-cache" / "shapes.json"
    )

    async def offline(client, repo_id, semaphore):
        return shape_fetch._MISSING

    monkeypatch.setattr(shape_fetch, "_fetch_config_shape", offline)


@pytest.fixture(autouse=True)
def _isolate_index(monkeypatch, tmp_path):
    """Commands read the prebuilt index first; keep tests off the network.

    With no reachable index and no cached copy every command falls back to
    the live path, which the tests mock. Index tests patch _fetch.
    """
    import httpx

    import whichllm.models.index_client as index_client

    cache = tmp_path / "index-cache"
    monkeypatch.setattr(index_client, "CACHE_DIR", cache)
    monkeypatch.setattr(index_client, "INDEX_CACHE_FILE", cache / "index.json.gz")
    monkeypatch.setattr(index_client, "INDEX_STATE_FILE", cache / "index-state.json")
    monkeypatch.delenv("WHICHLLM_INDEX_URL", raising=False)

    def offline(url, etag):
        raise httpx.ConnectError("tests are offline")

    monkeypatch.setattr(index_client, "_fetch", offline)


@pytest.fixture(autouse=True)
def _isolate_aa_benchmarks(monkeypatch):
    """Keep the per-benchmark AA fetch off the network.

    Tests that need task scores pass results to the builders directly or
    patch ``fetch_aa_eval_scores`` themselves.
    """
    import whichllm.models.benchmark_sources.aa_evals as aa_evals

    async def offline(client, evals):
        return {}

    monkeypatch.setattr(aa_evals, "fetch_aa_eval_scores", offline)
