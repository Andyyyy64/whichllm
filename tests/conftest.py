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
