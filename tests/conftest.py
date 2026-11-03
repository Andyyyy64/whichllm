"""Shared test isolation."""

from __future__ import annotations

import pytest

import whichllm.models.shape_fetch as shape_fetch


@pytest.fixture
def symlink():
    """Make a symlink, or skip where this user may not make one.

    Windows refuses symlinks (error 1314) outside Developer Mode or an
    elevated shell.
    """

    def make(link, target):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            pytest.skip("Windows refuses symlinks to this user")

    return make


@pytest.fixture(autouse=True)
def _isolate_device_tree(monkeypatch, tmp_path):
    """Keep Jetson detection off the host's device tree and GPU device nodes.

    On a Jetson the real ones would turn "no NVIDIA tool found" into a GPU.
    """
    import whichllm.hardware.jetson as jetson

    monkeypatch.setattr(jetson, "_COMPATIBLE", tmp_path / "no-device-tree")
    monkeypatch.setattr(jetson, "_GPU_DEVICE_NODES", (tmp_path / "no-gpu-node",))
    jetson.detect_jetson_module.cache_clear()
    yield
    jetson.detect_jetson_module.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_installed_engines(monkeypatch):
    """`whichllm engines` finds nothing installed unless a test says otherwise.

    The real search runs binaries on PATH and asks local ports.
    """
    monkeypatch.setattr(
        "whichllm.engines.detect.detect_installed", lambda cards, os_name: {}
    )


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


@pytest.fixture(autouse=True)
def _isolate_ram_modules(monkeypatch, tmp_path):
    """Keep RAM module probing off the machine running the tests.

    Tests that cover the probes pass fixture files or patch them directly.
    """
    import whichllm.hardware.ram_bandwidth as ram_bandwidth

    monkeypatch.setattr(ram_bandwidth, "_linux_memory_modules", lambda: [])
    monkeypatch.setattr(ram_bandwidth, "_is_wsl", lambda: False)
    monkeypatch.setattr(ram_bandwidth, "_cache_dir", lambda: tmp_path / "ram-cache")
    monkeypatch.setattr(
        ram_bandwidth, "_windows_memory_modules", lambda executable="powershell": []
    )
