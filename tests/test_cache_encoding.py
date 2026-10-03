"""Regression tests for cross-platform cache encoding."""

from __future__ import annotations

import json
import time

import whichllm.models.benchmark as benchmark_mod
import whichllm.models.cache as cache_mod


class _ReadableCacheFile:
    def __init__(self, payload: dict):
        self.payload = payload
        self.encoding = None

    def exists(self) -> bool:
        return True

    def read_text(self, *, encoding: str | None = None) -> str:
        self.encoding = encoding
        return json.dumps(self.payload, ensure_ascii=False)


class _WritableCacheFile:
    def __init__(self):
        self.encoding = None
        self.text = None

    def write_text(self, text: str, *, encoding: str | None = None) -> int:
        self.encoding = encoding
        self.text = text
        return len(text)


def test_model_cache_reads_and_writes_utf8(monkeypatch, tmp_path):
    reader = _ReadableCacheFile(
        {
            "schema_version": cache_mod.CACHE_SCHEMA_VERSION,
            "cached_at": time.time(),
            "models": [{"id": "test/Omega-Ω"}],
        }
    )
    monkeypatch.setattr(cache_mod, "CACHE_FILE", reader)
    assert cache_mod.load_cache() == [{"id": "test/Omega-Ω"}]
    assert reader.encoding == "utf-8"

    writer = _WritableCacheFile()
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache_mod, "CACHE_FILE", writer)
    cache_mod.save_cache([{"id": "test/Omega-Ω"}])
    assert writer.encoding == "utf-8"
    assert "Ω" in writer.text


def test_model_cache_rejects_missing_provenance_schema(monkeypatch):
    reader = _ReadableCacheFile(
        {"cached_at": time.time(), "models": [{"id": "test/old-cache"}]}
    )
    monkeypatch.setattr(cache_mod, "CACHE_FILE", reader)

    assert cache_mod.load_cache() is None


def _write_non_utf8_cache(path, payload):
    """Write a cache file the way a pre-0.5.12 Windows install left it.

    Those versions wrote with ensure_ascii=False through the locale
    codepage, so a cached model id containing non-ASCII lands on disk as
    cp1252 bytes that are not valid UTF-8.
    """
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="cp1252")
    return path


def test_model_cache_survives_non_utf8_file(monkeypatch, tmp_path):
    """A cache file that is not valid UTF-8 is a cache miss, not a crash."""
    cache_file = _write_non_utf8_cache(
        tmp_path / "models.json",
        {
            "schema_version": cache_mod.CACHE_SCHEMA_VERSION,
            "cached_at": time.time(),
            "models": [{"id": "test/Café-Münster"}],
        },
    )
    monkeypatch.setattr(cache_mod, "CACHE_FILE", cache_file)

    assert cache_mod.load_cache() is None


def test_benchmark_cache_survives_non_utf8_file(monkeypatch, tmp_path):
    """Same for the benchmark cache, which has its own loader."""
    cache_file = _write_non_utf8_cache(
        tmp_path / "benchmarks.json",
        {"cached_at": time.time(), "scores": {"test/Café-Münster": 1.0}},
    )
    monkeypatch.setattr(benchmark_mod, "BENCHMARK_CACHE", cache_file)

    assert benchmark_mod.load_benchmark_cache() is None


def test_benchmark_cache_reads_and_writes_utf8(monkeypatch, tmp_path):
    reader = _ReadableCacheFile(
        {"cached_at": time.time(), "scores": {"test/Omega-Ω": 1.0}}
    )
    monkeypatch.setattr(benchmark_mod, "BENCHMARK_CACHE", reader)
    assert benchmark_mod.load_benchmark_cache() == {"test/Omega-Ω": 1.0}
    assert reader.encoding == "utf-8"

    writer = _WritableCacheFile()
    monkeypatch.setattr(benchmark_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(benchmark_mod, "BENCHMARK_CACHE", writer)
    benchmark_mod.save_benchmark_cache({"test/Omega-Ω": 1.0})
    assert writer.encoding == "utf-8"
    assert "Ω" in writer.text
