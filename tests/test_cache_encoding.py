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
        {"cached_at": time.time(), "models": [{"id": "test/Omega-Ω"}]}
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
