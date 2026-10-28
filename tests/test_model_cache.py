"""Model cache coverage flags and lifetime."""

from __future__ import annotations

import json
import time

import whichllm.models.cache as cache_mod


def _use_cache_file(monkeypatch, tmp_path):
    cache_file = tmp_path / "models.json"
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache_mod, "CACHE_FILE", cache_file)
    return cache_file


def _write_payload(cache_file, **fields):
    payload = {
        "schema_version": cache_mod.CACHE_SCHEMA_VERSION,
        "cached_at": time.time(),
        "models": [{"id": "org/Model-7B"}],
    }
    payload.update(fields)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


def test_text_only_cache_does_not_satisfy_vision_request(monkeypatch, tmp_path):
    _use_cache_file(monkeypatch, tmp_path)
    cache_mod.save_cache([{"id": "org/Model-7B"}], include_vision=False)

    assert cache_mod.load_cache() == [{"id": "org/Model-7B"}]
    assert cache_mod.load_cache(include_vision=True) is None


def test_vision_cache_satisfies_both_requests(monkeypatch, tmp_path):
    _use_cache_file(monkeypatch, tmp_path)
    cache_mod.save_cache([{"id": "org/Model-7B"}], include_vision=True)

    assert cache_mod.load_cache() == [{"id": "org/Model-7B"}]
    assert cache_mod.load_cache(include_vision=True) == [{"id": "org/Model-7B"}]


def test_cache_without_coverage_fields_is_text_only_and_complete(monkeypatch, tmp_path):
    cache_file = _use_cache_file(monkeypatch, tmp_path)
    _write_payload(cache_file, cached_at=time.time() - 3600)

    assert cache_mod.load_cache() == [{"id": "org/Model-7B"}]
    assert cache_mod.load_cache(include_vision=True) is None


def test_incomplete_cache_expires_sooner(monkeypatch, tmp_path):
    cache_file = _use_cache_file(monkeypatch, tmp_path)
    age = cache_mod.INCOMPLETE_TTL_SECONDS + 60
    assert age < cache_mod.DEFAULT_TTL_SECONDS

    _write_payload(cache_file, cached_at=time.time() - age, complete=False)
    assert cache_mod.load_cache() is None

    _write_payload(cache_file, cached_at=time.time() - age, complete=True)
    assert cache_mod.load_cache() == [{"id": "org/Model-7B"}]


def test_update_cached_models_keeps_fetch_time_and_flags(monkeypatch, tmp_path):
    cache_file = _use_cache_file(monkeypatch, tmp_path)
    fetched_at = time.time() - 5 * 3600
    _write_payload(
        cache_file, cached_at=fetched_at, include_vision=True, complete=False
    )

    cache_mod.update_cached_models([{"id": "org/Model-7B", "published_at": "x"}])

    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["cached_at"] == fetched_at
    assert data["include_vision"] is True
    assert data["complete"] is False
    assert data["models"] == [{"id": "org/Model-7B", "published_at": "x"}]


def test_update_cached_models_does_not_create_a_fresh_cache(monkeypatch, tmp_path):
    cache_file = _use_cache_file(monkeypatch, tmp_path)

    cache_mod.update_cached_models([{"id": "org/Model-7B"}])

    assert not cache_file.exists()


def test_non_object_cache_payload_is_a_miss(monkeypatch, tmp_path):
    cache_file = _use_cache_file(monkeypatch, tmp_path)
    cache_file.write_text("[]", encoding="utf-8")

    assert cache_mod.load_cache() is None
