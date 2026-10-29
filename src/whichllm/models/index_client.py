"""Download and cache the prebuilt whichllm index.

One conditional GET replaces the Hub list calls, the curated model lookups,
the benchmark scrapes and the config.json reads of a live fetch. The file is
revalidated with its ETag every few hours; when the network is down a cached
index of any age is used, and without one callers fall back to a live fetch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from whichllm.models.index_format import DEFAULT_INDEX_URL, decode_index
from whichllm.models.serialization import dicts_to_models
from whichllm.models.types import ModelInfo
from whichllm.utils import _cache_dir

logger = logging.getLogger(__name__)

CACHE_DIR = _cache_dir()
INDEX_CACHE_FILE = CACHE_DIR / "index.json.gz"
INDEX_STATE_FILE = CACHE_DIR / "index-state.json"
REVALIDATE_SECONDS = 6 * 3600
STALE_AFTER_SECONDS = 3 * 24 * 3600
_TIMEOUT_SECONDS = 15.0
_DISABLED = {"", "0", "off", "false", "none"}


@dataclass
class IndexCatalog:
    models: list[ModelInfo]
    benchmark_scores: dict[str, float]
    built_at: str
    # True when the network was unreachable and a cached copy was used.
    offline: bool = False
    # Per-profile scores (coding, math, vision); empty for an older index.
    task_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float | None:
        try:
            built = datetime.strptime(self.built_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
        return time.time() - built.replace(tzinfo=timezone.utc).timestamp()


def index_url() -> str | None:
    """Index location; ``WHICHLLM_INDEX_URL=off`` disables the index."""
    value = os.environ.get("WHICHLLM_INDEX_URL")
    if value is None:
        return DEFAULT_INDEX_URL
    return None if value.strip().lower() in _DISABLED else value.strip()


def _read_state() -> dict:
    try:
        state = json.loads(INDEX_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _read_cached(url: str) -> dict | None:
    if _read_state().get("url") != url:
        return None
    try:
        return decode_index(INDEX_CACHE_FILE.read_bytes())
    except (OSError, ValueError) as e:
        logger.debug(f"Cached index unusable: {e}")
        return None


def _save(raw: bytes | None, url: str, etag: str | None) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            INDEX_CACHE_FILE.write_bytes(raw)
        state = {"url": url, "etag": etag, "checked_at": time.time()}
        INDEX_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError as e:
        logger.debug(f"Could not cache the index: {e}")


def _fetch(url: str, etag: str | None) -> httpx.Response:
    headers = {"If-None-Match": etag} if etag else {}
    return httpx.get(
        url, headers=headers, timeout=_TIMEOUT_SECONDS, follow_redirects=True
    )


def _catalog(payload: dict, offline: bool = False) -> IndexCatalog:
    return IndexCatalog(
        models=dicts_to_models(payload["models"]),
        benchmark_scores=payload["benchmark_scores"],
        built_at=payload["built_at"],
        offline=offline,
        task_scores=payload.get("task_scores") or {},
    )


def load_index(refresh: bool = False) -> IndexCatalog | None:
    """The current index, a cached copy if offline, or None to fetch live."""
    url = index_url()
    if url is None:
        return None
    state = _read_state()
    cached = _read_cached(url)
    fresh = time.time() - state.get("checked_at", 0) < REVALIDATE_SECONDS
    if cached is not None and fresh and not refresh:
        return _catalog(cached)

    try:
        resp = _fetch(url, state.get("etag") if cached is not None else None)
    except httpx.HTTPError as e:
        logger.debug(f"Index fetch failed: {e}")
        return _catalog(cached, offline=True) if cached is not None else None

    if resp.status_code == 304 and cached is not None:
        _save(None, url, state.get("etag"))
        return _catalog(cached)
    if resp.status_code == 200:
        try:
            payload = decode_index(resp.content)
        except ValueError as e:
            logger.warning(f"Downloaded index is invalid ({e}); not using it")
        else:
            _save(resp.content, url, resp.headers.get("ETag"))
            return _catalog(payload)
    else:
        logger.debug(f"Index fetch returned HTTP {resp.status_code}")
    return _catalog(cached, offline=True) if cached is not None else None
