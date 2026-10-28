"""Local JSON cache with TTL for model data."""

from __future__ import annotations

import json
import logging
import time

from whichllm.utils import _cache_dir

logger = logging.getLogger(__name__)

CACHE_DIR = _cache_dir()
CACHE_FILE = CACHE_DIR / "models.json"
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours
# A catalog fetched while Hugging Face was rate limiting us is missing models;
# keep it only long enough to avoid hammering the API, then fetch again.
INCOMPLETE_TTL_SECONDS = 30 * 60
CACHE_SCHEMA_VERSION = 4


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_cache_payload() -> dict | None:
    """Return the cache payload if it is readable and current, else None."""
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"Cache corrupted: {e}")
        return None
    if not isinstance(data, dict):
        logger.debug("Cache corrupted: payload is not an object")
        return None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        logger.debug("Model cache schema is outdated")
        return None
    return data


def load_cache(include_vision: bool = False) -> list[dict] | None:
    """Load cached model data if valid. Returns None if expired or missing.

    A cache written without vision candidates does not satisfy a caller that
    needs them. Caches written before these fields existed count as complete
    and without vision candidates.
    """
    data = _read_cache_payload()
    if data is None:
        return None
    if include_vision and not data.get("include_vision", False):
        logger.debug("Model cache has no vision candidates")
        return None
    ttl = DEFAULT_TTL_SECONDS if data.get("complete", True) else INCOMPLETE_TTL_SECONDS
    if time.time() - data.get("cached_at", 0) > ttl:
        logger.debug("Cache expired")
        return None
    return data.get("models", [])


def save_cache(
    models: list[dict],
    *,
    include_vision: bool = False,
    complete: bool = True,
) -> None:
    """Save a freshly fetched model catalog to cache."""
    _write_cache(
        models,
        cached_at=time.time(),
        include_vision=include_vision,
        complete=complete,
    )


def update_cached_models(models: list[dict]) -> None:
    """Rewrite cached models without extending the cache lifetime.

    Used after enriching cached models (for example backfilled publish
    dates). The original fetch time and coverage flags are kept, so the
    enrichment never makes an old or partial catalog look fresh.
    """
    data = _read_cache_payload()
    if data is None:
        logger.debug("No current model cache to update")
        return
    _write_cache(
        models,
        cached_at=data.get("cached_at", 0),
        include_vision=data.get("include_vision", False),
        complete=data.get("complete", True),
    )


def _write_cache(
    models: list[dict],
    *,
    cached_at: float,
    include_vision: bool,
    complete: bool,
) -> None:
    _ensure_cache_dir()
    data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at": cached_at,
        "include_vision": include_vision,
        "complete": complete,
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.debug(f"Saved {len(models)} models to cache")
