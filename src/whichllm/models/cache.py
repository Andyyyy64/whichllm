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
CACHE_SCHEMA_VERSION = 3


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> list[dict] | None:
    """Load cached model data if valid. Returns None if expired or missing."""
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if data.get("schema_version") != CACHE_SCHEMA_VERSION:
            logger.debug("Model cache schema is outdated")
            return None
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > DEFAULT_TTL_SECONDS:
            logger.debug("Cache expired")
            return None
        return data.get("models", [])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
        logger.debug(f"Cache corrupted: {e}")
        return None


def save_cache(models: list[dict]) -> None:
    """Save model data to cache."""
    _ensure_cache_dir()
    data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at": time.time(),
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.debug(f"Saved {len(models)} models to cache")
