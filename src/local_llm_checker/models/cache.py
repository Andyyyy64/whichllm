"""Local JSON cache with TTL for model data."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "local-llm-checker"
CACHE_FILE = CACHE_DIR / "models.json"
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> list[dict] | None:
    """Load cached model data if valid. Returns None if expired or missing."""
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > DEFAULT_TTL_SECONDS:
            logger.debug("Cache expired")
            return None
        return data.get("models", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Cache corrupted: {e}")
        return None


def save_cache(models: list[dict]) -> None:
    """Save model data to cache."""
    _ensure_cache_dir()
    data = {
        "cached_at": time.time(),
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False))
    logger.debug(f"Saved {len(models)} models to cache")
