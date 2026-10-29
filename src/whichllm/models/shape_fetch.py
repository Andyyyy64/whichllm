"""Fetch, cache and attach model shapes from repositories' config.json.

The Hub API list calls do not include layer counts or attention heads, so the
shape of a model needs one extra request for its config.json. To keep runs
fast, shapes are fetched only for the families that reach the top of a
ranking, cached on disk for a week, and shared within a family: quantized
and converted uploads have the same layers as the checkpoint they came from.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from whichllm.models.hf import _hf_file_url, _hf_headers
from whichllm.models.http import get_with_retries
from whichllm.models.moe_fields import fill_moe_from_shape
from whichllm.models.shape import ModelShape, parse_model_shape
from whichllm.models.types import ModelInfo
from whichllm.utils import _cache_dir

logger = logging.getLogger(__name__)

CACHE_DIR = _cache_dir()
SHAPE_CACHE_FILE = CACHE_DIR / "shapes.json"
SHAPE_TTL_SECONDS = 7 * 24 * 3600
# A missing, gated or unparseable config is retried sooner.
MISSING_TTL_SECONDS = 24 * 3600
SHAPE_CACHE_SCHEMA_VERSION = 1

# Families whose shape is looked up after one ranking pass, and requests
# per family before giving up.
MAX_FAMILIES_PER_RUN = 25
MAX_SOURCES_PER_FAMILY = 3
_CONCURRENCY = 8

# A family member may lend its shape only if its size matches; this keeps
# draft, MTP and pruned uploads that share a family id from donating layers.
_SHARE_PARAM_RATIO = (0.9, 1.1)

_MISSING = object()


def _load_entries(cache_file: Path | None = None) -> dict[str, dict]:
    cache_file = cache_file or SHAPE_CACHE_FILE
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"Shape cache corrupted: {e}")
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != (
        SHAPE_CACHE_SCHEMA_VERSION
    ):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_entries(entries: dict[str, dict], cache_file: Path | None = None) -> None:
    cache_file = cache_file or SHAPE_CACHE_FILE
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": SHAPE_CACHE_SCHEMA_VERSION, "entries": entries}
    cache_file.write_text(json.dumps(data), encoding="utf-8")


def _cached_shape(entry: object, now: float):
    """Return a ModelShape, None for a known miss, or _MISSING if stale."""
    if not isinstance(entry, dict):
        return _MISSING
    shape = ModelShape.from_dict(entry.get("shape"))
    ttl = SHAPE_TTL_SECONDS if shape is not None else MISSING_TTL_SECONDS
    if now - entry.get("fetched_at", 0) > ttl:
        return _MISSING
    return shape


def shape_sources(model: ModelInfo) -> list[str]:
    """Repositories whose config.json describes this model, best first."""
    own = model.id
    base = model.base_model
    if base and model.base_model_relation == "quantized":
        return [base, own]
    return [own, base] if base and base != own else [own]


def _family_sources(members: list[ModelInfo]) -> list[str]:
    ordered: list[str] = []
    for member in sorted(members, key=lambda m: m.downloads, reverse=True):
        for source in shape_sources(member):
            if source not in ordered:
                ordered.append(source)
    return ordered


def _same_size(a: ModelInfo, b: ModelInfo) -> bool:
    if not a.parameter_count or not b.parameter_count:
        return False
    ratio = a.parameter_count / b.parameter_count
    return _SHARE_PARAM_RATIO[0] <= ratio <= _SHARE_PARAM_RATIO[1]


def _attach(models: list[ModelInfo], shapes: dict[str, ModelShape | None]) -> int:
    """Give each shapeless model its own or a same-size family member's shape."""
    attached = 0
    by_family: dict[str, list[ModelInfo]] = {}
    for model in models:
        by_family.setdefault(model.family_id, []).append(model)
    for model in models:
        if model.shape is not None:
            continue
        shape = next(
            (shapes[s] for s in shape_sources(model) if shapes.get(s) is not None),
            None,
        )
        if shape is None:
            for other in by_family[model.family_id]:
                donor = other.shape or next(
                    (
                        shapes[s]
                        for s in shape_sources(other)
                        if shapes.get(s) is not None
                    ),
                    None,
                )
                if donor is not None and _same_size(model, other):
                    shape = donor
                    break
        if shape is not None:
            model.shape = shape
            fill_moe_from_shape(model)
            attached += 1
    return attached


def attach_cached_shapes(
    models: list[ModelInfo], cache_file: Path | None = None
) -> int:
    """Attach shapes already in the disk cache; returns how many were set."""
    now = time.time()
    shapes = {
        repo: shape
        for repo, entry in _load_entries(cache_file).items()
        if (shape := _cached_shape(entry, now)) is not _MISSING
    }
    return _attach(models, shapes)


async def _fetch_config_shape(
    client: httpx.AsyncClient, repo_id: str, semaphore: asyncio.Semaphore
):
    """ModelShape, None for a permanent miss, or _MISSING for a transient one."""
    async with semaphore:
        try:
            resp = await get_with_retries(client, _hf_file_url(repo_id, "config.json"))
        except (httpx.HTTPError, ValueError) as e:
            logger.debug(f"config.json fetch failed for {repo_id}: {e}")
            return _MISSING
    if resp.status_code in (401, 403, 404):
        return None
    if resp.status_code >= 400:
        logger.debug(f"config.json fetch for {repo_id}: HTTP {resp.status_code}")
        return _MISSING
    try:
        config = resp.json()
    except ValueError:
        return None
    return parse_model_shape(config)


async def fetch_family_shapes(
    models: list[ModelInfo],
    family_ids: list[str],
    *,
    max_families: int | None = None,
    cache_file: Path | None = None,
) -> int:
    """Fetch shapes for the given families, cache them, attach; returns count.

    Each family tries up to MAX_SOURCES_PER_FAMILY repositories, most
    downloaded first, until one has a usable config.json. Network failures
    are not cached and leave the heuristic estimate in place. At most
    max_families (default MAX_FAMILIES_PER_RUN) families are looked up.
    """
    now = time.time()
    limit = MAX_FAMILIES_PER_RUN if max_families is None else max_families
    entries = _load_entries(cache_file)
    members: dict[str, list[ModelInfo]] = {}
    for model in models:
        members.setdefault(model.family_id, []).append(model)

    pending: list[list[str]] = []
    for family_id in dict.fromkeys(family_ids):
        group = members.get(family_id, [])
        if not group or all(m.shape is not None for m in group):
            continue
        sources = _family_sources(group)
        if any(
            _cached_shape(entries.get(s), now) not in (_MISSING, None) for s in sources
        ):
            continue
        sources = [s for s in sources if _cached_shape(entries.get(s), now) is _MISSING]
        if sources:
            pending.append(sources[:MAX_SOURCES_PER_FAMILY])
        if len(pending) >= limit:
            break

    if pending:
        semaphore = asyncio.Semaphore(_CONCURRENCY)
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers=_hf_headers(),
        ) as client:

            async def resolve(sources: list[str]) -> None:
                for source in sources:
                    result = await _fetch_config_shape(client, source, semaphore)
                    if result is _MISSING:
                        continue
                    entries[source] = {
                        "fetched_at": now,
                        "shape": result.to_dict() if result else None,
                    }
                    if result is not None:
                        return

            await asyncio.gather(*(resolve(sources) for sources in pending))
        try:
            _save_entries(entries, cache_file)
        except OSError as e:
            logger.debug(f"Could not save shape cache: {e}")

    return attach_cached_shapes(models, cache_file)
