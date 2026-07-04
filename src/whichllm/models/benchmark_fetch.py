"""Fetch and merge benchmark scores from source adapters."""

from __future__ import annotations

import asyncio
import logging

import httpx

from whichllm.models.benchmark_lineage import _apply_lineage_recency_demotion
from whichllm.models.http import DEFAULT_ACCEPT_ENCODING
from whichllm.utils import _current_version

logger = logging.getLogger(__name__)


async def fetch_benchmark_scores() -> dict[str, float]:
    """Fetch and combine benchmark scores from multiple sources.

    Sources, merged in this order (later overwrites earlier on conflict):
      1. Open LLM Leaderboard v2 (archived 2025-06)
      2. Chatbot Arena ELO (frozen 2025-07-17)
      3. LiveBench (vendored snapshot)
      4. Aider polyglot (coding-specific)
      5. Artificial Analysis Intelligence Index
      6. Vision-language capability index

    Returns dict mapping model_id -> normalized score (0-100). All network
    sources are fetched concurrently; failures are logged and skipped.
    """
    from whichllm.models import benchmark_sources

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "User-Agent": f"whichllm/{_current_version()}",
        },
    ) as client:
        leaderboard_task = asyncio.create_task(
            benchmark_sources.fetch_leaderboard_with_fallback(client)
        )
        arena_task = asyncio.create_task(benchmark_sources.fetch_arena_scores(client))
        aa_task = asyncio.create_task(benchmark_sources.fetch_aa_index_scores(client))
        aider_task = asyncio.create_task(
            benchmark_sources.fetch_aider_polyglot_scores(client)
        )
        vision_task = asyncio.create_task(benchmark_sources.fetch_vision_scores(client))

        (
            lb_result,
            arena_result,
            aa_result,
            aider_result,
            vision_result,
        ) = await asyncio.gather(
            leaderboard_task,
            arena_task,
            aa_task,
            aider_task,
            vision_task,
            return_exceptions=True,
        )

    frozen: dict[str, float] = {}
    current: dict[str, float] = {}

    if isinstance(lb_result, BaseException):
        logger.warning(f"Leaderboard fetch failed: {lb_result}")
    else:
        frozen.update(lb_result)
        logger.debug(f"Leaderboard: {len(lb_result)} scores (frozen)")

    if isinstance(arena_result, BaseException):
        logger.warning(f"Arena fetch failed, using fallback: {arena_result}")
    else:
        for k, v in arena_result.items():
            if frozen.get(k, 0.0) < v:
                frozen[k] = v
        logger.debug(f"Arena: {len(arena_result)} scores (frozen)")

    livebench_result = benchmark_sources.get_livebench_data()
    for k, v in livebench_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"LiveBench: {len(livebench_result)} scores (current)")

    if isinstance(aa_result, BaseException):
        logger.warning(f"AA Index fetch failed, will use fallback: {aa_result}")
        aa_result = benchmark_sources.get_aa_curated_fallback()

    for k, v in aa_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"AA Index: {len(aa_result)} scores (current)")

    if isinstance(aider_result, BaseException):
        logger.warning(f"Aider fetch failed: {aider_result}")
    else:
        for k, v in aider_result.items():
            if current.get(k, 0.0) < v * 0.85:
                current[k] = v * 0.85
        logger.debug(f"Aider polyglot: {len(aider_result)} scores (current, 0.85x)")

    if isinstance(vision_result, BaseException):
        logger.warning(f"Vision fetch failed: {vision_result}")
    else:
        for k, v in vision_result.items():
            if current.get(k, 0.0) < v:
                current[k] = v
        logger.debug(f"Vision: {len(vision_result)} scores (current)")

    combined: dict[str, float] = dict(frozen)
    combined.update(current)
    combined = _apply_lineage_recency_demotion(combined, frozen, current)

    logger.debug(f"Combined: {len(combined)} benchmark scores")
    return combined
