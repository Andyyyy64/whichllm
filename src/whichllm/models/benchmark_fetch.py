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

    Sources:
      1. Open LLM Leaderboard v2 (archived 2025-06)
      2. Chatbot Arena ELO (frozen 2025-07-17)
      3. LiveBench (vendored snapshot)
      4. Artificial Analysis Intelligence Index
      5. Aider polyglot (coding-specific)
      6. Vision-language capability index

    The frozen sources (1-2) only score models no current source covers.
    LiveBench and AA are both general-ability indexes on AA's scale and are
    averaged (see :func:`_merge_general_scores`); Aider and vision scores
    can raise a model's score as specialist evidence.

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
    logger.debug(f"LiveBench: {len(livebench_result)} scores (current)")

    if isinstance(aa_result, BaseException):
        logger.warning(f"AA Index fetch failed, will use fallback: {aa_result}")
        aa_result = benchmark_sources.get_aa_curated_fallback()
    logger.debug(f"AA Index: {len(aa_result)} scores (current)")

    current.update(
        _merge_general_scores(
            (livebench_result, benchmark_sources.LIVEBENCH_ESTIMATED_IDS),
            (aa_result, benchmark_sources.AA_PEER_IDS),
        )
    )

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


def _merge_general_scores(
    *sources: tuple[dict[str, float], frozenset[str]],
) -> dict[str, float]:
    """Combine general-ability indexes that share one scale.

    Each source is ``(scores, estimated_ids)``. A model's score is the mean
    of the values that measured it; hand estimates count only when no source
    measured the model. Taking the highest value instead favoured models
    that appear in more sources, since each index is noisy.
    """
    merged: dict[str, float] = {}
    for model_id in set().union(*(scores for scores, _ in sources)):
        measured: list[float] = []
        estimated: list[float] = []
        for scores, estimated_ids in sources:
            if model_id in scores:
                bucket = estimated if model_id in estimated_ids else measured
                bucket.append(scores[model_id])
        values = measured or estimated
        merged[model_id] = round(sum(values) / len(values), 1)
    return merged
