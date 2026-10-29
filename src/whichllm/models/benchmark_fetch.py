"""Fetch and merge benchmark scores from source adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx

from whichllm.models.benchmark_lineage import _apply_lineage_recency_demotion
from whichllm.models.http import DEFAULT_ACCEPT_ENCODING
from whichllm.utils import _current_version

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """General scores, plus task scores for the coding, math and vision profiles."""

    general: dict[str, float]
    tasks: dict[str, dict[str, float]] = field(default_factory=dict)
    # One entry per source: what it scores, which version or snapshot was
    # read, and how many entries it gave.
    sources: list[dict] = field(default_factory=list)


async def fetch_benchmark_scores() -> dict[str, float]:
    """General-ability scores (see :func:`fetch_benchmark_data`)."""
    return (await fetch_benchmark_data()).general


async def fetch_benchmark_data(model_ids: Iterable[str] = ()) -> BenchmarkData:
    """Fetch and combine benchmark scores from multiple sources.

    General sources:
      1. Open LLM Leaderboard v2 (archived 2025-06)
      2. Chatbot Arena ELO (frozen 2025-07-17)
      3. LiveBench (vendored snapshot)
      4. Artificial Analysis Intelligence Index

    The frozen sources (1-2) only score models no current source covers.
    LiveBench and AA are both general-ability indexes on AA's scale and are
    averaged (see :func:`_merge_general_scores`).

    Task sources are AA's per-benchmark results and the curated vision index;
    :func:`whichllm.models.task_scores.build_task_scores` puts them on the
    general scale. ``model_ids`` are the repos AA display names may match,
    on top of the ids the general sources already know.

    Scores are keyed by model id and normalized to 0-100. All network sources
    are fetched concurrently; failures are logged and skipped.
    """
    from whichllm.models import benchmark_sources
    from whichllm.models.benchmark_sources import aa_evals
    from whichllm.models.task_scores import AA_TASK_EVALS

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "User-Agent": f"whichllm/{_current_version()}",
        },
    ) as client:
        (
            lb_result,
            arena_result,
            aa_result,
            aa_evals_result,
            vision_result,
        ) = await asyncio.gather(
            benchmark_sources.fetch_leaderboard_with_fallback(client),
            benchmark_sources.fetch_arena_scores(client),
            benchmark_sources.fetch_aa_index_scores(client),
            aa_evals.fetch_aa_eval_scores(client, AA_TASK_EVALS),
            benchmark_sources.fetch_vision_scores(client),
            return_exceptions=True,
        )

    frozen: dict[str, float] = {}
    current: dict[str, float] = {}

    sources: list[dict] = []
    if isinstance(lb_result, BaseException):
        logger.warning(f"Leaderboard fetch failed: {lb_result}")
    else:
        frozen.update(lb_result)
        logger.debug(f"Leaderboard: {len(lb_result)} scores (frozen)")
        sources.append(
            _source(
                "Open LLM Leaderboard v2",
                "general, frozen",
                "archived 2025-06",
                lb_result,
            )
        )

    if isinstance(arena_result, BaseException):
        logger.warning(f"Arena fetch failed, using fallback: {arena_result}")
    else:
        for k, v in arena_result.items():
            if frozen.get(k, 0.0) < v:
                frozen[k] = v
        logger.debug(f"Arena: {len(arena_result)} scores (frozen)")
        sources.append(
            _source("Chatbot Arena", "general, frozen", "2025-07-17", arena_result)
        )

    livebench_result = benchmark_sources.get_livebench_data()
    logger.debug(f"LiveBench: {len(livebench_result)} scores (current)")

    if isinstance(aa_result, BaseException):
        logger.warning(f"AA Index fetch failed, will use fallback: {aa_result}")
        aa_result = benchmark_sources.get_aa_curated_fallback()
    logger.debug(f"AA Index: {len(aa_result)} scores (current)")
    from whichllm.models.benchmark_sources.aa_index import (
        AA_INDEX_VERSION,
        AA_SNAPSHOT_READ_AT,
    )

    aa_live = aa_result != benchmark_sources.get_aa_curated_fallback()
    sources.append(
        _source(
            "Artificial Analysis Intelligence Index",
            "general",
            f"v{AA_INDEX_VERSION} "
            + ("live, over the " if aa_live else "")
            + f"{AA_SNAPSHOT_READ_AT} snapshot",
            aa_result,
            estimated=len(benchmark_sources.AA_PEER_IDS),
        )
    )
    sources.append(
        _source(
            "LiveBench",
            "general, fitted onto the AA scale",
            f"{benchmark_sources.LIVEBENCH_READ_AT} snapshot",
            livebench_result,
            estimated=len(benchmark_sources.LIVEBENCH_ESTIMATED_IDS),
        )
    )

    general_sources = (
        (livebench_result, benchmark_sources.LIVEBENCH_ESTIMATED_IDS),
        (aa_result, benchmark_sources.AA_PEER_IDS),
    )
    current.update(_merge_general_scores(*general_sources))

    combined: dict[str, float] = dict(frozen)
    combined.update(current)
    combined = _apply_lineage_recency_demotion(combined, frozen, current)
    logger.debug(f"Combined: {len(combined)} benchmark scores")

    if isinstance(aa_evals_result, BaseException):
        logger.warning(f"AA benchmarks fetch failed, using snapshot: {aa_evals_result}")
        aa_evals_result = aa_evals.load_aa_eval_snapshot()
        aa_evals_version = f"snapshot read {aa_evals.snapshot_read_at()}"
    else:
        aa_evals_version = "live"
    if isinstance(vision_result, BaseException):
        logger.warning(f"Vision fetch failed: {vision_result}")
        vision_result = {}
    sources.append(
        {
            "name": "Artificial Analysis benchmarks",
            "scores": "coding, math, vision; general for releases the above miss",
            "version": aa_evals_version,
            "entries": len(aa_evals_result),
            "benchmarks": sorted(AA_TASK_EVALS),
        }
    )
    sources.append(
        _source(
            "Curated vision index",
            "vision",
            benchmark_sources.VISION_READ_AT,
            vision_result,
        )
    )

    anchor = {
        model_id: current[model_id]
        for scores, estimated_ids in general_sources
        for model_id in scores
        if model_id not in estimated_ids and model_id in current
    }
    new_general, tasks = _task_scores(
        aa_evals_result,
        vision_result,
        anchor,
        combined,
        [*model_ids, *combined],
    )
    combined.update(new_general)
    return BenchmarkData(general=combined, tasks=tasks, sources=sources)


def _source(
    name: str, scores: str, version: str, entries: dict, estimated: int = 0
) -> dict:
    source = {
        "name": name,
        "scores": scores,
        "version": version,
        "entries": len(entries),
    }
    if estimated:
        source["hand_estimates"] = estimated
    return source


def _task_scores(
    aa_results: dict[str, dict[str, float]],
    vision: dict[str, float],
    anchor: dict[str, float],
    general: dict[str, float],
    model_ids: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """General scores for releases the general index lacks, and task scores."""
    from whichllm.models.benchmark_sources.aa_index import (
        AA_NAME_TO_HF_IDS,
        _canonical_name,
    )
    from whichllm.models.task_scores import (
        TASK_EVALS,
        build_task_scores,
        match_names_to_ids,
    )

    matches = match_names_to_ids(aa_results, model_ids)
    for display, ids in AA_NAME_TO_HF_IDS.items():
        name = _canonical_name(display)
        if name in aa_results:
            matches[name] = sorted({*matches.get(name, ()), *ids})

    # Repo ids differ in case between sources (gemma-4-31B-it, gemma-4-31b-it);
    # key everything by the general index's spelling so the two line up.
    spelling = {k.lower(): k for k in general}

    def key(model_id: str) -> str:
        return spelling.get(model_id.lower(), model_id)

    evals_by_id: dict[str, dict[str, float]] = {}
    # Every repo a display name matched shares that release's general score,
    # so a single-benchmark profile score is steadied the same way for all.
    release_general: dict[str, float] = dict(general)
    for name, ids in matches.items():
        ids = sorted({key(i) for i in ids})
        known = [general[i] for i in ids if i in general]
        for model_id in ids:
            results = evals_by_id.setdefault(model_id, {})
            for eval_id, value in aa_results[name].items():
                results[eval_id] = max(value, results.get(eval_id, value))
            if known and model_id not in release_general:
                release_general[model_id] = max(known)
    for model_id, value in vision.items():
        evals_by_id.setdefault(key(model_id), {})["vision-index"] = value

    fill = build_task_scores(evals_by_id, anchor, release_general, ["general"])
    new_general = {
        model_id: score
        for model_id, score in fill.get("general", {}).items()
        if model_id not in release_general
    }
    logger.debug(f"General scores from AA benchmarks: {len(new_general)} new models")
    tasks = build_task_scores(
        evals_by_id,
        anchor,
        {**release_general, **new_general},
        [p for p in TASK_EVALS if p != "general"],
    )
    for profile, scores in tasks.items():
        logger.debug(f"Task scores ({profile}): {len(scores)} models")
    return new_general, tasks


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
