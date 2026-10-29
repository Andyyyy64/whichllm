"""Build the whichllm index and decide whether it is fit to publish."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from whichllm.models.benchmark_fetch import fetch_benchmark_data
from whichllm.models.fetcher import ModelFetchReport, fetch_models, models_to_dicts
from whichllm.models.grouper import group_models
from whichllm.models.hf import _FRONTIER_MODEL_IDS, fetch_model_published_at
from whichllm.models.index_format import INDEX_SCHEMA_VERSION, structure_problems
from whichllm.models.shape_fetch import fetch_family_shapes

# Publishing gates. A build below these is treated as a broken fetch, not as
# the ecosystem shrinking overnight.
MIN_MODELS = 500
MIN_BENCHMARKS = 2000
MIN_FRONTIER_COVERAGE = 0.8
MIN_SHAPE_COVERAGE = 0.6
MAX_DROP_FROM_PREVIOUS = 0.2
# Task scores cover far fewer models; a profile below this lost its source.
MIN_TASK_SCORES = {"coding": 20, "vision": 15}
MAX_TASK_DROP_FROM_PREVIOUS = 0.5


async def build_index(
    *,
    previous: dict | None = None,
    shape_cache: Path | None = None,
    whichllm_version: str = "unknown",
    git_commit: str = "unknown",
) -> tuple[dict, ModelFetchReport]:
    """Fetch models, benchmark scores, shapes and publish dates into an index.

    ``shape_cache`` lets successive builds reuse config.json lookups, and
    publish dates already in the ``previous`` index are reused, so a daily
    build only asks the Hub about repositories it has not seen before.
    """
    report = ModelFetchReport()
    models = await fetch_models(include_vision=True, report=report)
    benchmarks = await fetch_benchmark_data(m.id for m in models)
    scores = benchmarks.general

    group_models(models)  # assigns the family ids shapes are shared within
    families = list(dict.fromkeys(m.family_id for m in models))
    await fetch_family_shapes(
        models, families, max_families=len(families), cache_file=shape_cache
    )

    known_dates = {
        m["id"]: m["published_at"]
        for m in (previous or {}).get("models", [])
        if m.get("published_at")
    }
    for model in models:
        if not model.published_at and model.id in known_dates:
            model.published_at = known_dates[model.id]
    missing_dates = [m.id for m in models if not m.published_at]
    published = await fetch_model_published_at(missing_dates)
    for model in models:
        if not model.published_at and model.id in published:
            model.published_at = published[model.id]

    model_dicts = models_to_dicts(models)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": {"whichllm_version": whichllm_version, "git_commit": git_commit},
        "stats": {
            "models": len(models),
            "benchmarks": len(scores),
            "shapes": sum(1 for m in models if m.shape),
            "published_dates": sum(1 for m in models if m.published_at),
            "frontier_failed": len(report.failed),
            "task_scores": {p: len(s) for p, s in sorted(benchmarks.tasks.items())},
        },
        "models": model_dicts,
        "benchmark_scores": dict(sorted(scores.items())),
        "task_scores": {
            p: dict(sorted(s.items())) for p, s in sorted(benchmarks.tasks.items())
        },
    }
    return payload, report


def quality_problems(
    payload: dict, report: ModelFetchReport, previous: dict | None = None
) -> list[str]:
    """Reasons not to publish this build; empty when it is fit to publish."""
    problems = structure_problems(payload)
    if problems:
        return problems

    models = payload["models"]
    scores = payload["benchmark_scores"]
    stats = payload.get("stats", {})
    if not report.complete:
        problems.append(
            f"{len(report.failed)} catalog requests failed after retries: "
            + ", ".join(report.failed[:5])
        )
    if len(models) < MIN_MODELS:
        problems.append(f"only {len(models)} models (minimum {MIN_MODELS})")
    if len(scores) < MIN_BENCHMARKS:
        problems.append(f"only {len(scores)} benchmark ids (minimum {MIN_BENCHMARKS})")

    ids = {m["id"] for m in models}
    expected = [i for i in _FRONTIER_MODEL_IDS if i not in report.unavailable]
    covered = sum(1 for i in expected if i in ids)
    if expected and covered / len(expected) < MIN_FRONTIER_COVERAGE:
        problems.append(
            f"curated recent models: {covered} of {len(expected)} present "
            f"(minimum {MIN_FRONTIER_COVERAGE:.0%})"
        )
    shapes = stats.get("shapes", 0)
    if models and shapes / len(models) < MIN_SHAPE_COVERAGE:
        problems.append(
            f"shapes for {shapes} of {len(models)} models "
            f"(minimum {MIN_SHAPE_COVERAGE:.0%})"
        )

    tasks = payload.get("task_scores") or {}
    for profile, minimum in MIN_TASK_SCORES.items():
        count = len(tasks.get(profile) or {})
        if count < minimum:
            problems.append(f"only {count} {profile} task scores (minimum {minimum})")

    if previous:
        before_tasks = previous.get("task_scores") or {}
        for profile, before_scores in before_tasks.items():
            before = len(before_scores or {})
            now = len(tasks.get(profile) or {})
            if before and now < before * (1 - MAX_TASK_DROP_FROM_PREVIOUS):
                problems.append(
                    f"{profile} task scores dropped from {before} to {now} "
                    "since the previous index"
                )
        for key, now in (("models", len(models)), ("benchmark_scores", len(scores))):
            before = len(previous.get(key) or [])
            if before and now < before * (1 - MAX_DROP_FROM_PREVIOUS):
                problems.append(
                    f"{key} dropped from {before} to {now} since the previous index"
                )
    return problems
