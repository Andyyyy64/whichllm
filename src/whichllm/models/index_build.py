"""Build the whichllm index and decide whether it is fit to publish."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from whichllm.models.benchmark_fetch import fetch_benchmark_data
from whichllm.models.fetcher import ModelFetchReport, fetch_models, models_to_dicts
from whichllm.models.gguf_sizes import fill_gguf_sizes
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

# A release counts as recent and popular when it was published in the last
# RECENT_DAYS and has at least RECENT_MIN_DOWNLOADS downloads. The build lists
# those without a benchmark score, so a gap shows up within a day.
RECENT_DAYS = 45
RECENT_MIN_DOWNLOADS = 5_000


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
    build only asks the Hub about repositories it has not seen before. GGUF
    file sizes from the ``previous`` index are reused while a repository's
    commit is unchanged.
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
    # Last, so the per-build cap only spends what the steps above left of the
    # Hub's rate limit.
    sizes = await fill_gguf_sizes(models, previous)

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
            "gguf_repos": sum(1 for m in models if m.gguf_variants),
            "gguf_real_sizes": sum(
                1
                for m in models
                if m.gguf_variants and not any(v.is_estimated for v in m.gguf_variants)
            ),
            "gguf_sizes": {
                "reused": sizes.reused,
                "fetched": sizes.fetched,
                "failed": sizes.failed,
                "pending": sizes.pending,
            },
            "frontier_failed": len(report.failed),
            "task_scores": {p: len(s) for p, s in sorted(benchmarks.tasks.items())},
        },
        "models": model_dicts,
        "benchmark_scores": dict(sorted(scores.items())),
        "sources": benchmarks.sources,
        "coverage": recent_coverage(models, scores),
        "task_scores": {
            p: dict(sorted(s.items())) for p, s in sorted(benchmarks.tasks.items())
        },
    }
    return payload, report


def recent_coverage(
    models: list, scores: dict[str, float], now: datetime | None = None
) -> dict:
    """How many recent popular releases have a general benchmark score.

    An original release without a score is listed by id. A derivative
    (finetune, abliteration, merge) without one is listed by its base, and
    only when nothing further up its lineage has a score either: then the gap
    is the base's, not the derivative's.
    """
    from whichllm.engine.ranking_filters import _is_auxiliary_model
    from whichllm.models.benchmark import (
        build_line_bucket_index,
        build_score_index,
        build_score_name_index,
        lookup_benchmark_evidence,
    )
    from whichllm.models.roles import find_base_checkpoints

    now = now or datetime.now(timezone.utc)
    base_checkpoints = find_base_checkpoints(models)
    by_id = {model.id: model for model in models}
    ci_index, line_index = build_score_index(scores)
    bucket_index = build_line_bucket_index(scores)
    name_index = build_score_name_index(scores)

    def scored(model_id: str, base_model: str | None = None) -> bool:
        evidence = lookup_benchmark_evidence(
            model_id,
            base_model,
            scores,
            ci_index=ci_index,
            line_index=line_index,
            line_bucket_index=bucket_index,
            name_index=name_index,
        )
        return evidence.source in ("direct", "variant", "base_model")

    def lineage(model) -> list[str]:
        chain: list[str] = []
        while model is not None and model.base_model:
            if model.base_model == model.id or model.base_model in chain:
                break
            chain.append(model.base_model)
            model = by_id.get(model.base_model)
        return chain

    recent, unscored, derivatives = 0, [], 0
    unscored_bases: list[str] = []
    for model in sorted(models, key=lambda m: -m.downloads):
        if model.downloads < RECENT_MIN_DOWNLOADS or not model.published_at:
            continue
        if model.base_model_relation == "quantized" or model.id in base_checkpoints:
            continue
        if _is_auxiliary_model(model):
            continue
        try:
            published = datetime.fromisoformat(
                model.published_at.replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if (now - published).days > RECENT_DAYS:
            continue
        recent += 1
        if scored(model.id, model.base_model):
            continue
        chain = lineage(model)
        if not chain:
            unscored.append(model.id)
        elif not any(scored(base) for base in chain[1:]):
            derivatives += 1
            if chain[0] not in unscored_bases:
                unscored_bases.append(chain[0])
    return {
        "recent_days": RECENT_DAYS,
        "recent_min_downloads": RECENT_MIN_DOWNLOADS,
        "recent_models": recent,
        "recent_unscored": unscored[:30],
        "recent_unscored_count": len(unscored),
        "recent_unscored_bases": unscored_bases[:30],
        "recent_unscored_derivatives": derivatives,
    }


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
