"""Maintain the golden ranking set in tests/golden/.

    uv run --locked python scripts/update_golden.py capture   # network
    uv run --locked python scripts/update_golden.py shapes    # network
    uv run --locked python scripts/update_golden.py benchmarks  # network
    uv run --locked python scripts/update_golden.py snapshot  # offline
    uv run --locked python scripts/update_golden.py report    # offline

``capture`` fetches the live model catalog, merged benchmark scores and every
family's config.json shape, freezes them into ``tests/golden/catalog.json.gz``,
then rewrites the snapshot. ``shapes`` only refreshes the shapes of the
existing frozen catalog, and ``benchmarks`` only its merged benchmark
scores (after a benchmark source changes). ``snapshot`` rewrites ``tests/golden/top_picks.md`` from the current code and
frozen catalog. ``report`` prints the top picks and every constraint result
without writing anything. Review the snapshot diff before committing.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.golden import harness  # noqa: E402
from whichllm.models import shape_fetch  # noqa: E402
from whichllm.models.benchmark import fetch_benchmark_data  # noqa: E402
from whichllm.models.fetcher import (  # noqa: E402
    dicts_to_models,
    fetch_models,
    models_to_dicts,
)
from whichllm.models.grouper import group_models  # noqa: E402


def _source_version() -> str:
    # Installed metadata can lag behind an editable checkout; pyproject is
    # the version of the code that produced the capture.
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def _fetch() -> tuple[list, dict[str, float], dict[str, dict[str, float]]]:
    models = await fetch_models(include_vision=True)
    data = await fetch_benchmark_data(m.id for m in models)
    return models, data.general, data.tasks


def _add_shapes(model_dicts: list[dict]) -> int:
    """Read config.json for every family and store each model's shape.

    The golden catalog carries shapes for all families, the state an index
    build provides, while a plain run only looks up the top families.
    """
    models = dicts_to_models(model_dicts)
    group_models(models)  # assigns the family ids shapes are shared within
    with tempfile.TemporaryDirectory() as tmp:
        shape_fetch.CACHE_DIR = Path(tmp)
        shape_fetch.SHAPE_CACHE_FILE = Path(tmp) / "shapes.json"
        shape_fetch.MAX_FAMILIES_PER_RUN = len(models)
        families = list(dict.fromkeys(m.family_id for m in models))
        asyncio.run(shape_fetch.fetch_family_shapes(models, families))
    shapes = {m.id: m.shape.to_dict() for m in models if m.shape}
    for data in model_dicts:
        data["shape"] = shapes.get(data["id"])
    return len(shapes)


def _write_catalog(payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # mtime=0 keeps the archive byte-identical for identical content.
    harness.CATALOG_PATH.write_bytes(gzip.compress(data, mtime=0))


def capture() -> None:
    models, scores, tasks = asyncio.run(_fetch())
    model_dicts = models_to_dicts(models)
    shaped = _add_shapes(model_dicts)
    payload = {
        "meta": {
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "whichllm_version": _source_version(),
            "git_commit": _git_commit(),
            "model_count": len(models),
            "benchmark_count": len(scores),
            "include_vision": True,
            "shape_count": shaped,
        },
        "models": model_dicts,
        "benchmark_scores": dict(sorted(scores.items())),
        "task_scores": _sorted_tasks(tasks),
    }
    _write_catalog(payload)
    size_kb = harness.CATALOG_PATH.stat().st_size / 1024
    print(
        f"Captured {len(models)} models ({shaped} with shapes) and {len(scores)} "
        f"benchmark ids ({size_kb:.0f} KiB) to "
        f"{harness.CATALOG_PATH.relative_to(REPO_ROOT)}"
    )


def shapes() -> None:
    """Add config.json shapes to the existing frozen catalog."""
    payload = harness.load_catalog()
    shaped = _add_shapes(payload["models"])
    payload["meta"]["shape_count"] = shaped
    _write_catalog(payload)
    print(f"Added shapes for {shaped} of {len(payload['models'])} catalog models")


def _sorted_tasks(tasks: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {p: dict(sorted(s.items())) for p, s in sorted(tasks.items())}


def benchmarks() -> None:
    """Re-fetch the merged benchmark and task scores, keeping the frozen models."""
    payload = harness.load_catalog()
    data = asyncio.run(fetch_benchmark_data(m["id"] for m in payload["models"]))
    scores = data.general
    payload["benchmark_scores"] = dict(sorted(scores.items()))
    payload["task_scores"] = _sorted_tasks(data.tasks)
    payload["meta"]["benchmark_count"] = len(scores)
    payload["meta"]["task_score_counts"] = {
        p: len(s) for p, s in sorted(data.tasks.items())
    }
    payload["meta"]["benchmarks_refreshed_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )
    _write_catalog(payload)
    print(f"Refreshed {len(scores)} benchmark ids")


def snapshot() -> None:
    text = harness.render_top_picks()
    previous = (
        harness.TOP_PICKS_PATH.read_text(encoding="utf-8")
        if harness.TOP_PICKS_PATH.exists()
        else ""
    )
    harness.TOP_PICKS_PATH.write_text(text, encoding="utf-8")
    state = "unchanged" if text == previous else "updated"
    print(f"{harness.TOP_PICKS_PATH.relative_to(REPO_ROOT)} {state}")


def report() -> None:
    print(harness.render_top_picks())
    failures = 0
    for case in harness.golden_cases():
        problem = harness.evaluate(case)
        expected = case.xfail_reason
        if problem is None:
            status = "XPASS (remove xfail)" if expected else "pass"
        else:
            status = "xfail" if expected else "FAIL"
        failures += status in {"FAIL", "XPASS (remove xfail)"}
        print(f"{status:22} {case.test_id}")
        if problem and not expected:
            print(f"{'':22} {problem}")
    print(f"\n{failures} unexpected result(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", choices=["capture", "shapes", "benchmarks", "snapshot", "report"]
    )
    args = parser.parse_args()
    if args.command == "capture":
        capture()
        snapshot()
    elif args.command == "shapes":
        shapes()
        harness.load_catalog.cache_clear()
        harness.catalog_models.cache_clear()
        snapshot()
    elif args.command == "benchmarks":
        benchmarks()
        harness.load_catalog.cache_clear()
        snapshot()
    elif args.command == "snapshot":
        snapshot()
    else:
        report()


if __name__ == "__main__":
    main()
