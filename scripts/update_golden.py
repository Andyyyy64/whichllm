"""Maintain the golden ranking set in tests/golden/.

    uv run --locked python scripts/update_golden.py capture   # network
    uv run --locked python scripts/update_golden.py snapshot  # offline
    uv run --locked python scripts/update_golden.py report    # offline

``capture`` fetches the live model catalog and merged benchmark scores, freezes
them into ``tests/golden/catalog.json.gz``, then rewrites the snapshot.
``snapshot`` rewrites ``tests/golden/top_picks.md`` from the current code and
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
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.golden import harness  # noqa: E402
from whichllm.models.benchmark import fetch_benchmark_scores  # noqa: E402
from whichllm.models.fetcher import fetch_models, models_to_dicts  # noqa: E402


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


async def _fetch() -> tuple[list, dict[str, float]]:
    models = await fetch_models(include_vision=True)
    scores = await fetch_benchmark_scores()
    return models, scores


def capture() -> None:
    models, scores = asyncio.run(_fetch())
    payload = {
        "meta": {
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "whichllm_version": _source_version(),
            "git_commit": _git_commit(),
            "model_count": len(models),
            "benchmark_count": len(scores),
            "include_vision": True,
        },
        "models": models_to_dicts(models),
        "benchmark_scores": dict(sorted(scores.items())),
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # mtime=0 keeps the archive byte-identical for identical content.
    harness.CATALOG_PATH.write_bytes(gzip.compress(data, mtime=0))
    size_kb = harness.CATALOG_PATH.stat().st_size / 1024
    print(
        f"Captured {len(models)} models and {len(scores)} benchmark ids "
        f"({size_kb:.0f} KiB) to {harness.CATALOG_PATH.relative_to(REPO_ROOT)}"
    )


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
    parser.add_argument("command", choices=["capture", "snapshot", "report"])
    args = parser.parse_args()
    if args.command == "capture":
        capture()
        snapshot()
    elif args.command == "snapshot":
        snapshot()
    else:
        report()


if __name__ == "__main__":
    main()
