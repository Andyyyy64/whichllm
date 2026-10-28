"""Build the whichllm index into a directory, refusing to publish a bad build.

    uv run --locked python scripts/build_index.py --out site/index/v1 \
        --previous https://andyyyy64.github.io/whichllm/index/v1/index.json.gz \
        --shape-cache .cache/shapes.json

Writes ``index.json.gz`` and ``index-meta.json``. Exits 1 without writing
when a quality gate fails (``--force`` writes anyway), so a scheduled job
keeps serving the last good index. Set ``HF_TOKEN`` for higher Hugging Face
rate limits.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from whichllm.models.index_build import build_index, quality_problems  # noqa: E402
from whichllm.models.index_format import (  # noqa: E402
    INDEX_FILENAME,
    META_FILENAME,
    decode_index,
    encode_index,
)


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


def _source_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _load_previous(location: str | None) -> dict | None:
    if not location:
        return None
    try:
        if location.startswith(("http://", "https://")):
            with urllib.request.urlopen(location, timeout=30) as response:
                raw = response.read()
        else:
            raw = Path(location).read_bytes()
        return decode_index(raw)
    except (OSError, ValueError, urllib.error.URLError) as e:
        print(f"Previous index unavailable ({e}); skipping drop checks")
        return None


def _summary(payload: dict, previous: dict | None) -> list[str]:
    stats = payload["stats"]
    lines = [
        f"models {stats['models']}, benchmark ids {stats['benchmarks']}, "
        f"shapes {stats['shapes']}, publish dates {stats['published_dates']}"
    ]
    if previous:
        before = {m["id"] for m in previous["models"]}
        now = {m["id"] for m in payload["models"]}
        added, removed = sorted(now - before), sorted(before - now)
        lines.append(
            f"vs previous ({previous['built_at']}): +{len(added)} -{len(removed)} "
            "models"
        )
        lines += [f"  + {model_id}" for model_id in added[:20]]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", help="path or URL of the last published index")
    parser.add_argument("--shape-cache", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="write even if quality gates fail"
    )
    args = parser.parse_args()

    previous = _load_previous(args.previous)
    payload, report = asyncio.run(
        build_index(
            previous=previous,
            shape_cache=args.shape_cache,
            whichllm_version=_source_version(),
            git_commit=_git_commit(),
        )
    )
    for line in _summary(payload, previous):
        print(line)

    problems = quality_problems(payload, report, previous)
    for problem in problems:
        print(f"quality gate: {problem}", file=sys.stderr)
    if problems and not args.force:
        print("Not writing the index.", file=sys.stderr)
        return 1

    raw = encode_index(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / INDEX_FILENAME).write_bytes(raw)
    meta = {
        "schema_version": payload["schema_version"],
        "built_at": payload["built_at"],
        "builder": payload["builder"],
        "stats": payload["stats"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    (args.out / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out / INDEX_FILENAME} ({len(raw) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
