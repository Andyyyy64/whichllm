"""The whichllm index: one prebuilt file with everything ranking needs.

A scheduled CI job builds the index from Hugging Face, the benchmark sources
and each family's config.json, and publishes it on GitHub Pages. The client
downloads one file instead of making hundreds of requests and scraping
leaderboards on every run.

Layout (``schema_version`` 1)::

    {
      "schema_version": 1,
      "built_at": "2026-10-29T03:00:00Z",
      "builder": {"whichllm_version": "...", "git_commit": "..."},
      "stats": {"models": 1129, "benchmarks": 4715, "shapes": 963, ...},
      "models": [... models_to_dicts(...) ...],
      "benchmark_scores": {"org/model": 0-100, ...},
      "task_scores": {"coding": {"org/model": 0-100, ...}, ...}
    }

task_scores is optional: an index built before it existed, or a mirror
that drops it, ranks every profile on benchmark_scores.
"""

from __future__ import annotations

import gzip
import json

INDEX_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.json.gz"
META_FILENAME = "index-meta.json"
DEFAULT_INDEX_URL = f"https://andyyyy64.github.io/whichllm/index/v1/{INDEX_FILENAME}"


def encode_index(payload: dict) -> bytes:
    """Gzip-encoded JSON; identical payloads give identical bytes."""
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return gzip.compress(data.encode("utf-8"), mtime=0)


def decode_index(raw: bytes) -> dict:
    """Decode an index file; raises ValueError if it is not a valid index."""
    try:
        payload = json.loads(gzip.decompress(raw).decode("utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"index is not gzip JSON: {e}") from e
    problems = structure_problems(payload)
    if problems:
        raise ValueError("; ".join(problems))
    return payload


def structure_problems(payload: object) -> list[str]:
    """Shape checks every reader relies on."""
    if not isinstance(payload, dict):
        return ["index is not a JSON object"]
    problems = []
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        problems.append(
            f"schema_version {payload.get('schema_version')!r} is not "
            f"{INDEX_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("built_at"), str):
        problems.append("built_at is missing")
    models = payload.get("models")
    if not isinstance(models, list) or not all(
        isinstance(m, dict) and isinstance(m.get("id"), str) for m in models
    ):
        problems.append("models must be a list of objects with an id")
    scores = payload.get("benchmark_scores")
    if not isinstance(scores, dict) or not all(
        isinstance(v, (int, float)) for v in scores.values()
    ):
        problems.append("benchmark_scores must map ids to numbers")
    tasks = payload.get("task_scores", {})
    if not isinstance(tasks, dict) or not all(
        isinstance(scores, dict)
        and all(isinstance(v, (int, float)) for v in scores.values())
        for scores in tasks.values()
    ):
        problems.append("task_scores must map profiles to id-number maps")
    return problems
