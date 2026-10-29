"""Per-benchmark results from the Artificial Analysis leaderboard.

Besides the Intelligence Index, the leaderboard page lists the raw result of
each benchmark AA runs for every model variant: the fraction of SciCode,
Terminal-Bench, LiveCodeBench, AIME or MMMU-Pro it solved. Those fractions do
not depend on the index version, so task profiles can rank on them while the
general score stays on the pinned index snapshot.

Only open-weights models are kept, and only the benchmarks a task profile
uses (:data:`whichllm.models.task_scores.TASK_EVALS`). Variants of one model
("(Reasoning)", "(high)", ...) collapse to the best result per benchmark, the
same rule the index uses.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import httpx

from whichllm.models.benchmark_sources.aa_index import (
    AA_LEADERBOARD_URL,
    _canonical_name,
    _decode_rsc_blob,
)
from whichllm.models.http import get_with_retries

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "aa_evals.json"

_MODELS_KEY = '"initialModels":'


def _records(html: str) -> list[dict]:
    blob = _decode_rsc_blob(html)
    start = blob.find(_MODELS_KEY)
    if start < 0:
        return []
    try:
        records, _ = json.JSONDecoder().raw_decode(blob, start + len(_MODELS_KEY))
    except json.JSONDecodeError:
        return []
    return (
        [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
    )


def parse_aa_evals(html: str, evals: frozenset[str]) -> dict[str, dict[str, float]]:
    """``{canonical model name: {benchmark: best fraction}}`` from the page."""
    out: dict[str, dict[str, float]] = {}
    for record in _records(html):
        name = record.get("name")
        scores = record.get("scores")
        if not record.get("isOpenWeights") or not isinstance(name, str):
            continue
        if not isinstance(scores, dict):
            continue
        model = out.setdefault(_canonical_name(name), {})
        for key, value in scores.items():
            if key not in evals or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            if 0.0 <= value <= 1.0 and value > model.get(key, -1.0):
                model[key] = float(value)
    return {name: scores for name, scores in out.items() if scores}


async def fetch_aa_eval_scores(
    client: httpx.AsyncClient, evals: frozenset[str]
) -> dict[str, dict[str, float]]:
    """Live per-benchmark results. Raises when the page yields none."""
    resp = await get_with_retries(client, AA_LEADERBOARD_URL)
    resp.raise_for_status()
    results = parse_aa_evals(resp.text, evals)
    if not results:
        raise ValueError("AA leaderboard: no per-benchmark results found")
    logger.debug(f"AA benchmarks: {len(results)} open-weights models")
    return results


def load_aa_eval_snapshot() -> dict[str, dict[str, float]]:
    """The committed snapshot, used when the live page cannot be read."""
    try:
        data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.debug(f"AA benchmark snapshot unreadable: {e}")
        return {}
    models = data.get("models")
    return models if isinstance(models, dict) else {}
