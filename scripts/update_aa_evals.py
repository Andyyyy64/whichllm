"""Refresh the committed Artificial Analysis per-benchmark snapshot.

The index builder reads AA's leaderboard live and falls back to
``src/whichllm/data/aa_evals.json`` when the page cannot be parsed. Run this
after checking that the live results look right:

    uv run python scripts/update_aa_evals.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import httpx

from whichllm.models.benchmark_sources.aa_evals import (
    SNAPSHOT_PATH,
    fetch_aa_eval_scores,
)
from whichllm.models.benchmark_sources.aa_index import AA_LEADERBOARD_URL
from whichllm.models.task_scores import AA_TASK_EVALS


async def _fetch() -> dict[str, dict[str, float]]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        return await fetch_aa_eval_scores(client, AA_TASK_EVALS)


def main() -> int:
    models = asyncio.run(_fetch())
    data = {
        "source": AA_LEADERBOARD_URL,
        "read_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "note": "Open-weights models only; best result per benchmark across "
        "variants, as fractions solved.",
        "models": {
            name: {k: round(v, 4) for k, v in sorted(scores.items())}
            for name, scores in sorted(models.items())
        },
    }
    SNAPSHOT_PATH.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(models)} models to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
