from __future__ import annotations

import io

import httpx

from whichllm.models.http import get_with_retries

LEADERBOARD_PARQUET_URL = (
    "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    "/parquet/default/train/0.parquet"
)
LEADERBOARD_ROWS_URL = "https://datasets-server.huggingface.co/rows"
LEADERBOARD_DATASET = "open-llm-leaderboard/contents"

# --- Leaderboard normalization ---
# OLLB v2 averages range ~5 to ~52. The leaderboard is archived 2025-06 with
# the top slot held by Qwen2.5-32B (47.6 raw = 91.5 if uncapped); capping at
# 78 prevents an older generation with a strong-but-frozen OLLB score from
# dominating rankings that now have AA Index / LiveBench coverage too.
_LB_AVG_MAX = 52
_OLLB_MAX_NORMALIZED = 78.0


async def _fetch_leaderboard_parquet(client: httpx.AsyncClient) -> dict[str, float]:
    """Download Open LLM Leaderboard parquet (requires pyarrow)."""
    import pyarrow.parquet as pq

    resp = await get_with_retries(
        client, LEADERBOARD_PARQUET_URL, follow_redirects=True
    )
    resp.raise_for_status()
    table = pq.read_table(
        io.BytesIO(resp.content),
        columns=["fullname", "Average ⬆️"],
    )
    d = table.to_pydict()
    scores: dict[str, float] = {}
    for i in range(len(d["fullname"])):
        name = d["fullname"][i]
        avg = d["Average ⬆️"][i]
        if name and avg and avg > 0:
            scores[name] = _normalize_leaderboard_avg(avg)
    return scores


async def _fetch_leaderboard_api(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch Open LLM Leaderboard via rows API (no pyarrow needed)."""
    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = await get_with_retries(
            client,
            LEADERBOARD_ROWS_URL,
            params={
                "dataset": LEADERBOARD_DATASET,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "100",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            name = row.get("fullname")
            avg = row.get("Average ⬆️")
            if name and avg and avg > 0:
                scores[name] = _normalize_leaderboard_avg(avg)

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores


def _normalize_leaderboard_avg(avg: float) -> float:
    """Normalize Open LLM Leaderboard average to 0-_OLLB_MAX_NORMALIZED scale."""
    score = avg / _LB_AVG_MAX * _OLLB_MAX_NORMALIZED
    return max(0.0, min(_OLLB_MAX_NORMALIZED, round(score, 1)))


async def fetch_leaderboard_with_fallback(
    client: httpx.AsyncClient,
) -> dict[str, float]:
    """Prefer the parquet path (one request, full table) and fall back to the
    paginated rows API when pyarrow is unavailable."""
    try:
        return await _fetch_leaderboard_parquet(client)
    except ImportError:
        return await _fetch_leaderboard_api(client)
