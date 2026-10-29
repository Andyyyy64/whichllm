"""Vision-language model benchmark source.

Text leaderboards (LiveBench, the AA index) do not score VLMs, so a
``--profile vision`` ranking had only one model with a ``direct`` hit
(Qwen2-VL-7B, a two-generations-old model) and every current VLM fell
back to size/popularity heuristics — letting the 8B legacy model
outrank Qwen3-VL-32B even on an 80 GB H100.

There is no single stable machine-readable VLM leaderboard (the MMMU /
OpenCompass spaces change their JSON shape frequently), so this source
is a curated snapshot rather than a live scrape. Values are a blended
0-100 capability index drawn from MMMU-Pro, MMBench, and general
multimodal evaluations, kept in ``data/vision.json`` with the month they
were curated. They feed the vision task score
next to AA's MMMU-Pro results (see ``whichllm.models.task_scores``), mostly
for older VLMs AA no longer lists, and never the general score.
"""

from __future__ import annotations

import logging

import httpx

from whichllm.data import read_json

logger = logging.getLogger(__name__)

_SNAPSHOT = read_json("vision.json")

# Curated multimodal capability index (0-100). Anchored so the current
# Qwen3-VL / InternVL3 frontier sits in the mid-50s-to-60s and
# two-generations-old 7B-class VLMs sit in the low 30s, which restores the
# correct generational ordering.
VISION_SNAPSHOT: dict[str, float] = _SNAPSHOT["models"]
# The month the index was last curated.
VISION_READ_AT: str = _SNAPSHOT["read_at"]


async def fetch_vision_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Return curated VLM capability scores.

    No stable live source exists, so this returns the frozen snapshot.
    The ``client`` parameter mirrors the other source functions so the
    caller can treat all sources uniformly and a live scrape can be
    slotted in later without changing call sites.
    """
    return dict(VISION_SNAPSHOT)
