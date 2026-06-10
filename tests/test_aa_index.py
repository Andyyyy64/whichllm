"""Tests for the Artificial Analysis Intelligence Index source.

These cover the Next.js App Router (RSC) scraper that replaced the old
``__NEXT_DATA__`` extraction, the variant-stripping name canonicalization,
and the merge-over-curated-fallback behaviour of ``fetch_aa_index_scores``.
All tests are offline — network is served from an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from whichllm.models.benchmark_sources.aa_index import (
    AA_LEADERBOARD_URL,
    _canonical_name,
    _decode_rsc_blob,
    _extract_aa_pairs_from_html,
    fetch_aa_index_scores,
    get_aa_curated_fallback,
)
from whichllm.models.benchmark_sources.types import ExtractionFailed


def _rsc_page(records: list[dict]) -> str:
    """Build a minimal HTML page that embeds ``records`` the way the live
    artificialanalysis.ai App Router page does: as a JSON-string-escaped
    fragment inside ``self.__next_f.push([n, "..."])``."""
    # The fragment is an arbitrary slice of the RSC stream; the scraper only
    # cares that it contains the "name"/"intelligenceIndex" key pairs.
    fragment = ",".join(
        '{"slug":"x","name":%s,"reasoningModel":false,'
        '"intelligenceIndex":%s,"codingIndex":1.0}'
        % (json.dumps(r["name"]), r["index"])
        for r in records
    )
    chunk = json.dumps("3:[" + fragment + "]\n")
    return (
        "<!DOCTYPE html><html><body>"
        "<script>self.__next_f.push([0])</script>"
        f"<script>self.__next_f.push([1,{chunk}])</script>"
        "</body></html>"
    )


def test_canonical_name_strips_variants_and_separators():
    assert _canonical_name("Qwen3 14B (Reasoning)") == "qwen3 14b"
    assert _canonical_name("Qwen3-14B") == "qwen3 14b"
    # Separators normalize to single spaces (the table side is canonicalized
    # the same way, so "GLM-5" and "GLM 5" still collide).
    assert _canonical_name("GLM-5 (Non-reasoning)") == "glm 5"
    assert _canonical_name("DeepSeek V4 Pro (Reasoning, Max Effort)") == (
        "deepseek v4 pro"
    )


def test_decode_rsc_blob_unescapes_chunks():
    page = _rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 33.0}])
    blob = _decode_rsc_blob(page)
    assert '"name":"Qwen3 14B (Reasoning)"' in blob
    assert '"intelligenceIndex":33.0' in blob


def test_extract_pairs_from_rsc_html():
    page = _rsc_page(
        [
            {"name": "Qwen3 14B (Reasoning)", "index": 33.0},
            {"name": "Qwen3 14B (Non-reasoning)", "index": 30.0},
            {"name": "GLM-5 (Reasoning)", "index": 50.0},
        ]
    )
    pairs = dict(_extract_aa_pairs_from_html(page))
    assert pairs["Qwen3 14B (Reasoning)"] == 33.0
    assert pairs["GLM-5 (Reasoning)"] == 50.0
    # The bounded regex must not leak one record's name into another's index.
    assert len(pairs) == 3


def test_extract_pairs_returns_empty_on_legacy_or_garbage_html():
    assert _extract_aa_pairs_from_html("<html>no rsc here</html>") == []


def _run_fetch(html: str) -> dict[str, float]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == AA_LEADERBOARD_URL
        return httpx.Response(200, text=html)

    async def go() -> dict[str, float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_aa_index_scores(client)

    return asyncio.run(go())


def test_fetch_maps_canonical_names_and_merges_over_fallback():
    # "Qwen3 14B (Reasoning)" canonicalizes onto the "Qwen3 14B" table entry
    # -> Qwen/Qwen3-14B, and a high live value must override the snapshot.
    page = _rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 55.0}])
    scores = _run_fetch(page)

    fallback = get_aa_curated_fallback()
    # Coverage never shrinks below the curated snapshot ...
    assert set(fallback).issubset(set(scores))
    # ... and the live number wins where it is higher.
    assert scores["Qwen/Qwen3-14B"] > fallback["Qwen/Qwen3-14B"]


def test_fetch_raises_when_no_records_found():
    with pytest.raises(ExtractionFailed):
        _run_fetch("<html><body>nothing to see</body></html>")
