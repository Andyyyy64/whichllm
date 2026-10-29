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

import whichllm.models.benchmark_sources as benchmark_sources
from whichllm.models.benchmark import fetch_benchmark_scores
from whichllm.models.benchmark_sources import aa_index
from whichllm.models.benchmark_sources.aa_index import (
    AA_INDEX_VERSION,
    AA_LEADERBOARD_URL,
    AA_NAME_TO_HF_IDS,
    _AA_CANON_TO_HF_IDS,
    _AA_INDEX_MIN,
    _canonical_name,
    _decode_rsc_blob,
    _extract_aa_pairs_from_html,
    _extract_aa_pairs,
    _normalize_aa_index,
    fetch_aa_index_scores,
    get_aa_curated_fallback,
)
from whichllm.models.benchmark_sources.types import ExtractionFailed


def _rsc_page(records: list[dict], version: str | None = AA_INDEX_VERSION) -> str:
    """Build a minimal HTML page that embeds ``records`` the way the live
    artificialanalysis.ai App Router page does: as a JSON-string-escaped
    fragment inside ``self.__next_f.push([n, "..."])``."""
    # The fragment is an arbitrary slice of the RSC stream; the scraper only
    # cares that it contains the "name"/"intelligenceIndex" key pairs and the
    # index version the page reports.
    fragment = (
        ""
        if version is None
        else ('{"name":"Artificial Analysis Intelligence Index v%s"},' % version)
    )
    fragment += ",".join(
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


@pytest.mark.parametrize("raw_score", [55.0, 10.1, 5.0, 0.0, _AA_INDEX_MIN])
def test_fetch_maps_canonical_names_and_merges_over_fallback(raw_score):
    # "Qwen3 14B (Reasoning)" canonicalizes onto the "Qwen3 14B" table entry
    # -> Qwen/Qwen3-14B. A mapped live value must override the snapshot.
    page = _rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": raw_score}])
    scores = _run_fetch(page)

    fallback = get_aa_curated_fallback()
    # Coverage never shrinks below the curated snapshot ...
    assert set(fallback).issubset(set(scores))
    # Live scores replace the snapshot even when lower or normalized to zero.
    assert scores["Qwen/Qwen3-14B"] == _normalize_aa_index(raw_score)
    assert {k: v for k, v in scores.items() if k != "Qwen/Qwen3-14B"} == {
        k: v for k, v in fallback.items() if k != "Qwen/Qwen3-14B"
    }


def test_fetch_raises_when_no_records_found():
    with pytest.raises(ExtractionFailed):
        _run_fetch(_rsc_page([]))


@pytest.mark.parametrize("version", ["5.0", None])
def test_fetch_uses_snapshot_for_other_or_unknown_index_version(version):
    # v5.0 values are on another scale (MiMo-V2.6-Pro reads ~29.9 there), so
    # they must not go through the v4.3.2 bounds.
    page = _rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 29.9}], version)
    assert _run_fetch(page) == get_aa_curated_fallback()


@pytest.mark.parametrize("score", [0.0, -19.4, 10.1])
def test_payload_preserves_finite_scores(score):
    assert _extract_aa_pairs({"name": "Qwen3 14B", "intelligenceIndex": score}) == [
        ("Qwen3 14B", score)
    ]


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf")])
def test_payload_rejects_missing_or_invalid_scores(score):
    assert _extract_aa_pairs({"name": "Qwen3 14B", "intelligenceIndex": score}) == []


def test_each_checkpoint_maps_to_its_own_aa_row():
    # V3-0324 and GLM-4.5-Air are separate models, not quantizations of their
    # siblings: parameters.py puts GLM-4.5 at 32B active against 12B for Air,
    # and livebench already scores each checkpoint separately. Sharing one AA
    # display name handed both checkpoints in each pair the same index.
    assert AA_NAME_TO_HF_IDS["DeepSeek V3"] == ["deepseek-ai/DeepSeek-V3"]
    assert AA_NAME_TO_HF_IDS["DeepSeek V3 0324"] == ["deepseek-ai/DeepSeek-V3-0324"]
    assert AA_NAME_TO_HF_IDS["GLM-4.5"] == ["zai-org/GLM-4.5"]
    assert AA_NAME_TO_HF_IDS["GLM-4.5-Air"] == ["zai-org/GLM-4.5-Air"]
    # AA has no row for Kimi-K2-Base or the 08-2024 Command R+ release, so
    # neither inherits its sibling's index.
    assert AA_NAME_TO_HF_IDS["Kimi K2"] == ["moonshotai/Kimi-K2-Instruct"]
    assert AA_NAME_TO_HF_IDS["Kimi K2 0905"] == ["moonshotai/Kimi-K2-Instruct-0905"]
    assert AA_NAME_TO_HF_IDS["Command R+"] == ["CohereForAI/c4ai-command-r-plus"]

    # The canonical map unions ids across display names that collapse onto one
    # key, so it is the invariant that actually has to hold.
    assert _AA_CANON_TO_HF_IDS["deepseek v3"] == ["deepseek-ai/DeepSeek-V3"]
    assert _AA_CANON_TO_HF_IDS["glm 4.5"] == ["zai-org/GLM-4.5"]
    assert _AA_CANON_TO_HF_IDS["kimi k2"] == ["moonshotai/Kimi-K2-Instruct"]
    assert _AA_CANON_TO_HF_IDS["kimi k2 0905"] == ["moonshotai/Kimi-K2-Instruct-0905"]
    assert _AA_CANON_TO_HF_IDS["command r+"] == ["CohereForAI/c4ai-command-r-plus"]


def test_live_scores_land_on_one_checkpoint_each():
    # A page listing both GLM-4.5 rows must give each HF id its own score.
    # While the ids shared a display name the Air row was dropped and both
    # inherited the base model's index.
    page = _rsc_page(
        [
            {"name": "GLM-4.5", "index": 19.5},
            {"name": "GLM-4.5-Air", "index": 30.0},
        ]
    )
    scores = _run_fetch(page)

    assert scores["zai-org/GLM-4.5"] != scores["zai-org/GLM-4.5-Air"]


@pytest.mark.parametrize(
    "name,model_id",
    [
        ("DeepSeek V3 (Dec '24)", "deepseek-ai/DeepSeek-V3"),
        ("DeepSeek V3 0324", "deepseek-ai/DeepSeek-V3-0324"),
        ("GLM-4.5 (Reasoning)", "zai-org/GLM-4.5"),
        ("GLM-4.5-Air (Reasoning)", "zai-org/GLM-4.5-Air"),
        ("Kimi K2", "moonshotai/Kimi-K2-Instruct"),
        ("Kimi K2 0905", "moonshotai/Kimi-K2-Instruct-0905"),
        ("Command-R+ (Apr '24)", "CohereForAI/c4ai-command-r-plus"),
    ],
)
@pytest.mark.parametrize("raw_score", [0.0, _AA_INDEX_MIN])
def test_checkpoint_specific_decrease_preserves_absent_siblings(
    name, model_id, raw_score
):
    scores = _run_fetch(_rsc_page([{"name": name, "index": raw_score}]))
    fallback = get_aa_curated_fallback()

    assert scores[model_id] == _normalize_aa_index(raw_score)
    assert {k: v for k, v in scores.items() if k != model_id} == {
        k: v for k, v in fallback.items() if k != model_id
    }


def test_live_normalization_anchors_on_reworked_scale():
    # Retuned bounds keep the calibration: the top open model on AA lands ~95
    # and an 8B-class model lands ~40, on AA's rescaled raw values (2026-10-27).
    assert _normalize_aa_index(46.3) == 95.0  # MiMo-V2.6-Pro, top open on AA
    assert _normalize_aa_index(6.0) == 40.0  # Qwen3-8B (AA estimate)
    # Values below the floor clamp at 0 (raw is always positive in practice).
    assert _normalize_aa_index(-100.0) == 0.0
    assert _normalize_aa_index(60.0) == 100.0


def test_curated_fallback_normalizes_refreshed_snapshot():
    # The snapshot holds refreshed raw AA values; get_aa_curated_fallback maps
    # them onto the 0-100 scale with the retuned bounds.
    fb = get_aa_curated_fallback()
    assert fb["deepseek-ai/DeepSeek-V4-Pro-0813"] == _normalize_aa_index(36.0)
    assert fb["Qwen/Qwen3-8B"] == 40.0
    assert fb["XiaomiMiMo/MiMo-V2.5-Pro"] == _normalize_aa_index(26.0)
    # Reworked scale ranks the strong 8B above the small/old peers.
    assert fb["Qwen/Qwen3-8B"] > fb["Qwen/Qwen3-0.6B"]
    assert all(0.0 < v <= 100.0 for v in fb.values())


# Normalized scores of the hand-estimated peers before the 2026-10-27 bound
# retune. Their raw values were re-derived from these, so retuning the bounds
# must not move any of them.
_PEER_NORMALIZED = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 44.6,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 30.9,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 17.2,
    "Qwen/Qwen3-4B-Thinking-2507": 21.8,
    "Qwen/Qwen3-4B-Instruct-2507": 35.5,
    "Qwen/Qwen3-4B": 30.9,
    "Qwen/Qwen3-1.7B": 17.2,
    "Qwen/Qwen3-0.6B": 8.1,
    "meta-llama/Llama-3.1-8B-Instruct": 21.8,
    "meta-llama/Meta-Llama-3-8B-Instruct": 17.2,
    "google/gemma-2-9b-it": 24.0,
    "microsoft/Phi-4-mini-instruct": 26.3,
    "mistralai/Mistral-7B-Instruct-v0.3": 17.2,
    "Qwen/Qwen2.5-7B-Instruct": 21.8,
    "Qwen/Qwen2.5-14B-Instruct": 30.9,
    "Qwen/Qwen2.5-32B-Instruct": 40.0,
    "Qwen/Qwen3-30B-A3B": 44.6,
    "meta-llama/Llama-3.3-70B-Instruct": 46.9,
    "google/gemma-3-27b-it": 46.9,
    "google/gemma-3-12b-it": 40.0,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 44.6,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 40.0,
    "stepfun-ai/Step-3.5-Flash": 58.4,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 53.7,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 46.9,
    "allenai/Olmo-3-7B-Instruct": 21.8,
    "allenai/Olmo-3-1025-7B": 21.8,
    "ibm-granite/granite-4.0-h-small": 40.0,
    "ibm-granite/granite-4.0-h-tiny": 21.8,
    "ibm-granite/granite-3.3-8b-instruct": 24.0,
    "ibm-granite/granite-3.3-2b-instruct": 10.3,
    "mistralai/Codestral-22B-v0.1": 35.5,
}


def test_bound_retune_preserves_peer_scores():
    fb = get_aa_curated_fallback()
    assert {k: fb[k] for k in _PEER_NORMALIZED} == _PEER_NORMALIZED


def test_separate_deepseek_v4_checkpoints():
    # AA lists dated DeepSeek V4 releases, and each row's weights link names a
    # different repo, so each checkpoint must keep its own score.
    page = _rsc_page(
        [
            {"name": "DeepSeek V4 Pro 0424 (Max)", "index": 30.4},
            {"name": "DeepSeek V4 Pro 0813 (Max)", "index": 36.0},
            {"name": "DeepSeek V4 Flash 0420 (High)", "index": 24.4},
            {"name": "DeepSeek V4 Flash 0731 (Max)", "index": 34.3},
        ]
    )
    scores = _run_fetch(page)

    assert scores["deepseek-ai/DeepSeek-V4-Pro"] == _normalize_aa_index(30.4)
    assert scores["deepseek-ai/DeepSeek-V4-Pro-0813"] == _normalize_aa_index(36.0)
    assert scores["deepseek-ai/DeepSeek-V4-Flash"] == _normalize_aa_index(24.4)
    assert scores["deepseek-ai/DeepSeek-V4-Flash-0731"] == _normalize_aa_index(34.3)


@pytest.mark.parametrize(
    "model_id,raw",
    [
        ("moonshotai/Kimi-K2-Instruct", 12.7),
        ("moonshotai/Kimi-K2-Instruct-0905", 15.3),
        ("moonshotai/Kimi-K2-Thinking", 22.0),
    ],
)
def test_deprecated_rows_fall_back_to_current_scale(model_id, raw):
    # AA's leaderboard page carries no index for deprecated releases, so even a
    # successful live fetch leaves these to the snapshot. The snapshot must hold
    # the value AA shows today (read 2026-10-27), not the pre-rescale one.
    scores = _run_fetch(_rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 6.7}]))

    assert scores[model_id] == _normalize_aa_index(raw)


def test_other_index_version_uses_snapshot_end_to_end(monkeypatch):
    # A v5.0 leaderboard goes through the real fetcher and the aggregation
    # path; the model must end up with its v4.3.2 snapshot score, not 29.9
    # normalized through the v4.3.2 bounds.
    page = _rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 29.9}], "5.0")

    async def leaderboard(client, url):
        return httpx.Response(200, text=page, request=httpx.Request("GET", url))

    async def empty_source(client):
        return {}

    monkeypatch.setattr(aa_index, "get_with_retries", leaderboard)
    for name in (
        "fetch_leaderboard_with_fallback",
        "fetch_arena_scores",
        "fetch_vision_scores",
    ):
        monkeypatch.setattr(benchmark_sources, name, empty_source)
    monkeypatch.setattr(benchmark_sources, "get_livebench_data", lambda: {})

    scores = asyncio.run(fetch_benchmark_scores())
    assert scores["Qwen/Qwen3-14B"] == get_aa_curated_fallback()["Qwen/Qwen3-14B"]
