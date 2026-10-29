"""Artificial Analysis Intelligence Index source.

AA publishes a model-quality index (https://artificialanalysis.ai/) that
covers post-2025-08 frontier releases (DeepSeek V4, GLM-5, Kimi K2.6,
MiMo V2.5, Qwen3.6, etc.) that whichllm's primary sources have stopped
tracking. The index is exposed via the JSON payload embedded in
``__NEXT_DATA__`` on the leaderboard page.

The fetcher is defensive: any failure (network, schema drift, parsing) is
caught and an empty dict is returned so it never blocks the main benchmark
pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import re

import httpx

from whichllm.data import read_json
from whichllm.models.benchmark_sources.constants import _NEXT_DATA_RE
from whichllm.models.benchmark_sources.types import ExtractionFailed
from whichllm.models.benchmark_sources.utils import _walk
from whichllm.models.http import get_with_retries

logger = logging.getLogger(__name__)

# AA display name -> Hugging Face ids, used to map AA-reported labels back to
# repos. Only the most common fully-open-weights releases need entries; anything
# else is dropped. data/aa_names.json also records why some names map the way
# they do.
AA_NAME_TO_HF_IDS: dict[str, list[str]] = read_json("aa_names.json")["names"]

# Bounds used to normalize raw AA index values onto the 0-100 scale the rest
# of the ranking system uses. AA rescaled their Intelligence Index again after
# the 2026-06-29 refresh (most open-weights values dropped by about a third;
# the values read on 2026-09-14 already match the 2026-10-27 ones). The window is
# anchored by the same two-point fit as before: top open frontier on AA
# (MiMo-V2.6-Pro = 46.3 on AA_LEADERBOARD_URL, not mapped here) → 95 normalized,
# and 8B-class (Qwen3-8B = 6.0, an AA estimate, /models/qwen3-8b-instruct) → 40
# normalized, both read 2026-10-27. This keeps a strong 8B model
# competitive with frozen-OLLB 7B scores while leaving headroom for
# frontier-tier models, including open releases not mapped yet. The fit puts
# the floor below zero; that is fine, live AA values are always positive and
# clamp at 0.
_AA_INDEX_MIN = -23.3
_AA_INDEX_MAX = 50.0

_SNAPSHOT = read_json("aa_index.json")

# The AA index version the bounds and snapshot were read on. Raw values from
# another version are on a different scale, so fetch_aa_index_scores() returns
# the snapshot when the leaderboard reports any other (or no) version.
AA_INDEX_VERSION: str = _SNAPSHOT["index_version"]
# The day the curated snapshot was read.
AA_SNAPSHOT_READ_AT: str = _SNAPSHOT["read_at"]
_AA_VERSION_RE = re.compile(r"Artificial Analysis Intelligence Index v(\d+(?:\.\d+)*)")

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# Snapshot of the AA Intelligence Index (open-weights only), data/aa_index.json.
# An entry of kind "live" was read from https://artificialanalysis.ai/models/<slug>
# and is the highest variant of the release, as fetch_aa_index_scores() takes
# it; "estimated" marks values AA flags as estimated rather than measured. Used
# as a fallback when the live HTML scrape returns no results (e.g. because the
# Next.js payload format changes again) or reports another index version, and
# for mapped models whose AA release is deprecated: the leaderboard page no
# longer carries an index for those (Kimi K2, GLM-5, ...), so even a successful
# live fetch falls back to these values. Entries are raw AA index values,
# normalized through _normalize_aa_index() in get_aa_curated_fallback().
#
# Entries of kind "peer" are models AA does not track; their raw value is set
# so that, under the current bounds, they reproduce their previous normalized
# score (hand-estimated LB-equivalents). Several peers fall below the index
# floor and read as negative raw; _normalize_aa_index() clamps them at 0.
AA_INDEX_SNAPSHOT: dict[str, float] = {
    model_id: entry["raw"] for model_id, entry in _SNAPSHOT["models"].items()
}

# The snapshot entries marked "peer": hand estimates for models AA does not
# track. The merge uses them only for models no source measured.
AA_PEER_IDS: frozenset[str] = frozenset(
    model_id
    for model_id, entry in _SNAPSHOT["models"].items()
    if entry["kind"] == "peer"
)


def _normalize_aa_index(index: float) -> float:
    """Normalize a raw AA index value onto the 0-100 scale."""
    if not isinstance(index, (int, float)):
        return 0.0
    span = _AA_INDEX_MAX - _AA_INDEX_MIN
    normalized = (index - _AA_INDEX_MIN) / span * 100.0
    return max(0.0, min(100.0, round(normalized, 1)))


# --- Next.js App Router (RSC) scraping -------------------------------------
#
# artificialanalysis.ai migrated off the classic ``__NEXT_DATA__`` blob to the
# App Router streaming format: model data arrives in ``self.__next_f.push([n,
# "…"])`` calls whose second element is a JSON-string-escaped fragment of the
# RSC payload. We concatenate + unescape those chunks, then pull every
# ``{"name": …, …, "intelligenceIndex": …}`` record out with a bounded regex
# (the payload is a flat RSC stream, not a single parseable JSON document).

_RSC_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[\d+,(?P<s>"(?:[^"\\]|\\.)*")\]\)')
# A model record: a "name" string followed, within the SAME object, by its
# "intelligenceIndex". The middle group forbids another '"name":"' so the
# match cannot leak across into a neighbouring record that lacks an index.
_AA_RECORD_RE = re.compile(
    r'"name":"(?P<name>(?:[^"\\]|\\.)*)"'
    r'(?:(?!"name":").)*?'
    r'"intelligenceIndex":(?P<idx>-?\d+(?:\.\d+)?)',
    re.DOTALL,
)
# Variant qualifiers AA appends that don't change the underlying HF weights:
# "(Reasoning)", "(Non-reasoning)", "(high)", "(Reasoning, Max Effort)", etc.
_PAREN_RE = re.compile(r"\([^)]*\)")


def _canonical_name(name: str) -> str:
    """Normalize an AA display name for fuzzy matching against the HF map.

    Drops parenthetical variant qualifiers and collapses separator/case noise
    so ``"Qwen3 14B (Reasoning)"`` and ``"Qwen3-14B"`` both canonicalize to
    ``"qwen3 14b"``.
    """
    name = _PAREN_RE.sub("", name)
    name = name.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


# Canonical-name -> HF ids, derived once from AA_NAME_TO_HF_IDS. Several display
# names can collapse to one canonical key; we union their HF ids.
_AA_CANON_TO_HF_IDS: dict[str, list[str]] = {}
for _disp, _ids in AA_NAME_TO_HF_IDS.items():
    _AA_CANON_TO_HF_IDS.setdefault(_canonical_name(_disp), []).extend(_ids)


def _decode_rsc_blob(html: str) -> str:
    """Concatenate and unescape the App Router RSC chunks into one string."""
    parts: list[str] = []
    for m in _RSC_CHUNK_RE.finditer(html):
        try:
            parts.append(json.loads(m.group("s")))
        except (ValueError, json.JSONDecodeError):
            continue
    return "".join(parts)


def _extract_aa_pairs_from_html(html: str) -> list[tuple[str, float]]:
    """Extract (name, intelligence_index) pairs from the RSC stream."""
    blob = _decode_rsc_blob(html)
    if not blob:
        return []
    pairs: list[tuple[str, float]] = []
    for m in _AA_RECORD_RE.finditer(blob):
        try:
            name = json.loads('"' + m.group("name") + '"').strip()
            score = float(m.group("idx"))
        except (ValueError, json.JSONDecodeError):
            continue
        if name and math.isfinite(score):
            pairs.append((name, score))
    return pairs


def _extract_aa_pairs(payload: dict) -> list[tuple[str, float]]:
    """Walk the Next.js payload looking for {name, intelligenceIndex}-shaped
    objects regardless of where they are nested."""
    pairs: list[tuple[str, float]] = []
    for node in _walk(payload):
        # Look for the most common shapes AA has used in past iterations.
        name = None
        score = None
        for name_key in ("model_name", "modelName", "name", "displayName"):
            v = node.get(name_key)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break
        for score_key in (
            "intelligence_index",
            "intelligenceIndex",
            "aa_index",
            "aaIndex",
            "score",
        ):
            v = node.get(score_key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                score = float(v)
                break
        if name and score is not None and math.isfinite(score):
            pairs.append((name, score))
    return pairs


async def fetch_aa_index_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch Artificial Analysis Intelligence Index scores.

    Returns ``{hf_id: normalized_score_0_100}`` for every model AA reports
    that we can map back to a HuggingFace repo via :data:`AA_NAME_TO_HF_IDS`.

    Returns the curated snapshot instead when the leaderboard reports an
    index version other than :data:`AA_INDEX_VERSION`, or none.

    Raises on HTTP / parse failure.
    """
    resp = await get_with_retries(client, AA_LEADERBOARD_URL)
    resp.raise_for_status()
    versions = set(_AA_VERSION_RE.findall(resp.text))
    if versions != {AA_INDEX_VERSION}:
        logger.debug(
            f"AA index: leaderboard reports version {', '.join(sorted(versions)) or 'unknown'}, "
            f"bounds are calibrated on v{AA_INDEX_VERSION}; using the snapshot"
        )
        return get_aa_curated_fallback()
    # Primary: Next.js App Router RSC stream (current site format).
    pairs = _extract_aa_pairs_from_html(resp.text)
    # Legacy fallback: classic __NEXT_DATA__ JSON blob (older site format).
    if not pairs:
        match = _NEXT_DATA_RE.search(resp.text)
        if match:
            try:
                pairs = _extract_aa_pairs(json.loads(match.group("json")))
            except (ValueError, json.JSONDecodeError):
                pairs = []
    if not pairs:
        raise ExtractionFailed(
            "AA leaderboard: no (name, score) pairs found "
            "(neither RSC __next_f nor __NEXT_DATA__ matched)"
        )
    # When the same display name appears multiple times (different size /
    # reasoning tiers), keep the maximum value — it represents the most
    # capable variant available.
    best_by_name: dict[str, float] = {}
    for name, score in pairs:
        current = best_by_name.get(name)
        if current is None or score > current:
            best_by_name[name] = score

    live: dict[str, float] = {}
    for name, score in best_by_name.items():
        # Exact display name first, then canonicalized (variant-stripped) match.
        hf_ids = AA_NAME_TO_HF_IDS.get(name) or _AA_CANON_TO_HF_IDS.get(
            _canonical_name(name)
        )
        if not hf_ids:
            continue
        normalized = _normalize_aa_index(score)
        for hf_id in hf_ids:
            if hf_id not in live or normalized > live[hf_id]:
                live[hf_id] = normalized
    if not live:
        raise ExtractionFailed("AA index: live fetch returned 0 mapped scores")

    # Overlay live scores on top of the curated snapshot so a successful live
    # fetch can only ADD coverage, never shrink it below the fallback. Live
    # numbers win wherever both exist; the snapshot fills the long tail of
    # models AA labels in a way we can't map (or no longer tracks).
    scores = get_aa_curated_fallback()
    scores.update(live)
    logger.debug(f"AA index: {len(live)} live + {len(scores)} merged scores")
    return scores


def get_aa_curated_fallback() -> dict[str, float]:
    """Return the curated snapshot, normalized to the 0-100 scale.

    Live entries were read on :data:`AA_SNAPSHOT_READ_AT`, on index
    :data:`AA_INDEX_VERSION`.

    Used when the live fetch fails or reports another index version, and
    for mapped models AA no longer lists.
    """
    result: dict[str, float] = {}
    for hf_id, raw in AA_INDEX_SNAPSHOT.items():
        normalized = _normalize_aa_index(raw)
        if normalized > 0:
            result[hf_id] = normalized
    return result
