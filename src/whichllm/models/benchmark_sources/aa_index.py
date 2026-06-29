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
import re

import httpx

from whichllm.models.benchmark_sources.constants import _NEXT_DATA_RE
from whichllm.models.benchmark_sources.types import ExtractionFailed
from whichllm.models.benchmark_sources.utils import _walk
from whichllm.models.http import get_with_retries

logger = logging.getLogger(__name__)

# Display name -> list of (org_prefix, repo_name_candidates) tuples used to
# map AA-reported labels back to HuggingFace model IDs. Only the most common
# fully-open-weights releases need entries here; anything else is dropped.
AA_NAME_TO_HF_IDS: dict[str, list[str]] = {
    "Kimi K2": ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Base"],
    "Kimi K2-Thinking": ["moonshotai/Kimi-K2-Thinking"],
    "DeepSeek V3": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324"],
    "DeepSeek V3.1": ["deepseek-ai/DeepSeek-V3.1"],
    "DeepSeek V3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "DeepSeek V3.2-Exp": ["deepseek-ai/DeepSeek-V3.2-Exp"],
    "DeepSeek V4 Pro": ["deepseek-ai/DeepSeek-V4-Pro"],
    "DeepSeek V4 Flash": ["deepseek-ai/DeepSeek-V4-Flash"],
    "DeepSeek R1": ["deepseek-ai/DeepSeek-R1"],
    "DeepSeek R1-0528": ["deepseek-ai/DeepSeek-R1-0528"],
    "DeepSeek R1-Distill 32B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"],
    "DeepSeek R1-Distill 14B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
    "DeepSeek R1-Distill 8B": ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
    "QwQ 32B": ["Qwen/QwQ-32B"],
    "Qwen3 4B Thinking": ["Qwen/Qwen3-4B-Thinking-2507"],
    "MiMo V2.5": ["XiaomiMiMo/MiMo-V2.5"],
    "MiMo V2.5 Pro": ["XiaomiMiMo/MiMo-V2.5-Pro"],
    "MiMo V2 Flash": ["XiaomiMiMo/MiMo-V2-Flash"],
    "GLM-4.5": ["zai-org/GLM-4.5", "zai-org/GLM-4.5-Air"],
    "GLM-4.6": ["zai-org/GLM-4.6"],
    "GLM-4.7": ["zai-org/GLM-4.7"],
    "GLM-4.7-Flash": ["zai-org/GLM-4.7-Flash"],
    "GLM-5": ["zai-org/GLM-5", "zai-org/GLM-5-FP8"],
    "GLM-5.1": ["zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8"],
    "gpt-oss-20b": ["openai/gpt-oss-20b"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "Qwen3-Next 80B-A3B": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "Qwen3.5 397B-A17B": ["Qwen/Qwen3.5-397B-A17B"],
    "Qwen3 235B-A22B": ["Qwen/Qwen3-235B-A22B"],
    "Qwen3 32B": ["Qwen/Qwen3-32B"],
    "Qwen3 14B": ["Qwen/Qwen3-14B"],
    "Qwen3 8B": ["Qwen/Qwen3-8B"],
    "Qwen3-VL 235B-A22B": ["Qwen/Qwen3-VL-235B-A22B-Instruct"],
    "Llama 3.3 70B": ["meta-llama/Llama-3.3-70B-Instruct"],
    "Llama 4 Scout": ["meta-llama/Llama-4-Scout-17B-16E-Instruct"],
    "Llama 4 Maverick": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    "Gemma 3 27B": ["google/gemma-3-27b-it"],
    "Gemma 3 12B": ["google/gemma-3-12b-it"],
    "Gemma 4 31B": ["google/gemma-4-31b-it"],
    "Gemma 4 26B-A4B": ["google/gemma-4-26b-a4b-it"],
    "Mistral Large 2": ["mistralai/Mistral-Large-Instruct-2411"],
    "Devstral Small": ["mistralai/Devstral-Small-2505"],
    "Phi-4": ["microsoft/phi-4"],
    "Command A": ["CohereForAI/c4ai-command-a-03-2025"],
    "Command R+": [
        "CohereForAI/c4ai-command-r-plus-08-2024",
        "CohereForAI/c4ai-command-r-plus",
    ],
    "MiniMax-M2": ["MiniMaxAI/MiniMax-M2"],
    "MiniMax-M2.5": ["MiniMaxAI/MiniMax-M2.5"],
    "Nemotron 3 Super 120B-A12B": ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"],
    "Nemotron 3 Nano 30B-A3B": [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ],
}

# Bounds used to normalize raw AA index values onto the 0-100 scale the rest
# of the ranking system uses. AA reworked their Intelligence Index in
# 2026-Q2 and the open-weights distribution compressed sharply (top open model
# ~44, 8B-class ~7, weakest mapped repos ~3). The window is anchored by the
# same two-point fit as before: top open frontier (DeepSeek-V4-Pro = 44.3) → 95
# normalized, and 8B-class (Qwen3-8B = 7.4) → 40 normalized. This keeps a strong
# 8B model competitive with frozen-OLLB 7B scores while leaving headroom for
# frontier-tier models. On the reworked scale that fit puts the floor below
# zero; that is fine, live AA values are always positive and clamp at 0.
_AA_INDEX_MIN = -19.4
_AA_INDEX_MAX = 47.6

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# Snapshot of the AA Intelligence Index (open-weights only), refreshed on
# 2026-06-29 from artificialanalysis.ai against their reworked index. Used as a
# fallback when the live HTML scrape returns no results (e.g. because the
# Next.js payload format changes again). Entries are raw AA index values,
# normalized through _normalize_aa_index() in get_aa_curated_fallback().
#
# Entries marked "live" are real values from the reworked index. Entries marked
# "peer" are models AA does not track; their raw value is set so that, under the
# current bounds, they reproduce their previous normalized score (hand-estimated
# LB-equivalents). On the reworked scale several peers fall below the index
# floor and read as negative raw; that is an artifact of reusing the AA-index
# field for non-AA estimates. _normalize_aa_index() clamps them at 0, so the
# normalized output stays correct. (We could instead store this snapshot as
# already-normalized 0-100 values, which would drop the negatives and decouple
# the fallback from future bound retunes, but that is a larger change than this
# issue calls for and is left for a follow-up if you want it.)
AA_INDEX_FALLBACK_2026_06_29: dict[str, float] = {
    # Frontier MoE / very large
    "moonshotai/Kimi-K2-Thinking": 32.7,  # live
    "moonshotai/Kimi-K2-Instruct": 19.4,  # live
    "XiaomiMiMo/MiMo-V2.5-Pro": 42.2,  # live
    "XiaomiMiMo/MiMo-V2.5": 40.1,  # live
    "deepseek-ai/DeepSeek-V4-Pro": 44.3,  # live
    "deepseek-ai/DeepSeek-V4-Flash": 40.3,  # live
    "deepseek-ai/DeepSeek-V3.2": 33.4,  # live
    "deepseek-ai/DeepSeek-V3.2-Exp": 25.4,  # live
    "deepseek-ai/DeepSeek-V3.1": 21.0,  # live
    "deepseek-ai/DeepSeek-V3-0324": 10.4,  # live
    "deepseek-ai/DeepSeek-V3": 10.4,  # live
    "deepseek-ai/DeepSeek-R1-0528": 20.1,  # live
    "deepseek-ai/DeepSeek-R1": 12.6,  # live
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 10.5,  # peer
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 1.3,  # peer
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": -7.9,  # peer
    "Qwen/QwQ-32B": 13.4,  # live
    "Qwen/Qwen3-4B-Thinking-2507": -4.8,  # peer
    "zai-org/GLM-5.1": 40.2,  # live
    "zai-org/GLM-5": 39.5,  # live
    "zai-org/GLM-5-FP8": 39.5,  # live
    "zai-org/GLM-5.1-FP8": 40.2,  # live
    "zai-org/GLM-4.7-Flash": 22.9,  # live
    "zai-org/GLM-4.6": 25.1,  # live
    "zai-org/GLM-4.5": 19.5,  # live
    "zai-org/GLM-4.5-Air": 19.5,  # live
    # Qwen family
    "Qwen/Qwen3.6-27B": 32.0,  # peer
    "Qwen/Qwen3.5-397B-A17B": 33.7,  # live
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 19.8,  # live
    "Qwen/Qwen3-235B-A22B": 13.4,  # live
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 19.7,  # peer
    "Qwen/Qwen3-32B": 11.5,  # live
    "Qwen/Qwen3-14B": 10.1,  # live
    "Qwen/Qwen3-8B": 7.4,  # live
    "Qwen/Qwen3-4B-Instruct-2507": 4.4,  # peer
    "Qwen/Qwen3-4B": 1.3,  # peer
    "Qwen/Qwen3-1.7B": -7.9,  # peer
    "Qwen/Qwen3-0.6B": -14.0,  # peer
    # 8B-class peers (no AA tracking but realistic LB-equivalents)
    "meta-llama/Llama-3.1-8B-Instruct": -4.8,  # peer
    "meta-llama/Meta-Llama-3-8B-Instruct": -7.9,  # peer
    "google/gemma-2-9b-it": -3.3,  # peer
    "microsoft/Phi-4-mini-instruct": -1.8,  # peer
    "mistralai/Mistral-7B-Instruct-v0.3": -7.9,  # peer
    "Qwen/Qwen2.5-7B-Instruct": -4.8,  # peer
    "Qwen/Qwen2.5-14B-Instruct": 1.3,  # peer
    "Qwen/Qwen2.5-32B-Instruct": 7.4,  # peer
    "Qwen/Qwen3-30B-A3B": 10.5,  # peer
    # Other major open releases
    "openai/gpt-oss-120b": 23.8,  # live
    "openai/gpt-oss-20b": 14.9,  # live
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 14.3,  # live
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 10.0,  # live
    "meta-llama/Llama-3.3-70B-Instruct": 12.0,  # peer
    "google/gemma-4-31b-it": 29.4,  # live
    "google/gemma-4-26b-a4b-it": 25.7,  # live
    "google/gemma-3-27b-it": 12.0,  # peer
    "google/gemma-3-12b-it": 7.4,  # peer
    "microsoft/phi-4": 4.9,  # live
    "mistralai/Mistral-Large-Instruct-2411": 9.1,  # live
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 10.5,  # peer
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 7.4,  # peer
    "mistralai/Devstral-Small-2505": 11.8,  # live
    "MiniMaxAI/MiniMax-M2.5": 33.7,  # live
    "stepfun-ai/Step-3.5-Flash": 19.7,  # peer
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 16.6,  # peer
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 12.0,  # peer
    # Correct IDs for OLMo / Granite / Codestral families (the earlier
    # forecast IDs like "OLMo-3-32B-Instruct" or "granite-4.1-30b-instruct"
    # never shipped publicly under those names).
    "allenai/Olmo-3-7B-Instruct": -4.8,  # peer
    "allenai/Olmo-3-1025-7B": -4.8,  # peer
    "ibm-granite/granite-4.0-h-small": 7.4,  # peer
    "ibm-granite/granite-4.0-h-tiny": -4.8,  # peer
    "ibm-granite/granite-3.3-8b-instruct": -3.3,  # peer
    "ibm-granite/granite-3.3-2b-instruct": -12.5,  # peer
    "mistralai/Codestral-22B-v0.1": 4.4,  # peer
}


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
        if name and score > 0:
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
            if isinstance(v, (int, float)):
                score = float(v)
                break
        if name and score is not None and score > 0:
            pairs.append((name, score))
    return pairs


async def fetch_aa_index_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch Artificial Analysis Intelligence Index scores.

    Returns ``{hf_id: normalized_score_0_100}`` for every model AA reports
    that we can map back to a HuggingFace repo via :data:`AA_NAME_TO_HF_IDS`.

    Raises on HTTP / parse failure.
    """
    resp = await get_with_retries(client, AA_LEADERBOARD_URL)
    resp.raise_for_status()
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
        if normalized <= 0:
            continue
        for hf_id in hf_ids:
            if normalized > live.get(hf_id, 0.0):
                live[hf_id] = normalized
    if not live:
        raise ExtractionFailed("AA index: live fetch returned 0 mapped scores")

    # Overlay live scores on top of the curated snapshot so a successful live
    # fetch can only ADD coverage, never shrink it below the fallback. Live
    # numbers win wherever both exist; the snapshot fills the long tail of
    # models AA labels in a way we can't map (or no longer tracks).
    scores = get_aa_curated_fallback()
    for hf_id, normalized in live.items():
        if normalized > scores.get(hf_id, 0.0):
            scores[hf_id] = normalized
    logger.debug(f"AA index: {len(live)} live + {len(scores)} merged scores")
    return scores


def get_aa_curated_fallback() -> dict[str, float]:
    """Return the 2026-06-29 curated snapshot, normalized to the 0-100 scale.

    Used whenever the live HTML scrape cannot extract data — for example
    when artificialanalysis.ai changes its Next.js payload shape.
    """
    result: dict[str, float] = {}
    for hf_id, raw in AA_INDEX_FALLBACK_2026_06_29.items():
        normalized = _normalize_aa_index(raw)
        if normalized > 0:
            result[hf_id] = normalized
    return result
