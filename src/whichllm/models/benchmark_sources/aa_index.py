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

from whichllm.models.benchmark_sources.constants import _NEXT_DATA_RE
from whichllm.models.benchmark_sources.types import ExtractionFailed
from whichllm.models.benchmark_sources.utils import _walk
from whichllm.models.http import get_with_retries

logger = logging.getLogger(__name__)

# Display name -> list of (org_prefix, repo_name_candidates) tuples used to
# map AA-reported labels back to HuggingFace model IDs. Only the most common
# fully-open-weights releases need entries here; anything else is dropped.
AA_NAME_TO_HF_IDS: dict[str, list[str]] = {
    # AA's "Kimi K2" row links its weights to Kimi-K2-Instruct (modelWeightsSourceUrl).
    # No AA row evaluates Kimi-K2-Base, so it is left unmapped.
    "Kimi K2": ["moonshotai/Kimi-K2-Instruct"],
    # AA's "Kimi K2 0905" row links its weights to Kimi-K2-Instruct-0905.
    "Kimi K2 0905": ["moonshotai/Kimi-K2-Instruct-0905"],
    "Kimi K2-Thinking": ["moonshotai/Kimi-K2-Thinking"],
    "DeepSeek V3": ["deepseek-ai/DeepSeek-V3"],
    "DeepSeek V3 0324": ["deepseek-ai/DeepSeek-V3-0324"],
    "DeepSeek V3.1": ["deepseek-ai/DeepSeek-V3.1"],
    "DeepSeek V3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "DeepSeek V3.2-Exp": ["deepseek-ai/DeepSeek-V3.2-Exp"],
    # AA splits DeepSeek V4 into dated releases. Each row's modelWeightsSourceUrl
    # names its own repo: 0424 and 0420 are the original undated repos.
    "DeepSeek V4 Pro 0424": ["deepseek-ai/DeepSeek-V4-Pro"],
    "DeepSeek V4 Pro 0813": ["deepseek-ai/DeepSeek-V4-Pro-0813"],
    "DeepSeek V4 Flash 0420": ["deepseek-ai/DeepSeek-V4-Flash"],
    "DeepSeek V4 Flash 0731": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
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
    "GLM-4.5": ["zai-org/GLM-4.5"],
    "GLM-4.5-Air": ["zai-org/GLM-4.5-Air"],
    "GLM-4.6": ["zai-org/GLM-4.6"],
    "GLM-4.7": ["zai-org/GLM-4.7"],
    "GLM-4.7-Flash": ["zai-org/GLM-4.7-Flash"],
    "GLM-5": ["zai-org/GLM-5", "zai-org/GLM-5-FP8"],
    "GLM-5.1": ["zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8"],
    "gpt-oss-20b": ["openai/gpt-oss-20b"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "Qwen3-Next 80B-A3B": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "Qwen3.8 27B": ["Qwen/Qwen3.8-27B"],
    "Qwen3.6 35B-A3B": ["Qwen/Qwen3.6-35B-A3B"],
    "Qwen3.6 27B": ["Qwen/Qwen3.6-27B"],
    "Qwen3.5 397B-A17B": ["Qwen/Qwen3.5-397B-A17B"],
    "Qwen3.5 122B-A10B": ["Qwen/Qwen3.5-122B-A10B"],
    "Qwen3.5 35B-A3B": ["Qwen/Qwen3.5-35B-A3B"],
    "Qwen3.5 27B": ["Qwen/Qwen3.5-27B"],
    "Qwen3.5 9B": ["Qwen/Qwen3.5-9B"],
    "Qwen3.5 4B": ["Qwen/Qwen3.5-4B"],
    "Qwen3 Coder Next": ["Qwen/Qwen3-Coder-Next"],
    "Qwen3 Coder 480B-A35B Instruct": ["Qwen/Qwen3-Coder-480B-A35B-Instruct"],
    "Qwen3 Coder 30B A3B Instruct": ["Qwen/Qwen3-Coder-30B-A3B-Instruct"],
    "Qwen2.5 Coder Instruct 32B": ["Qwen/Qwen2.5-Coder-32B-Instruct"],
    "Qwen2.5 Coder Instruct 7B": ["Qwen/Qwen2.5-Coder-7B-Instruct"],
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
    # AA's only Command R+ row is "Command-R+ (Apr '24)", whose weights link to
    # c4ai-command-r-plus. No AA row evaluates the 08-2024 release, so it is left unmapped.
    "Command R+": ["CohereForAI/c4ai-command-r-plus"],
    "MiniMax-M2": ["MiniMaxAI/MiniMax-M2"],
    "MiniMax-M2.5": ["MiniMaxAI/MiniMax-M2.5"],
    "Nemotron 3 Super 120B-A12B": ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"],
    "Nemotron 3 Nano 30B-A3B": [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ],
}

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

# The AA index version the bounds and snapshot were read on. Raw values from
# another version are on a different scale, so fetch_aa_index_scores() returns
# the snapshot when the leaderboard reports any other (or no) version.
AA_INDEX_VERSION = "4.3.2"
# The day the curated snapshot below was read.
AA_SNAPSHOT_READ_AT = "2026-10-27"
_AA_VERSION_RE = re.compile(r"Artificial Analysis Intelligence Index v(\d+(?:\.\d+)*)")

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# Snapshot of the AA Intelligence Index v4.3.2 (open-weights only). Every "live"
# entry was read on 2026-10-27 from https://artificialanalysis.ai/models/<slug>, where
# <slug> is named in the entry's comment (e.g. "kimi-k2" is
# https://artificialanalysis.ai/models/kimi-k2) and is the highest variant of the
# release, as fetch_aa_index_scores() takes it. "estimated" marks values AA
# flags as estimated rather than measured. Used as a fallback when the live
# HTML scrape returns no results (e.g. because the Next.js payload format
# changes again), and for mapped models whose AA release is deprecated: the
# leaderboard page no longer carries an index for those (Kimi K2, GLM-5, ...),
# so even a successful live fetch falls back to these values. Entries are raw
# AA index values, normalized through _normalize_aa_index() in
# get_aa_curated_fallback().
#
# Entries marked "peer" are models AA does not track; their raw value is set so
# that, under the current bounds, they reproduce their previous normalized score
# (hand-estimated LB-equivalents). Several peers fall below the index floor
# and read as negative raw; that is an artifact of reusing the AA-index
# field for non-AA estimates. _normalize_aa_index() clamps them at 0, so the
# normalized output stays correct. (We could instead store this snapshot as
# already-normalized 0-100 values, which would drop the negatives and decouple
# the fallback from future bound retunes, but that is a larger change than this
# issue calls for and is left for a follow-up if you want it.)
AA_INDEX_FALLBACK_2026_10_27: dict[str, float] = {
    # Frontier MoE / very large
    "moonshotai/Kimi-K2-Thinking": 22.0,  # live, estimated: kimi-k2-thinking
    "moonshotai/Kimi-K2-Instruct": 12.7,  # live, estimated: kimi-k2
    "moonshotai/Kimi-K2-Instruct-0905": 15.3,  # live, estimated: kimi-k2-0905
    "XiaomiMiMo/MiMo-V2.5-Pro": 26.0,  # live: mimo-v2-5-pro
    "XiaomiMiMo/MiMo-V2.5": 21.8,  # live: mimo-v2-5-0424
    "deepseek-ai/DeepSeek-V4-Pro": 30.4,  # live: deepseek-v4-pro-0424
    "deepseek-ai/DeepSeek-V4-Flash": 24.4,  # live: deepseek-v4-flash-0420-high
    "deepseek-ai/DeepSeek-V4-Pro-0813": 36.0,  # live: deepseek-v4-pro
    "deepseek-ai/DeepSeek-V4-Flash-0731": 34.3,  # live: deepseek-v4-flash
    "deepseek-ai/DeepSeek-V3.2": 21.5,  # live, estimated: deepseek-v3-2-reasoning
    "deepseek-ai/DeepSeek-V3.2-Exp": 16.6,  # live, estimated: deepseek-v3-2-reasoning-0925
    "deepseek-ai/DeepSeek-V3.1": 13.7,  # live, estimated: deepseek-v3-1
    "deepseek-ai/DeepSeek-V3-0324": 9.7,  # live: deepseek-v3-0324
    "deepseek-ai/DeepSeek-V3": 8.5,  # live: deepseek-v3
    "deepseek-ai/DeepSeek-R1-0528": 13.1,  # live, estimated: deepseek-r1
    "deepseek-ai/DeepSeek-R1": 11.4,  # live: deepseek-r1-0120
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 9.4,  # peer
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": -0.65,  # peer
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": -10.7,  # peer
    "Qwen/QwQ-32B": 9.5,  # live, estimated: qwq-32b
    "Qwen/Qwen3-4B-Thinking-2507": -7.3,  # peer
    "zai-org/GLM-5.1": 26.1,  # live: glm-5-1
    "zai-org/GLM-5": 27.9,  # live, estimated: glm-5
    "zai-org/GLM-5-FP8": 27.9,  # live, estimated: glm-5
    "zai-org/GLM-5.1-FP8": 26.1,  # live: glm-5-1
    "zai-org/GLM-4.7-Flash": 14.9,  # live, estimated: glm-4-7-flash
    "zai-org/GLM-4.6": 18.5,  # live, estimated: glm-4-6-reasoning
    "zai-org/GLM-4.5": 12.8,  # live, estimated: glm-4.5
    "zai-org/GLM-4.5-Air": 11.1,  # live, estimated: glm-4-5-air
    # Qwen family
    "Qwen/Qwen3.6-27B": 21.4,  # live: qwen3-6-27b
    "Qwen/Qwen3.5-397B-A17B": 21.4,  # live, estimated: qwen3-5-397b-a17b-non-reasoning
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 11.2,  # live, estimated: qwen3-next-80b-a3b-reasoning
    "Qwen/Qwen3-235B-A22B": 9.5,  # live, estimated: qwen3-235b-a22b-instruct-reasoning
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 9.6,  # live, estimated: qwen3-coder-30b-a3b-instruct
    "Qwen/Qwen3-32B": 7.3,  # live, estimated: qwen3-32b-instruct
    "Qwen/Qwen3-14B": 6.7,  # live, estimated: qwen3-14b-instruct
    "Qwen/Qwen3-8B": 6.0,  # live, estimated: qwen3-8b-instruct
    "Qwen/Qwen3-4B-Instruct-2507": 2.7,  # peer
    "Qwen/Qwen3-4B": -0.65,  # peer
    "Qwen/Qwen3-1.7B": -10.7,  # peer
    "Qwen/Qwen3-0.6B": -17.36,  # peer
    # Qwen3.5 / Qwen3.6 / Qwen3.8 releases and the current coder line.
    "Qwen/Qwen3.8-27B": 33.7,  # live: qwen3-8-27b
    "Qwen/Qwen3.6-35B-A3B": 18.2,  # live: qwen3-6-35b-a3b
    "Qwen/Qwen3.5-27B": 22.9,  # live, estimated: qwen3-5-27b
    "Qwen/Qwen3.5-122B-A10B": 17.7,  # live, estimated: qwen3-5-122b-a10b-non-reasoning
    "Qwen/Qwen3.5-35B-A3B": 19.3,  # live, estimated: qwen3-5-35b-a3b
    "Qwen/Qwen3.5-9B": 13.3,  # live, estimated: qwen3-5-9b-non-reasoning
    "Qwen/Qwen3.5-4B": 13.1,  # live, estimated: qwen3-5-4b
    "Qwen/Qwen3-Coder-Next": 9.2,  # live: qwen3-coder-next
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": 11.9,  # live, estimated: qwen3-coder-480b-a35b-instruct
    "Qwen/Qwen2.5-Coder-32B-Instruct": 6.7,  # live, estimated: qwen2-5-coder-32b-instruct
    "Qwen/Qwen2.5-Coder-7B-Instruct": 5.8,  # live, estimated: qwen2-5-coder-7b-instruct
    # 8B-class peers (no AA tracking but realistic LB-equivalents)
    "meta-llama/Llama-3.1-8B-Instruct": -7.3,  # peer
    "meta-llama/Meta-Llama-3-8B-Instruct": -10.7,  # peer
    "google/gemma-2-9b-it": -5.7,  # peer
    "microsoft/Phi-4-mini-instruct": -4.0,  # peer
    "mistralai/Mistral-7B-Instruct-v0.3": -10.7,  # peer
    "Qwen/Qwen2.5-7B-Instruct": -7.3,  # peer
    "Qwen/Qwen2.5-14B-Instruct": -0.65,  # peer
    "Qwen/Qwen2.5-32B-Instruct": 6.0,  # peer
    "Qwen/Qwen3-30B-A3B": 9.4,  # peer
    # Other major open releases
    "openai/gpt-oss-120b": 11.6,  # live: gpt-oss-120b
    "openai/gpt-oss-20b": 10.0,  # live, estimated: gpt-oss-20b-low
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 9.3,  # live: llama-4-maverick
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 6.5,  # live: llama-4-scout
    "meta-llama/Llama-3.3-70B-Instruct": 11.1,  # peer
    "google/gemma-4-31b-it": 14.7,  # live: gemma-4-31b
    "google/gemma-4-26b-a4b-it": 16.7,  # live, estimated: gemma-4-26b-a4b
    "google/gemma-3-27b-it": 11.1,  # peer
    "google/gemma-3-12b-it": 6.0,  # peer
    "microsoft/phi-4": 5.9,  # live, estimated: phi-4
    "mistralai/Mistral-Large-Instruct-2411": 7.6,  # live, estimated: mistral-large-2
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 9.4,  # peer
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 6.0,  # peer
    "mistralai/Devstral-Small-2505": 8.7,  # live, estimated: devstral-small-2505
    "CohereForAI/c4ai-command-a-03-2025": 7.0,  # live, estimated: command-a
    "MiniMaxAI/MiniMax-M2.5": 22.8,  # live, estimated: minimax-m2-5
    "stepfun-ai/Step-3.5-Flash": 19.5,  # peer
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 16.06,  # peer
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 11.1,  # peer
    # Correct IDs for OLMo / Granite / Codestral families (the earlier
    # forecast IDs like "OLMo-3-32B-Instruct" or "granite-4.1-30b-instruct"
    # never shipped publicly under those names).
    "allenai/Olmo-3-7B-Instruct": -7.3,  # peer
    "allenai/Olmo-3-1025-7B": -7.3,  # peer
    "ibm-granite/granite-4.0-h-small": 6.0,  # peer
    "ibm-granite/granite-4.0-h-tiny": -7.3,  # peer
    "ibm-granite/granite-3.3-8b-instruct": -5.7,  # peer
    "ibm-granite/granite-3.3-2b-instruct": -15.75,  # peer
    "mistralai/Codestral-22B-v0.1": 2.7,  # peer
}


# The snapshot entries marked "peer": hand estimates for models AA does not
# track. The merge uses them only for models no source measured.
AA_PEER_IDS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "allenai/Olmo-3-1025-7B",
        "allenai/Olmo-3-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "google/gemma-2-9b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "ibm-granite/granite-3.3-2b-instruct",
        "ibm-granite/granite-3.3-8b-instruct",
        "ibm-granite/granite-4.0-h-small",
        "ibm-granite/granite-4.0-h-tiny",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "mistralai/Codestral-22B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "stepfun-ai/Step-3.5-Flash",
    }
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

    All live entries read on 2026-10-27 (AA index v4.3.2).

    Used when the live fetch fails or reports another index version, and
    for mapped models AA no longer lists.
    """
    result: dict[str, float] = {}
    for hf_id, raw in AA_INDEX_FALLBACK_2026_10_27.items():
        normalized = _normalize_aa_index(raw)
        if normalized > 0:
            result[hf_id] = normalized
    return result
