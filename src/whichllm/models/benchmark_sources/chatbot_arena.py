from __future__ import annotations

import re

import httpx

from whichllm.models.http import get_with_retries

# --- Data source URLs ---
ARENA_ROWS_URL = "https://datasets-server.huggingface.co/rows"
ARENA_DATASET = "mathewhe/chatbot-arena-elo"

# --- Arena ELO normalization ---
# Open-source ELO range: ~1030 (worst) to ~1424 (best). Arena is frozen
# 2025-07-17 (no new models added) so the leaderboard cannot reflect any
# 2025-Q3+ release; we cap the normalized output at 82 so newer benchmark
# sources (AA Index / LiveBench, which can reach 95+) decisively win on
# conflict.
_ARENA_ELO_MIN = 1030
_ARENA_ELO_MAX = 1430
_ARENA_MAX_NORMALIZED = 82.0

# --- Arena display name -> HuggingFace org mapping ---
_ARENA_ORG_TO_HF: dict[str, list[str]] = {
    "Alibaba": ["Qwen"],
    "Meta": ["meta-llama"],
    "DeepSeek": ["deepseek-ai"],
    "DeepSeek AI": ["deepseek-ai"],
    "Google": ["google"],
    "Mistral": ["mistralai"],
    "Microsoft": ["microsoft"],
    "Nvidia": ["nvidia"],
    "01 AI": ["01-ai"],
    "Allen AI": ["allenai"],
    "Ai2": ["allenai"],
    "AllenAI/UW": ["allenai"],
    "Cohere": ["CohereForAI"],
    "HuggingFace": ["HuggingFaceH4", "huggingface"],
    "AI21 Labs": ["ai21labs"],
    "NousResearch": ["NousResearch"],
    "NexusFlow": ["Nexusflow"],
    "Princeton": ["princeton-nlp"],
    "IBM": ["ibm-granite"],
    "InternLM": ["internlm"],
    "Together AI": ["togethercomputer"],
    "TII": ["tiiuae"],
    "MiniMax": ["MiniMaxAI"],
    "MosaicML": ["mosaicml"],
    "Databricks": ["databricks"],
    "Moonshot": ["moonshotai"],
    "UC Berkeley": ["berkeley-nest"],
    "Cognitive Computations": ["cognitivecomputations"],
    "Upstage AI": ["upstage"],
    "UW": ["timdettmers"],
    "Snowflake": ["Snowflake"],
    "LMSYS": ["lmsys"],
    "OpenChat": ["openchat"],
}


def _normalize_arena_elo(elo: float) -> float:
    """Normalize Arena ELO to a frozen-source-aware 0-_ARENA_MAX_NORMALIZED scale."""
    score = (
        (elo - _ARENA_ELO_MIN)
        / (_ARENA_ELO_MAX - _ARENA_ELO_MIN)
        * _ARENA_MAX_NORMALIZED
    )
    return max(0.0, min(_ARENA_MAX_NORMALIZED, round(score, 1)))


def _arena_name_to_hf_ids(model_name: str, org: str) -> list[str]:
    """Convert Arena display name to potential HuggingFace model IDs."""
    hf_orgs = _ARENA_ORG_TO_HF.get(org, [])
    candidates = []

    # Clean the model name: remove date suffixes like "(03-2025)"
    clean_name = re.sub(r"\s*\([\d-]+\)\s*$", "", model_name).strip()
    # Remove -bf16, -fp8 suffixes for base matching
    base_name = re.sub(r"-(bf16|fp8|fp16)$", "", clean_name, flags=re.IGNORECASE)

    for hf_org in hf_orgs:
        candidates.append(f"{hf_org}/{clean_name}")
        if base_name != clean_name:
            candidates.append(f"{hf_org}/{base_name}")
        # Try with -Instruct suffix stripped for base model matching
        no_instruct = re.sub(r"-Instruct$", "", clean_name)
        if no_instruct != clean_name:
            candidates.append(f"{hf_org}/{no_instruct}")

    return candidates


async def fetch_arena_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch Chatbot Arena ELO scores via rows API."""
    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = await get_with_retries(
            client,
            ARENA_ROWS_URL,
            params={
                "dataset": ARENA_DATASET,
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
            model_name = str(row.get("Model", ""))
            elo = row.get("Arena Score", 0)
            org = str(row.get("Organization", ""))
            lic = str(row.get("License", ""))

            if not model_name or not elo or elo <= 0:
                continue
            # Skip proprietary models (can't run locally)
            if "Proprietary" in lic or "Propretary" in lic:
                continue

            normalized = _normalize_arena_elo(elo)
            # Map to all potential HF IDs
            hf_ids = _arena_name_to_hf_ids(model_name, org)
            for hf_id in hf_ids:
                scores[hf_id] = normalized

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores
