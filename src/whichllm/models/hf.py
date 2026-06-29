"""HuggingFace Hub client orchestration."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from whichllm.models.http import DEFAULT_ACCEPT_ENCODING, get_with_retries
from whichllm.models.parser import _extract_published_at, _parse_model
from whichllm.models.types import ModelInfo

logger = logging.getLogger(__name__)

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"

_MODEL_EXPANDS = [
    "config",
    "safetensors",
    "gguf",
    "cardData",
    "siblings",
    "evalResults",
]

_MODEL_DETAIL_EXPANDS = [
    "config",
    "safetensors",
    "gguf",
    "cardData",
    "siblings",
    "evalResults",
    "downloads",
    "likes",
    "createdAt",
    "lastModified",
]

_FRONTIER_MODEL_IDS = (
    # Newest releases that lead 2026-Q2 benchmarks
    "moonshotai/Kimi-K2-Thinking",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Instruct-0905",
    "XiaomiMiMo/MiMo-V2.5-Pro",
    "XiaomiMiMo/MiMo-V2.5",
    "XiaomiMiMo/MiMo-V2-Flash",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1-0528",
    "zai-org/GLM-5.1",
    "zai-org/GLM-5",
    "zai-org/GLM-5-FP8",
    "zai-org/GLM-5.1-FP8",
    "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-4.6",
    "zai-org/GLM-4.5",
    "zai-org/GLM-4.5-Air",
    # Open-weight mid-size frontier
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-4B-Instruct-2507",
    # Reasoning/thinking lines that do not auto-surface via cardinality
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-4B-Thinking-2507",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Other current open releases
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Devstral-Small-2505",
    "mistralai/Codestral-22B-v0.1",
    "MiniMaxAI/MiniMax-M2",
    "MiniMaxAI/MiniMax-M2.5",
    # IBM Granite latest open releases
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-3.3-2b-instruct",
    # AllenAI Olmo-3
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-1025-7B",
    # Nemotron 3 series
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)


def _hf_api_url(path: str) -> str:
    raw_endpoint = os.environ.get("HF_ENDPOINT")
    endpoint = _DEFAULT_HF_ENDPOINT if raw_endpoint is None else raw_endpoint.strip()
    if not endpoint:
        raise ValueError("HF_ENDPOINT must not be empty")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("HF_ENDPOINT must start with http:// or https://")
    endpoint = endpoint.rstrip("/")
    return f"{endpoint}/api/{path.lstrip('/')}"


def _model_list_params(limit: int, sort: str, filter_value: str | None = None) -> dict:
    params = {
        "pipeline_tag": "text-generation",
        "sort": sort,
        "limit": str(limit),
        "expand[]": _MODEL_EXPANDS,
    }
    if filter_value:
        params["filter"] = filter_value
    return params


def _append_new_models(
    data_list: list[dict],
    models: list[ModelInfo],
    seen_ids: set[str],
) -> None:
    for data in data_list:
        if data.get("id") not in seen_ids:
            model = _parse_model(data)
            if model:
                models.append(model)
                seen_ids.add(model.id)


async def _fetch_model_list(
    client: httpx.AsyncClient,
    params: dict,
) -> list[dict]:
    resp = await get_with_retries(client, _hf_api_url("models"), params=params)
    resp.raise_for_status()
    return resp.json()


async def _fetch_frontier_models(
    client: httpx.AsyncClient,
    models: list[ModelInfo],
    seen_ids: set[str],
) -> None:
    for model_id in _FRONTIER_MODEL_IDS:
        if model_id in seen_ids:
            continue
        try:
            resp = await get_with_retries(
                client,
                _hf_api_url(f"models/{model_id}"),
                params={"expand[]": _MODEL_DETAIL_EXPANDS},
            )
            if resp.status_code >= 400:
                logger.debug(
                    f"Frontier fetch skipped {model_id}: HTTP {resp.status_code}"
                )
                continue
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.debug(f"Frontier fetch failed for {model_id}: {e}")
            continue
        model = _parse_model(data)
        if model:
            models.append(model)
            seen_ids.add(model.id)


async def fetch_models(
    limit: int = 300, include_vision: bool = True
) -> list[ModelInfo]:
    """Fetch popular models from HuggingFace Hub."""
    models: list[ModelInfo] = []

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"Accept-Encoding": DEFAULT_ACCEPT_ENCODING},
    ) as client:
        logger.debug(f"Fetching models from HF API (limit={limit})")
        data_list = await _fetch_model_list(
            client, _model_list_params(limit, sort="downloads")
        )
        for data in data_list:
            model = _parse_model(data)
            if model:
                models.append(model)

        logger.debug("Fetching GGUF models from HF API")
        gguf_data_list = await _fetch_model_list(
            client, _model_list_params(limit, sort="downloads", filter_value="gguf")
        )

        seen_ids = {m.id for m in models}
        _append_new_models(gguf_data_list, models, seen_ids)

        logger.debug("Fetching recent GGUF models from HF API")
        recent_data_list = await _fetch_model_list(
            client, _model_list_params(limit, sort="lastModified", filter_value="gguf")
        )
        _append_new_models(recent_data_list, models, seen_ids)

        for filter_value in (None, "gguf"):
            logger.debug(
                f"Fetching trending {filter_value or 'all'} models from HF API"
            )
            try:
                trending_data_list = await _fetch_model_list(
                    client,
                    _model_list_params(
                        limit, sort="trending", filter_value=filter_value
                    ),
                )
            except (httpx.HTTPError, ValueError) as e:
                logger.debug(f"Trending fetch skipped: {e}")
                continue
            _append_new_models(trending_data_list, models, seen_ids)

        await _fetch_frontier_models(client, models, seen_ids)

        if include_vision:
            for pipeline_tag in ("image-text-to-text",):
                mm_params = {
                    "pipeline_tag": pipeline_tag,
                    "sort": "downloads",
                    "limit": str(limit),
                    "expand[]": _MODEL_EXPANDS,
                }
                logger.debug(f"Fetching {pipeline_tag} models from HF API")
                mm_data_list = await _fetch_model_list(client, mm_params)
                _append_new_models(mm_data_list, models, seen_ids)

    logger.debug(f"Fetched {len(models)} models total")
    return models


async def fetch_model_published_at(model_ids: list[str]) -> dict[str, str]:
    """Fetch published timestamps for specific model IDs."""
    unique_ids = sorted({m for m in model_ids if m})
    if not unique_ids:
        return {}

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"Accept-Encoding": DEFAULT_ACCEPT_ENCODING},
    ) as client:
        tasks = [
            client.get(
                _hf_api_url(f"models/{model_id}"),
                params={"expand[]": ["createdAt", "lastModified"]},
            )
            for model_id in unique_ids
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    result: dict[str, str] = {}
    for model_id, resp in zip(unique_ids, responses, strict=False):
        if isinstance(resp, Exception):
            logger.debug("Failed to fetch model detail for %s: %s", model_id, resp)
            continue
        if resp.status_code >= 400:
            logger.debug(
                "Failed to fetch model detail for %s: HTTP %s",
                model_id,
                resp.status_code,
            )
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        published_at = _extract_published_at(data)
        if published_at:
            result[model_id] = published_at
    return result
