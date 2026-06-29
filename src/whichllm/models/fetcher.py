"""Compatibility shim for HuggingFace model fetching helpers.

The implementation is split by responsibility across ``whichllm.models``:

- ``hf`` for HuggingFace API orchestration
- ``parser`` for API payload to ModelInfo conversion
- ``parameters`` for parameter-count and MoE normalization
- ``gguf`` for GGUF filename/variant parsing
- ``sliding_window`` for SWA metadata resolution
- ``serialization`` for cache serialization

Existing imports from ``whichllm.models.fetcher`` are re-exported here.
"""

from __future__ import annotations

from whichllm.models import hf as _hf_module
from whichllm.models.gguf import (
    _estimate_gguf_size,
    _extract_gguf_variants,
    _extract_quant_type,
)
from whichllm.models.hf import (
    _DEFAULT_HF_ENDPOINT,
    _FRONTIER_MODEL_IDS,
    _hf_api_url,
)
from whichllm.models.http import get_with_retries
from whichllm.models.parameters import (
    _AUTHORITATIVE_PARAM_COUNTS,
    _KNOWN_MOE_ACTIVE_PARAMS,
    _KNOWN_PARAM_COUNTS,
    _extract_active_size_hint_from_id,
    _extract_size_hint_from_id,
    _is_quantized_repo_name,
    _lookup_curated_count,
    _normalize_param_count,
    _resolve_moe_active_params,
)
from whichllm.models.parser import (
    _extract_architecture,
    _extract_hf_eval_score,
    _extract_param_count,
    _extract_published_at,
    _is_general_eval_entry,
    _normalize_eval_value,
    _parse_model,
)
from whichllm.models.serialization import dicts_to_models, models_to_dicts
from whichllm.models.sliding_window import (
    _SWA_ARCH_ALIASES,
    _SWA_ARCH_DEFAULTS,
    _resolve_sliding_window,
    _swa_arch_key,
    _swa_key_from_arch,
)


async def fetch_models(limit: int = 300, include_vision: bool = True):
    """Fetch popular models from HuggingFace Hub.

    The wrapper keeps legacy monkey-patching of ``fetcher.get_with_retries``
    working while the implementation lives in ``whichllm.models.hf``.
    """
    original = _hf_module.get_with_retries
    _hf_module.get_with_retries = get_with_retries
    try:
        return await _hf_module.fetch_models(limit=limit, include_vision=include_vision)
    finally:
        _hf_module.get_with_retries = original


async def fetch_model_published_at(model_ids: list[str]):
    """Fetch published timestamps for specific model IDs."""
    return await _hf_module.fetch_model_published_at(model_ids)


__all__ = [
    "_AUTHORITATIVE_PARAM_COUNTS",
    "_DEFAULT_HF_ENDPOINT",
    "_FRONTIER_MODEL_IDS",
    "_KNOWN_MOE_ACTIVE_PARAMS",
    "_KNOWN_PARAM_COUNTS",
    "_SWA_ARCH_ALIASES",
    "_SWA_ARCH_DEFAULTS",
    "_estimate_gguf_size",
    "_extract_active_size_hint_from_id",
    "_extract_architecture",
    "_extract_gguf_variants",
    "_extract_hf_eval_score",
    "_extract_param_count",
    "_extract_published_at",
    "_extract_quant_type",
    "_extract_size_hint_from_id",
    "_hf_api_url",
    "_is_general_eval_entry",
    "_is_quantized_repo_name",
    "_lookup_curated_count",
    "_normalize_eval_value",
    "_normalize_param_count",
    "_parse_model",
    "_resolve_moe_active_params",
    "_resolve_sliding_window",
    "_swa_arch_key",
    "_swa_key_from_arch",
    "dicts_to_models",
    "fetch_model_published_at",
    "fetch_models",
    "get_with_retries",
    "models_to_dicts",
]
