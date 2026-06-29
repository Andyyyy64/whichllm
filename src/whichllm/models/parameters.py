"""Parameter-count and MoE metadata normalization helpers."""

from __future__ import annotations

import re


def _extract_size_hint_from_id(model_id: str | None) -> int | None:
    """Extract parameter size hint (in params) from model ID like 27B or 30B-A3B."""
    if not model_id:
        return None
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        max_b = max(float(m) for m in matches)
    except ValueError:
        return None
    if max_b <= 0:
        return None
    return int(max_b * 1e9)


def _extract_active_size_hint_from_id(model_id: str | None) -> int | None:
    """Extract MoE active parameter hint from names like 35B-A3B."""
    if not model_id:
        return None
    lower = model_id.lower()
    matches = re.findall(r"\d+(?:\.\d+)?b[-_]?a(\d+(?:\.\d+)?)b", lower)
    if not matches:
        return None
    try:
        max_b = max(float(m) for m in matches)
    except ValueError:
        return None
    if max_b <= 0:
        return None
    return int(max_b * 1e9)


def _is_quantized_repo_name(model_id: str) -> bool:
    """Detect quantized/non-base repository naming patterns."""
    lower = model_id.lower()
    return bool(re.search(r"(gptq|awq|bnb|4bit|int4|int8|fp8|gguf|quant)", lower))


def _lookup_curated_count(mapping: dict[str, int], model_id: str) -> int | None:
    value = mapping.get(model_id)
    if value is not None:
        return value

    model_id_folded = model_id.casefold()
    for key, value in mapping.items():
        if key.casefold() == model_id_folded:
            return value
    return None


def _resolve_moe_active_params(
    total_params: int,
    *model_refs: str | None,
) -> int | None:
    """Resolve active params from curated data or A*B naming hints."""
    for ref in model_refs:
        if not ref:
            continue
        active = _lookup_curated_count(_KNOWN_MOE_ACTIVE_PARAMS, ref)
        if active and active > 0:
            return active

    for ref in model_refs:
        active = _extract_active_size_hint_from_id(ref)
        if active and active > 0 and (total_params <= 0 or active < total_params):
            return active
    return None


def _normalize_param_count(
    extracted: int,
    model_id: str,
    base_model: str | None,
) -> int:
    """Normalize parameter count when metadata is inconsistent."""
    authoritative = _lookup_curated_count(_AUTHORITATIVE_PARAM_COUNTS, model_id)
    if authoritative and authoritative > 0:
        return authoritative
    known = _lookup_curated_count(_KNOWN_PARAM_COUNTS, model_id)
    if extracted <= 0:
        return known or extracted
    if known and extracted < int(known * 0.35):
        return known

    hints = [
        h
        for h in (
            _extract_size_hint_from_id(model_id),
            _extract_size_hint_from_id(base_model),
        )
        if h is not None
    ]
    if not hints:
        return extracted

    hinted = max(hints)
    if _is_quantized_repo_name(model_id):
        if extracted < int(hinted * 0.70):
            return hinted
    elif extracted < int(hinted * 0.35):
        return hinted

    return extracted


# Curated MoE active-parameter counts. Used when HF config lacks the
# `num_local_experts` / `num_experts_per_tok` keys that whichllm reads.
# Without this, frontier MoEs are scored as dense models which over-counts
# their VRAM cost and under-counts their inference speed.
_KNOWN_MOE_ACTIVE_PARAMS: dict[str, int] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 17_000_000_000,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 17_000_000_000,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 3_000_000_000,
    "Qwen/Qwen3-30B-A3B": 3_000_000_000,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 3_000_000_000,
    "Qwen/Qwen3-235B-A22B": 22_000_000_000,
    "Qwen/Qwen3.5-397B-A17B": 17_000_000_000,
    "deepseek-ai/DeepSeek-V3": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3-0324": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.1": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2-Exp": 37_000_000_000,
    "deepseek-ai/DeepSeek-R1": 37_000_000_000,
    "deepseek-ai/DeepSeek-R1-0528": 37_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro": 49_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 13_000_000_000,
    "zai-org/GLM-4.5": 32_000_000_000,
    "zai-org/GLM-4.5-Air": 12_000_000_000,
    "zai-org/GLM-4.6": 32_000_000_000,
    "zai-org/GLM-4.7": 32_000_000_000,
    "zai-org/GLM-4.7-Flash": 12_000_000_000,
    "zai-org/GLM-5": 40_000_000_000,
    "zai-org/GLM-5-FP8": 40_000_000_000,
    "zai-org/GLM-5.1": 40_000_000_000,
    "zai-org/GLM-5.1-FP8": 40_000_000_000,
    "moonshotai/Kimi-K2-Instruct": 32_000_000_000,
    "moonshotai/Kimi-K2-Thinking": 32_000_000_000,
    "MiniMaxAI/MiniMax-M2": 10_000_000_000,
    "MiniMaxAI/MiniMax-M2.5": 10_000_000_000,
    "XiaomiMiMo/MiMo-V2.5": 15_000_000_000,
    "XiaomiMiMo/MiMo-V2.5-Pro": 42_000_000_000,
    "XiaomiMiMo/MiMo-V2-Flash": 15_000_000_000,
    "google/gemma-4-26B-A4B-it": 3_800_000_000,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 3_000_000_000,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": 3_000_000_000,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 12_000_000_000,
    # OpenAI gpt-oss MoE family - 5B active for 20b/120b.
    "openai/gpt-oss-20b": 3_600_000_000,
    "openai/gpt-oss-120b": 5_100_000_000,
}

# Hardcoded parameter counts for frontier models that HF's API leaves with
# missing safetensors/gguf/config metadata. Used as a last-resort fallback
# inside _extract_param_count so these models still enter the cache and become
# rankable. Maintain only entries that lack a size hint in the model ID itself.
_KNOWN_PARAM_COUNTS: dict[str, int] = {
    "microsoft/phi-4": 14_700_000_000,
    "microsoft/Phi-4-mini-instruct": 3_800_000_000,
    "microsoft/Phi-4-multimodal-instruct": 5_600_000_000,
    "microsoft/Phi-4-reasoning": 14_700_000_000,
    "microsoft/Phi-4-reasoning-plus": 14_700_000_000,
    "openai/gpt-oss-20b": 20_000_000_000,
    "openai/gpt-oss-120b": 120_000_000_000,
    # IBM Granite 4.0 family
    "ibm-granite/granite-4.0-h-small": 32_000_000_000,
    "ibm-granite/granite-4.0-h-tiny": 7_000_000_000,
    "ibm-granite/granite-3.3-8b-instruct": 8_000_000_000,
    "ibm-granite/granite-3.3-2b-instruct": 2_000_000_000,
    # AllenAI Olmo-3
    "allenai/Olmo-3-7B-Instruct": 7_000_000_000,
    "allenai/Olmo-3-1025-7B": 7_000_000_000,
    # Llama 4 MoE totals: repo names advertise active size, but the total
    # weight footprint is much larger.
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 109_000_000_000,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 400_000_000_000,
    "deepseek-ai/DeepSeek-R1": 671_000_000_000,
    "deepseek-ai/DeepSeek-R1-0528": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3-0324": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3.1": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3.2": 685_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro": 1_600_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 284_000_000_000,
    "moonshotai/Kimi-K2-Instruct": 1_026_000_000_000,
    "moonshotai/Kimi-K2-Thinking": 1_026_000_000_000,
    "XiaomiMiMo/MiMo-V2.5": 310_000_000_000,
    "XiaomiMiMo/MiMo-V2.5-Pro": 1_020_000_000_000,
    "XiaomiMiMo/MiMo-V2-Flash": 309_000_000_000,
    "zai-org/GLM-4.5": 355_000_000_000,
    "zai-org/GLM-4.5-Air": 106_000_000_000,
    "zai-org/GLM-4.6": 355_000_000_000,
    "zai-org/GLM-4.7": 355_000_000_000,
    "zai-org/GLM-4.7-Flash": 30_000_000_000,
    "zai-org/GLM-5": 744_000_000_000,
    "zai-org/GLM-5-FP8": 744_000_000_000,
    "zai-org/GLM-5.1": 744_000_000_000,
    "zai-org/GLM-5.1-FP8": 744_000_000_000,
    "MiniMaxAI/MiniMax-M2": 230_000_000_000,
    "MiniMaxAI/MiniMax-M2.5": 230_000_000_000,
    "stepfun-ai/Step-3.5-Flash": 30_000_000_000,
}

# Curated counts that should win even when the HF API exposes safetensors
# metadata. Some mixed-precision MoEs publish compressed checkpoint tensor
# counts that understate the model-card capacity used for ranking and planning.
_AUTHORITATIVE_PARAM_COUNTS: dict[str, int] = {
    "deepseek-ai/DeepSeek-V4-Pro": 1_600_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 284_000_000_000,
}
