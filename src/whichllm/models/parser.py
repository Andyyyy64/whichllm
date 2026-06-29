"""Parse HuggingFace API model payloads into ModelInfo objects."""

from __future__ import annotations

import statistics

from whichllm.models.gguf import _extract_gguf_variants
from whichllm.models.parameters import (
    _AUTHORITATIVE_PARAM_COUNTS,
    _KNOWN_PARAM_COUNTS,
    _extract_size_hint_from_id,
    _lookup_curated_count,
    _normalize_param_count,
    _resolve_moe_active_params,
)
from whichllm.models.sliding_window import _resolve_sliding_window
from whichllm.models.types import ModelInfo

_GENERAL_EVAL_KEYWORDS = (
    "mmlu",
    "gpqa",
    "gsm8k",
    "hellaswag",
    "arc",
    "bbh",
    "ifeval",
    "truthfulqa",
    "ceval",
    "cmmlu",
)


def _extract_published_at(data: dict) -> str | None:
    """Extract the best published timestamp candidate from an API response."""
    created = data.get("createdAt")
    if isinstance(created, str) and created:
        return created
    modified = data.get("lastModified")
    if isinstance(modified, str) and modified:
        return modified
    return None


def _normalize_eval_value(raw: object) -> float | None:
    """Convert eval value to a comparable 0-100 score."""
    if not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0:
        return None
    if value <= 1.0:
        value *= 100.0
    if value > 100.0:
        return None
    return value


def _is_general_eval_entry(entry: dict) -> bool:
    """Keep eval entries that are broadly useful for general chat quality."""
    data = entry.get("data")
    if not isinstance(data, dict):
        return False

    notes = str(data.get("notes", "")).lower()
    if "with tools" in notes:
        return False

    dataset = data.get("dataset")
    dataset_id = ""
    task_id = ""
    if isinstance(dataset, dict):
        dataset_id = str(dataset.get("id", "")).lower()
        task_id = str(dataset.get("task_id", "")).lower()
    filename = str(entry.get("filename", "")).lower()

    return any(
        k in dataset_id or k in task_id or k in filename for k in _GENERAL_EVAL_KEYWORDS
    )


def _extract_hf_eval_score(data: dict) -> float | None:
    """Extract conservative aggregate score from HF evalResults."""
    eval_results = data.get("evalResults")
    if not isinstance(eval_results, list) or not eval_results:
        return None

    values: list[float] = []
    for entry in eval_results:
        if not isinstance(entry, dict):
            continue
        if not _is_general_eval_entry(entry):
            continue
        data_obj = entry.get("data")
        if not isinstance(data_obj, dict):
            continue
        normalized = _normalize_eval_value(data_obj.get("value"))
        if normalized is not None:
            values.append(normalized)

    if not values:
        return None
    return round(statistics.median(values), 1)


def _extract_param_count(model_data: dict) -> int:
    """Extract parameter count from model data.

    Resolution order:
      1. authoritative model-card overrides for known mixed-precision MoEs
      2. safetensors metadata
      3. gguf metadata
      4. config estimate
      5. curated known counts
      6. name-based size hint

    Returns 0 if none of the above succeed.
    """
    model_id = model_data.get("id", "") or ""
    authoritative = _lookup_curated_count(_AUTHORITATIVE_PARAM_COUNTS, model_id)
    if authoritative and authoritative > 0:
        return authoritative

    safetensors = model_data.get("safetensors")
    if safetensors and isinstance(safetensors, dict):
        params = safetensors.get("total")
        if params:
            return int(params)
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict):
            total = sum(parameters.values())
            if total > 0:
                return total

    gguf_meta = model_data.get("gguf", {}) or {}
    if isinstance(gguf_meta, dict):
        total = gguf_meta.get("total")
        if total and total > 0:
            return int(total)

    config = model_data.get("config", {}) or {}
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if hidden and layers and vocab:
        return 12 * layers * hidden * hidden + vocab * hidden * 2

    known = _lookup_curated_count(_KNOWN_PARAM_COUNTS, model_id)
    if known and known > 0:
        return known
    name_hint = _extract_size_hint_from_id(model_id)
    if name_hint and name_hint > 0:
        return name_hint

    return 0


def _extract_architecture(config: dict) -> str:
    """Extract architecture string from config."""
    arch_list = config.get("architectures", [])
    if arch_list:
        arch = arch_list[0].lower()
        for name in [
            "llama",
            "qwen2",
            "mistral",
            "mixtral",
            "gemma",
            "phi",
            "starcoder",
            "command",
            "deepseek",
        ]:
            if name in arch:
                return name
        return arch.replace("forcausallm", "").replace("forconditionalgeneration", "")
    model_type = config.get("model_type", "")
    return model_type.lower()


def _extract_base_model(card_data: dict) -> str | None:
    base_model_raw = card_data.get("base_model")
    if isinstance(base_model_raw, str):
        return base_model_raw
    if isinstance(base_model_raw, list) and base_model_raw:
        return base_model_raw[0]
    return None


def _resolve_active_params(
    config: dict,
    param_count: int,
    model_id: str,
    base_model: str | None,
) -> tuple[bool, int | None]:
    num_experts = 0
    for k in (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_moe_experts",
        "n_local_experts",
    ):
        v = config.get(k, 0)
        if isinstance(v, int) and v > num_experts:
            num_experts = v

    experts_per_tok = 0
    for k in (
        "num_experts_per_tok",
        "moe_topk",
        "moe_top_k",
        "num_experts_per_token",
        "top_k",
    ):
        v = config.get(k, 0)
        if isinstance(v, int) and v > experts_per_tok:
            experts_per_tok = v

    known_moe_active = _resolve_moe_active_params(param_count, model_id, base_model)
    is_moe = num_experts > 0 or known_moe_active is not None
    active_params = None
    if is_moe:
        if known_moe_active is not None:
            active_params = known_moe_active
        elif num_experts > 0:
            ept = experts_per_tok if experts_per_tok > 0 else 2
            active_ratio = ept / num_experts
            expert_fraction = 0.6
            active_params = int(
                param_count * (1 - expert_fraction + expert_fraction * active_ratio)
            )
    return is_moe, active_params


def _parse_model(data: dict) -> ModelInfo | None:
    """Parse HF API response into ModelInfo."""
    model_id = data.get("id", "")
    if not model_id:
        return None

    config = data.get("config", {}) or {}
    card_data = data.get("cardData", {}) or {}
    base_model = _extract_base_model(card_data)

    param_count = _extract_param_count(data)
    param_count = _normalize_param_count(param_count, model_id, base_model)
    if param_count == 0:
        return None

    is_moe, active_params = _resolve_active_params(
        config, param_count, model_id, base_model
    )
    gguf_variants = _extract_gguf_variants(data, param_count)

    architecture = _extract_architecture(config)
    gguf_meta = data.get("gguf", {}) or {}
    if not architecture and isinstance(gguf_meta, dict):
        architecture = gguf_meta.get("architecture", "")

    context_length = config.get("max_position_embeddings") or config.get(
        "max_sequence_length"
    )
    if not context_length and isinstance(gguf_meta, dict):
        context_length = gguf_meta.get("context_length")

    gguf_arch = gguf_meta.get("architecture") if isinstance(gguf_meta, dict) else None
    sliding_window, swa_global_ratio = _resolve_sliding_window(
        config, model_id, gguf_arch
    )

    benchmark_scores: dict[str, float] = {}
    eval_score = _extract_hf_eval_score(data)
    if eval_score is not None:
        benchmark_scores["hf_eval"] = eval_score

    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name=model_id.split("/")[-1],
        parameter_count=param_count,
        parameter_count_active=active_params,
        architecture=architecture,
        is_moe=is_moe,
        context_length=context_length,
        license=card_data.get("license"),
        published_at=_extract_published_at(data),
        downloads=data.get("downloads", 0),
        likes=data.get("likes", 0),
        gguf_variants=gguf_variants,
        benchmark_scores=benchmark_scores,
        base_model=base_model,
        sliding_window=sliding_window,
        sliding_window_global_ratio=swa_global_ratio,
    )
