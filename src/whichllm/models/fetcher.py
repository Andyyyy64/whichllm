"""HuggingFace Hub API model fetcher."""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from whichllm.constants import QUANT_BYTES_PER_WEIGHT
from whichllm.models.types import GGUFVariant, ModelInfo

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"
_GGUF_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _extract_published_at(data: dict) -> str | None:
    """APIレスポンスから公開日時候補を取り出す。"""
    created = data.get("createdAt")
    if isinstance(created, str) and created:
        return created
    modified = data.get("lastModified")
    if isinstance(modified, str) and modified:
        return modified
    return None


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


def _is_quantized_repo_name(model_id: str) -> bool:
    """Detect quantized/non-base repository naming patterns."""
    lower = model_id.lower()
    return bool(
        re.search(r"(gptq|awq|bnb|4bit|int4|int8|fp8|gguf|quant)", lower)
    )


def _normalize_param_count(
    extracted: int,
    model_id: str,
    base_model: str | None,
) -> int:
    """Normalize parameter count when metadata is inconsistent."""
    if extracted <= 0:
        return extracted

    hints = [
        h for h in (
            _extract_size_hint_from_id(model_id),
            _extract_size_hint_from_id(base_model),
        ) if h is not None
    ]
    if not hints:
        return extracted

    hinted = max(hints)
    if _is_quantized_repo_name(model_id):
        # 量子化派生モデルはsafetensors metadataが縮んだ値になることがある
        if extracted < int(hinted * 0.70):
            return hinted
    elif extracted < int(hinted * 0.35):
        return hinted

    return extracted


def _extract_quant_type(filename: str) -> str:
    """Extract quantization type from GGUF filename."""
    # Common patterns: model-Q4_K_M.gguf, model.Q4_K_M.gguf
    patterns = [
        r"[.-](Q\d+_K_[SMLA])",
        r"[.-](Q\d+_\d+)",
        r"[.-](Q\d+_K)",
        r"[.-](IQ\d+_\w+)",
        r"[.-](F16|FP16|BF16|F32)",
    ]
    upper = filename.upper()
    for pattern in patterns:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)
    return "unknown"


def _estimate_gguf_size(param_count: int, quant_type: str) -> int:
    """Estimate GGUF file size from parameter count and quantization type."""
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant_type.upper(), 0.5625)  # default Q4_K_M
    return int(param_count * bpw)


def _extract_param_count(model_data: dict) -> int:
    """Extract parameter count from model data."""
    # Try safetensors metadata first
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

    # Try gguf metadata
    gguf_meta = model_data.get("gguf", {}) or {}
    if isinstance(gguf_meta, dict):
        total = gguf_meta.get("total")
        if total and total > 0:
            return int(total)

    # Try config
    config = model_data.get("config", {}) or {}
    # Estimate from hidden_size and num_layers if available
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if hidden and layers and vocab:
        # Rough: 12 * layers * hidden^2 + vocab * hidden * 2
        return 12 * layers * hidden * hidden + vocab * hidden * 2

    return 0


def _extract_architecture(config: dict) -> str:
    """Extract architecture string from config."""
    arch_list = config.get("architectures", [])
    if arch_list:
        arch = arch_list[0].lower()
        # Normalize
        for name in ["llama", "qwen2", "mistral", "mixtral", "gemma", "phi", "starcoder", "command", "deepseek"]:
            if name in arch:
                return name
        return arch.replace("forcausallm", "").replace("forconditionalgeneration", "")
    model_type = config.get("model_type", "")
    return model_type.lower()


def _parse_model(data: dict) -> ModelInfo | None:
    """Parse HF API response into ModelInfo."""
    model_id = data.get("id", "")
    if not model_id:
        return None

    config = data.get("config", {}) or {}
    card_data = data.get("cardData", {}) or {}

    # Base model from card data
    base_model_raw = card_data.get("base_model")
    base_model = None
    if isinstance(base_model_raw, str):
        base_model = base_model_raw
    elif isinstance(base_model_raw, list) and base_model_raw:
        base_model = base_model_raw[0]

    param_count = _extract_param_count(data)
    param_count = _normalize_param_count(param_count, model_id, base_model)
    if param_count == 0:
        return None

    # MoE detection
    num_experts = config.get("num_local_experts", 0)
    is_moe = num_experts > 0
    active_params = None
    if is_moe:
        experts_per_tok = config.get("num_experts_per_tok", 2)
        if num_experts > 0:
            active_ratio = experts_per_tok / num_experts
            # Active params = non-expert params + expert_ratio * expert_params
            # Rough: active ≈ param_count * (1 - expert_fraction + expert_fraction * active_ratio)
            # For simplicity: active ≈ param_count * active_ratio (for heavily MoE models)
            # Better estimate: typically ~40-50% of MoE total are in experts
            expert_fraction = 0.6  # rough estimate
            active_params = int(
                param_count * (1 - expert_fraction + expert_fraction * active_ratio)
            )

    # GGUF variants from siblings
    quant_sizes: dict[str, int] = {}
    quant_first_filename: dict[str, str] = {}
    siblings = data.get("siblings", []) or []
    for sib in siblings:
        fname = sib.get("rfilename", "")
        if not fname.endswith(".gguf") or fname.startswith("."):
            continue
        quant = _extract_quant_type(fname)
        if quant == "unknown":
            continue
        size = sib.get("size", 0)
        if not isinstance(size, int) or size < 0:
            size = 0

        # 分割GGUFは量子化ごとに合算して1候補として扱う。
        quant_sizes[quant] = quant_sizes.get(quant, 0) + size
        if quant not in quant_first_filename or _GGUF_SPLIT_RE.search(quant_first_filename[quant]):
            quant_first_filename[quant] = fname

    gguf_variants = []
    for quant, total_size in quant_sizes.items():
        if total_size <= 0:
            total_size = _estimate_gguf_size(param_count, quant)
        gguf_variants.append(
            GGUFVariant(
                filename=quant_first_filename[quant],
                quant_type=quant,
                file_size_bytes=total_size,
            )
        )

    architecture = _extract_architecture(config)
    # Fallback architecture from gguf metadata
    gguf_meta = data.get("gguf", {}) or {}
    if not architecture and isinstance(gguf_meta, dict):
        architecture = gguf_meta.get("architecture", "")

    context_length = config.get("max_position_embeddings") or config.get(
        "max_sequence_length"
    )
    # Fallback context length from gguf metadata
    if not context_length and isinstance(gguf_meta, dict):
        context_length = gguf_meta.get("context_length")

    return ModelInfo(
        id=model_id,
        family_id=model_id,  # will be set by grouper
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
        base_model=base_model,
    )


async def fetch_models(limit: int = 300, include_vision: bool = True) -> list[ModelInfo]:
    """Fetch popular models from HuggingFace Hub."""
    models: list[ModelInfo] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch top text-generation models
        params = {
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit),
            "expand[]": ["config", "safetensors", "gguf", "cardData", "siblings"],
        }
        logger.debug(f"Fetching models from HF API (limit={limit})")
        resp = await client.get(f"{HF_API_BASE}/models", params=params)
        resp.raise_for_status()
        data_list = resp.json()
        for data in data_list:
            model = _parse_model(data)
            if model:
                models.append(model)

        # Also fetch GGUF-specific models
        gguf_params = {
            "pipeline_tag": "text-generation",
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit),
            "expand[]": ["config", "safetensors", "gguf", "cardData", "siblings"],
        }
        logger.debug("Fetching GGUF models from HF API")
        resp = await client.get(f"{HF_API_BASE}/models", params=gguf_params)
        resp.raise_for_status()
        gguf_data_list = resp.json()

        seen_ids = {m.id for m in models}
        for data in gguf_data_list:
            if data.get("id") not in seen_ids:
                model = _parse_model(data)
                if model:
                    models.append(model)
                    seen_ids.add(model.id)

        # Fetch recently updated GGUF models (catch new releases)
        recent_params = {
            "pipeline_tag": "text-generation",
            "filter": "gguf",
            "sort": "lastModified",
            "direction": "-1",
            "limit": str(limit),
            "expand[]": ["config", "safetensors", "gguf", "cardData", "siblings"],
        }
        logger.debug("Fetching recent GGUF models from HF API")
        resp = await client.get(f"{HF_API_BASE}/models", params=recent_params)
        resp.raise_for_status()
        recent_data_list = resp.json()

        for data in recent_data_list:
            if data.get("id") not in seen_ids:
                model = _parse_model(data)
                if model:
                    models.append(model)
                    seen_ids.add(model.id)

        if include_vision:
            # 画像入力系は用途が異なるため、明示的に有効化されたときだけ取得する。
            for pipeline_tag in ("image-text-to-text",):
                mm_params = {
                    "pipeline_tag": pipeline_tag,
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": str(limit),
                    "expand[]": ["config", "safetensors", "gguf", "cardData", "siblings"],
                }
                logger.debug(f"Fetching {pipeline_tag} models from HF API")
                resp = await client.get(f"{HF_API_BASE}/models", params=mm_params)
                resp.raise_for_status()
                mm_data_list = resp.json()

                for data in mm_data_list:
                    if data.get("id") not in seen_ids:
                        model = _parse_model(data)
                        if model:
                            models.append(model)
                            seen_ids.add(model.id)

    logger.debug(f"Fetched {len(models)} models total")
    return models


def models_to_dicts(models: list[ModelInfo]) -> list[dict]:
    """Serialize models to dicts for caching."""
    result = []
    for m in models:
        result.append(
            {
                "id": m.id,
                "family_id": m.family_id,
                "name": m.name,
                "parameter_count": m.parameter_count,
                "parameter_count_active": m.parameter_count_active,
                "architecture": m.architecture,
                "is_moe": m.is_moe,
                "context_length": m.context_length,
                "license": m.license,
                "published_at": m.published_at,
                "downloads": m.downloads,
                "likes": m.likes,
                "gguf_variants": [
                    {
                        "filename": v.filename,
                        "quant_type": v.quant_type,
                        "file_size_bytes": v.file_size_bytes,
                    }
                    for v in m.gguf_variants
                ],
                "benchmark_scores": m.benchmark_scores,
                "base_model": m.base_model,
            }
        )
    return result


def dicts_to_models(data: list[dict]) -> list[ModelInfo]:
    """Deserialize models from cached dicts."""
    models = []
    for d in data:
        base_model = d.get("base_model")
        param_count = _normalize_param_count(
            d["parameter_count"],
            d["id"],
            base_model,
        )
        models.append(
            ModelInfo(
                id=d["id"],
                family_id=d.get("family_id", d["id"]),
                name=d["name"],
                parameter_count=param_count,
                parameter_count_active=d.get("parameter_count_active"),
                architecture=d.get("architecture", ""),
                is_moe=d.get("is_moe", False),
                context_length=d.get("context_length"),
                license=d.get("license"),
                published_at=d.get("published_at"),
                downloads=d.get("downloads", 0),
                likes=d.get("likes", 0),
                gguf_variants=[
                    GGUFVariant(
                        filename=v["filename"],
                        quant_type=v["quant_type"],
                        file_size_bytes=v["file_size_bytes"],
                    )
                    for v in d.get("gguf_variants", [])
                ],
                benchmark_scores=d.get("benchmark_scores", {}),
                base_model=base_model,
            )
        )
    return models


async def fetch_model_published_at(model_ids: list[str]) -> dict[str, str]:
    """Fetch published timestamps for specific model IDs."""
    unique_ids = sorted({m for m in model_ids if m})
    if not unique_ids:
        return {}

    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = [
            client.get(
                f"{HF_API_BASE}/models/{model_id}",
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
            logger.debug("Failed to fetch model detail for %s: HTTP %s", model_id, resp.status_code)
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        published_at = _extract_published_at(data)
        if published_at:
            result[model_id] = published_at
    return result
