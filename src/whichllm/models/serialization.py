"""ModelInfo cache serialization helpers."""

from __future__ import annotations

from whichllm.models.parameters import (
    _normalize_param_count,
    _resolve_moe_active_params,
)
from whichllm.models.types import GGUFVariant, ModelInfo


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
                "sliding_window": m.sliding_window,
                "sliding_window_global_ratio": m.sliding_window_global_ratio,
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
        active_params = _resolve_moe_active_params(
            param_count,
            d["id"],
            base_model,
            d.get("name"),
            d.get("architecture"),
        )
        if active_params is None:
            active_params = d.get("parameter_count_active")
        models.append(
            ModelInfo(
                id=d["id"],
                family_id=d.get("family_id", d["id"]),
                name=d["name"],
                parameter_count=param_count,
                parameter_count_active=active_params,
                architecture=d.get("architecture", ""),
                is_moe=d.get("is_moe", False) or active_params is not None,
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
                sliding_window=d.get("sliding_window"),
                sliding_window_global_ratio=d.get("sliding_window_global_ratio"),
            )
        )
    return models
