"""VRAM usage estimation."""

from __future__ import annotations

from local_llm_checker.constants import FRAMEWORK_OVERHEAD_BYTES
from local_llm_checker.models.types import GGUFVariant, ModelInfo


def estimate_kv_cache(model: ModelInfo, context_length: int) -> int:
    """Estimate KV cache size in bytes for a given context length.

    KV cache per layer = 2 * num_kv_heads * head_dim * context_length * 2 bytes (FP16)
    Simplified: ~0.5MB per billion params per 1K context tokens.
    """
    # Rough estimate based on parameter count
    # Typical: 2 bytes per element, 2 (K+V), context_length tokens
    # For a 7B model at 4096 context: ~256MB
    params_b = model.parameter_count / 1e9
    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * 0.5 * 1024 * 1024)  # ~0.5MB per B per 1K
    return max(kv_bytes, 0)


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
) -> int:
    """Estimate total VRAM required to run a model."""
    # Model weights
    if variant:
        weights = variant.file_size_bytes
    else:
        weights = model.parameter_count * 2  # FP16 fallback

    # KV cache
    kv_cache = estimate_kv_cache(model, context_length)

    # Activation overhead: ~0.5GB + 0.08 bytes per param
    activation = 500_000_000 + int(model.parameter_count * 0.08)

    # Framework overhead
    framework = FRAMEWORK_OVERHEAD_BYTES

    return weights + kv_cache + activation + framework
