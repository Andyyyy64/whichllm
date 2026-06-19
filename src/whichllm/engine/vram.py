"""VRAM usage estimation."""

from __future__ import annotations

from whichllm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.models.types import GGUFVariant, ModelInfo

# Empirical KV-cache coefficient: bytes per B-active-param per K-context-token
# for FP16 K/V tensors. Calibrated against published llama.cpp memory reports
# for Qwen2.5-7B (0.45 GB @ 8K), Qwen3-32B (3.1 GB @ 32K), and Llama-3.1-70B
# (5.4 GB @ 32K with GQA), then bumped slightly because real llama.cpp also
# allocates a graph-compute buffer proportional to KV size.
_KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024  # 3.5 MB

# MoE attention scales with the *attention-layer count*, which is roughly
# proportional to active_params * this multiplier. For Qwen3-Next-80B-A3B
# (3B active, 48 layers), the multiplier lands near 4.
_MOE_ATTENTION_PARAM_MULTIPLIER = 4.0


def _effective_kv_context(model: ModelInfo, context_length: int) -> float:
    """Context length that actually contributes to the KV cache.

    For sliding-window-attention (SWA) models, local-attention layers only keep
    the last ``sliding_window`` tokens, so their KV footprint plateaus once the
    request exceeds the window. Hybrid models interleave a fraction of global
    (full-context) layers; that fraction is ``sliding_window_global_ratio``.

    Effective context blends the two layer types::

        global_ratio * ctx + (1 - global_ratio) * min(ctx, window)

    This is only applied for architectures whose mainline runtimes honor SWA
    (the fetcher leaves ``sliding_window`` ``None`` otherwise), and it can only
    ever lower the estimate — so a model that does not advertise an honored
    window keeps the full-context KV figure and stays conservative.
    """
    window = model.sliding_window
    ratio = model.sliding_window_global_ratio
    if not window or window <= 0 or ratio is None:
        return float(context_length)
    ratio = min(max(ratio, 0.0), 1.0)
    windowed = min(context_length, window)
    return ratio * context_length + (1.0 - ratio) * windowed


def estimate_kv_cache(model: ModelInfo, context_length: int) -> int:
    """Estimate KV cache size in bytes for a given context length.

    Dense models: KV ≈ 3 MB × params_b × ctx_k (FP16 K+V across all layers).
    MoE models: scale from active params × an empirical multiplier because
    attention shares across experts.
    Sliding-window models cap local-layer KV at the window size (see
    :func:`_effective_kv_context`).
    """
    if model.is_moe and model.parameter_count_active:
        # Active-params × MoE multiplier gives a reasonable proxy for the
        # attention-layer footprint without needing config.num_hidden_layers.
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * _MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    effective_ctx = _effective_kv_context(model, context_length)
    ctx_k = effective_ctx / 1024
    kv_bytes = int(params_b * ctx_k * _KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def _activation_bytes(model: ModelInfo, context_length: int) -> int:
    """Activation/scratch buffer size.

    Empirically activation memory grows mildly with both model size and
    context length. The prior constant-plus-linear-param formula
    over-counted small models and under-counted long contexts.
    """
    # Use effective (active for MoE) size as the param-dependent base
    if model.is_moe and model.parameter_count_active:
        effective_p = model.parameter_count_active
    else:
        effective_p = model.parameter_count

    base = 400_000_000  # 400 MB framework activation floor
    param_term = int(effective_p * 0.08)  # ~0.08 byte/param
    ctx_term = int((context_length / 4096) * 150_000_000)  # +150 MB per 4K
    return base + param_term + ctx_term


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
) -> int:
    """Estimate total VRAM required to run a model."""
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, context_length)
    activation = _activation_bytes(model, context_length)
    framework = FRAMEWORK_OVERHEAD_BYTES
    return weights + kv_cache + activation + framework
