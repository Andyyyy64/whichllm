"""Model shape facts from a raw Hugging Face ``config.json``.

The Hub API's ``config`` expand only carries ``architectures`` and
``model_type``; layer counts, attention heads and expert sizes live in each
repository's own config.json. :func:`parse_model_shape` turns that file into
the few numbers memory estimates need. It returns None for any layout it does
not understand, so callers keep the parameter-count heuristic instead of
trusting a wrong shape.

Cache sizes follow llama.cpp's defaults: FP16 K/V tensors, sliding-window
layers sized to their window where the runtime honours it, latent K/V for
multi-head latent attention, and FP32 per-sequence state for recurrent layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from whichllm.models.sliding_window import _SWA_ARCH_DEFAULTS, _swa_key_from_arch

KV_BYTES_PER_ELEMENT = 2  # FP16 K/V, the llama.cpp and LM Studio default
STATE_BYTES_PER_ELEMENT = 4  # recurrent conv/SSM state is kept in FP32

_LAYER_KINDS = {
    "full_attention": "full",
    "attention": "full",
    "sliding_attention": "sliding",
    "linear_attention": "linear",
    "conv": "conv",
    "short_convolution": "conv",
    "mamba": "mamba",
}


@dataclass(frozen=True)
class ModelShape:
    """Shape facts for one checkpoint, aggregated over its layers."""

    model_type: str
    num_layers: int
    # K+V bytes per context token for layers that cache the whole context.
    full_kv_bytes_per_token: int
    # K+V bytes per token for sliding-window layers.
    sliding_kv_bytes_per_token: int = 0
    sliding_window: int | None = None
    # Whether mainline runtimes size sliding layers to the window. When False
    # sliding layers are counted at full context, which can only overstate.
    honors_sliding_window: bool = False
    # Per-sequence state of recurrent layers (linear attention, convolution,
    # Mamba). It does not grow with context.
    recurrent_state_bytes: int = 0
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    first_dense_layers: int = 0

    def kv_cache_bytes(self, context_length: int) -> int:
        """K/V and recurrent cache bytes for one sequence of this length."""
        context = max(int(context_length), 0)
        sliding_context = context
        if self.honors_sliding_window and self.sliding_window:
            sliding_context = min(context, self.sliding_window)
        return (
            self.full_kv_bytes_per_token * context
            + self.sliding_kv_bytes_per_token * sliding_context
            + self.recurrent_state_bytes
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> ModelShape | None:
        """Rebuild from :meth:`to_dict` output; None for anything malformed."""
        if not isinstance(data, dict):
            return None
        known = {f.name for f in fields(cls)}
        try:
            arch = cls(**{k: v for k, v in data.items() if k in known})
        except TypeError:
            return None
        if _positive_int(arch.num_layers) is None or not isinstance(
            arch.model_type, str
        ):
            return None
        return arch


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _text_config(config: dict) -> dict:
    """Multimodal checkpoints nest the language model under ``text_config``."""
    text = config.get("text_config")
    if isinstance(text, dict) and "num_hidden_layers" in text:
        return text
    return config


def _layer_kinds(text: dict, num_layers: int, swa_key: str | None) -> list[str] | None:
    declared = text.get("layer_types")
    if declared is not None:
        if not isinstance(declared, list) or len(declared) != num_layers:
            return None
        kinds = [_LAYER_KINDS.get(kind) for kind in declared]
        return None if None in kinds else kinds

    interval = _positive_int(text.get("full_attention_interval"))
    if interval:
        return [
            "full" if (i + 1) % interval == 0 else "linear" for i in range(num_layers)
        ]

    if swa_key is not None and text.get("use_sliding_window") is not False:
        pattern = _positive_int(text.get("sliding_window_pattern"))
        if pattern is None:
            pattern = round(1 / _SWA_ARCH_DEFAULTS[swa_key][1])
        return [
            "full" if (i + 1) % pattern == 0 else "sliding" for i in range(num_layers)
        ]

    return ["full"] * num_layers


def _attention_bytes_per_token(text: dict, global_layer: bool) -> int | None:
    """FP16 K+V bytes per token for one attention layer."""
    kv_lora_rank = _positive_int(text.get("kv_lora_rank"))
    if kv_lora_rank:
        # Multi-head latent attention: llama.cpp caches the compressed latent
        # plus the decoupled RoPE key once, and reads V from the same tensor.
        rope_dim = _positive_int(text.get("qk_rope_head_dim")) or 0
        return (kv_lora_rank + rope_dim) * KV_BYTES_PER_ELEMENT

    heads = _positive_int(text.get("num_attention_heads"))
    kv_heads = _positive_int(text.get("num_key_value_heads")) or heads
    hidden = _positive_int(text.get("hidden_size"))
    head_dim = _positive_int(text.get("head_dim"))
    if head_dim is None and hidden and heads and hidden % heads == 0:
        head_dim = hidden // heads
    if global_layer:
        kv_heads = _positive_int(text.get("num_global_key_value_heads")) or kv_heads
        head_dim = _positive_int(text.get("global_head_dim")) or head_dim
    if not (heads and kv_heads and head_dim) or kv_heads > heads:
        return None
    return 2 * kv_heads * head_dim * KV_BYTES_PER_ELEMENT


def _linear_attention_state_bytes(text: dict) -> int | None:
    """Gated DeltaNet conv + recurrent state (Qwen3-Next, Qwen3.5/3.6)."""
    key_heads = _positive_int(text.get("linear_num_key_heads"))
    value_heads = _positive_int(text.get("linear_num_value_heads"))
    key_dim = _positive_int(text.get("linear_key_head_dim"))
    value_dim = _positive_int(text.get("linear_value_head_dim"))
    kernel = _positive_int(text.get("linear_conv_kernel_dim"))
    if not (key_heads and value_heads and key_dim and value_dim and kernel):
        return None
    conv = (kernel - 1) * (2 * key_heads * key_dim + value_heads * value_dim)
    state = value_heads * key_dim * value_dim
    return (conv + state) * STATE_BYTES_PER_ELEMENT


def _conv_state_bytes(text: dict) -> int | None:
    """LFM2 short-convolution history: hidden × (L_cache - 1)."""
    hidden = _positive_int(text.get("hidden_size"))
    kernel = _positive_int(text.get("conv_L_cache"))
    if not hidden or not kernel or kernel < 2:
        return None
    return hidden * (kernel - 1) * STATE_BYTES_PER_ELEMENT


def _mamba_state_bytes(text: dict) -> int | None:
    """Mamba2 conv + SSM state (Granite 4.0 hybrids)."""
    heads = _positive_int(text.get("mamba_n_heads"))
    head_dim = _positive_int(text.get("mamba_d_head"))
    d_state = _positive_int(text.get("mamba_d_state"))
    d_conv = _positive_int(text.get("mamba_d_conv"))
    groups = _positive_int(text.get("mamba_n_groups")) or 1
    if not (heads and head_dim and d_state and d_conv):
        return None
    d_inner = heads * head_dim
    conv = (d_conv - 1) * (d_inner + 2 * groups * d_state)
    ssm = heads * head_dim * d_state
    return (conv + ssm) * STATE_BYTES_PER_ELEMENT


_STATE_BYTES = {
    "linear": _linear_attention_state_bytes,
    "conv": _conv_state_bytes,
    "mamba": _mamba_state_bytes,
}


def parse_model_shape(config: dict) -> ModelShape | None:
    """Parse a raw config.json; None when the layout is not understood."""
    if not isinstance(config, dict):
        return None
    text = _text_config(config)
    model_type = text.get("model_type") or config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        return None
    num_layers = _positive_int(text.get("num_hidden_layers"))
    if num_layers is None or num_layers > 10_000:
        return None

    swa_key = _swa_key_from_arch(model_type) or _swa_key_from_arch(
        config.get("model_type")
    )
    kinds = _layer_kinds(text, num_layers, swa_key)
    if kinds is None:
        return None

    sliding_window = _positive_int(text.get("sliding_window"))
    if "sliding" in kinds and sliding_window is None and swa_key is not None:
        sliding_window = _SWA_ARCH_DEFAULTS[swa_key][0]
    if "sliding" in kinds and sliding_window is None:
        return None

    full_bytes = sliding_bytes = state_bytes = 0
    per_kind: dict[str, int | None] = {}
    for kind in kinds:
        if kind not in per_kind:
            if kind in _STATE_BYTES:
                per_kind[kind] = _STATE_BYTES[kind](text)
            else:
                per_kind[kind] = _attention_bytes_per_token(
                    text, global_layer=kind == "full" and "sliding" in kinds
                )
        size = per_kind[kind]
        if size is None:
            return None
        if kind == "full":
            full_bytes += size
        elif kind == "sliding":
            sliding_bytes += size
        else:
            state_bytes += size

    # Explicit per-layer sliding layouts come from architectures built around
    # interleaved SWA, so supporting runtimes size those layers to the window.
    # A bare ``sliding_window`` (Mistral v0.1, Qwen2) is often ignored and so
    # does not count.
    declared_sliding = isinstance(text.get("layer_types"), list) and "sliding" in kinds
    honors_sliding_window = "sliding" in kinds and (
        swa_key is not None or declared_sliding
    )

    num_experts = next(
        (
            n
            for n in (
                _positive_int(text.get(key))
                for key in ("n_routed_experts", "num_experts", "num_local_experts")
            )
            if n
        ),
        None,
    )
    moe_intermediate = _positive_int(text.get("moe_intermediate_size"))
    shared = _positive_int(text.get("shared_expert_intermediate_size"))
    n_shared = _positive_int(text.get("n_shared_experts"))
    if shared is None and n_shared and moe_intermediate:
        shared = n_shared * moe_intermediate

    return ModelShape(
        model_type=model_type,
        num_layers=num_layers,
        full_kv_bytes_per_token=full_bytes,
        sliding_kv_bytes_per_token=sliding_bytes,
        sliding_window=sliding_window if "sliding" in kinds else None,
        honors_sliding_window=honors_sliding_window,
        recurrent_state_bytes=state_bytes,
        hidden_size=_positive_int(text.get("hidden_size")),
        intermediate_size=_positive_int(text.get("intermediate_size")),
        num_experts=num_experts,
        experts_per_token=_positive_int(text.get("num_experts_per_tok"))
        if num_experts
        else None,
        moe_intermediate_size=moe_intermediate if num_experts else None,
        shared_expert_intermediate_size=shared if num_experts else None,
        first_dense_layers=_positive_int(text.get("first_k_dense_replace")) or 0,
    )
