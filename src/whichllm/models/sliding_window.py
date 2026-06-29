"""Sliding-window-attention metadata resolution."""

from __future__ import annotations


# Sliding-window-attention (SWA) registry. We only model SWA KV-cache savings
# for architectures whose mainline runtimes actually honor interleaved SWA
# (llama.cpp's ISWA path, MLX). Each entry is (default_window_tokens,
# global_layer_ratio) where the ratio is the fraction of layers that use full
# (global) attention. Models outside this allowlist keep full-context KV so
# estimates stay conservative.
_SWA_ARCH_DEFAULTS: dict[str, tuple[int, float]] = {
    "gemma2": (4096, 0.5),
    "gemma3": (1024, 1.0 / 6.0),
    "gpt_oss": (128, 0.5),
    "cohere2": (4096, 0.25),
}

# Map the many spellings an arch string can take (HF model_type, the
# ForCausalLM/ForConditionalGeneration class prefix, and GGUF metadata) onto a
# canonical key in _SWA_ARCH_DEFAULTS.
_SWA_ARCH_ALIASES: dict[str, str] = {
    "gemma2": "gemma2",
    "gemma2_text": "gemma2",
    "gemma3": "gemma3",
    "gemma3_text": "gemma3",
    "gpt_oss": "gpt_oss",
    "gptoss": "gpt_oss",
    "cohere2": "cohere2",
}


def _swa_key_from_arch(arch: str | None) -> str | None:
    """Resolve an arch string (model_type / class / gguf metadata) to a key."""
    if not arch:
        return None
    arch = arch.lower()
    if arch in _SWA_ARCH_ALIASES:
        return _SWA_ARCH_ALIASES[arch]
    stripped = arch.replace("forcausallm", "").replace("forconditionalgeneration", "")
    if stripped in _SWA_ARCH_ALIASES:
        return _SWA_ARCH_ALIASES[stripped]
    return None


def _swa_arch_key(config: dict, model_id: str, gguf_arch: str | None) -> str | None:
    """Identify the SWA architecture key for a model, or None if not honored.

    Relies on authoritative metadata only: raw HF config model_type /
    architectures and GGUF metadata architecture. When none are present the
    model is left unhonored (full-context estimate), since a false positive
    would under-count VRAM.
    """
    model_type = config.get("model_type")
    key = _swa_key_from_arch(model_type if isinstance(model_type, str) else None)
    if key:
        return key

    arch_list = config.get("architectures") or []
    if arch_list and isinstance(arch_list[0], str):
        key = _swa_key_from_arch(arch_list[0])
        if key:
            return key

    return _swa_key_from_arch(gguf_arch)


def _resolve_sliding_window(
    config: dict, model_id: str, gguf_arch: str | None = None
) -> tuple[int | None, float | None]:
    """Resolve (sliding_window, global_ratio) for honored SWA architectures.

    Returns (None, None) for every model outside the allowlist so the KV
    estimate stays at full context (conservative).
    """
    if config.get("use_sliding_window") is False:
        return None, None

    key = _swa_arch_key(config, model_id, gguf_arch)
    if key is None:
        return None, None

    default_window, default_ratio = _SWA_ARCH_DEFAULTS[key]

    window = config.get("sliding_window")
    if not isinstance(window, int) or window <= 0:
        window = default_window

    pattern = config.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0:
        global_ratio = 1.0 / pattern
    else:
        global_ratio = default_ratio

    return window, global_ratio
