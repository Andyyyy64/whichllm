"""The longest context a ranked row holds without changing how it runs."""

from __future__ import annotations

from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import ModelInfo

# llama.cpp contexts are set in 1K steps, and nothing here runs past 1M tokens.
_STEP = 1024
_CEILING = 1024 * 1024
# Without the model's own limit the search stops here, and the cell says "+"
# rather than promise a length the model was never trained for.
_UNKNOWN_CEILING = 128 * 1024


def model_context_limit(model: ModelInfo) -> int | None:
    """The context the model was built for: the Hub metadata, else config.json."""
    if model.context_length:
        return model.context_length
    return model.shape.max_context if model.shape else None


def _placed_the_same(result: CompatibilityResult, check: CompatibilityResult) -> bool:
    kind = result.placement.kind if result.placement else None
    other = check.placement.kind if check.placement else None
    return (
        check.can_run
        and check.fit_type == result.fit_type
        and other == kind
        and check.tensor_parallel_size == result.tensor_parallel_size
        and (check.paged_experts_resident is None)
        == (result.paged_experts_resident is None)
    )


def max_context_tokens(
    result: CompatibilityResult, hardware: HardwareInfo, context_length: int
) -> int:
    """Longest context that keeps ``result``'s placement, up to the model's limit.

    A full-GPU row stays on the GPU, and a row with experts or layers in RAM keeps
    them there without running out of memory. Memory only grows with context, so
    a binary search over 1K steps finds the edge. ``hardware`` must carry the
    budgets the ranking used.
    """
    known = model_context_limit(result.model)
    limit = min(known, _CEILING) if known else max(_UNKNOWN_CEILING, context_length)
    if limit <= context_length:
        return limit

    def holds(tokens: int) -> bool:
        check = check_compatibility(result.model, result.gguf_variant, hardware, tokens)
        return _placed_the_same(result, check)

    if holds(limit):
        return limit
    # The row was ranked at context_length, so that much holds and limit does not.
    low, high = context_length // _STEP, limit // _STEP
    while high - low > 1:
        middle = (low + high) // 2
        if holds(middle * _STEP):
            low = middle
        else:
            high = middle
    return max(low * _STEP, context_length)


def format_context(tokens: int) -> str:
    """1024-based like llama.cpp's own sizes: 131072 is 128k."""
    if tokens >= 1024 * 1024:
        return f"{tokens / (1024 * 1024):g}M"
    return f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)


def context_cell(result: CompatibilityResult) -> str:
    """ "262k→47k" when the fit runs out first, the limit alone when it does not.

    Without a known limit, "?→18k" gives the edge and "128k+" says only that
    the search ceiling fits.
    """
    if result.max_context_tokens is None:
        return "-"
    fits = format_context(result.max_context_tokens)
    limit = model_context_limit(result.model)
    if limit is None:
        if result.max_context_tokens >= _UNKNOWN_CEILING:
            return f"{fits}+"
        return f"?→{fits}"
    if result.max_context_tokens >= limit:
        return format_context(limit)
    return f"{format_context(limit)}→{fits}"


__all__ = [
    "context_cell",
    "format_context",
    "max_context_tokens",
    "model_context_limit",
]
