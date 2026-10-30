"""One ranking path for every command that picks models for a machine.

``whichllm``, ``run``, ``snippet``, ``upgrade`` and the golden set all rank
through here, so they agree on the pick: same memory budgets, same size floor,
same evidence rules.
"""

from __future__ import annotations

from whichllm.cli_hardware import (
    _apply_memory_budgets,
    _apply_ram_bandwidth,
    _auto_min_params_for_profile,
)
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import ModelInfo


def apply_default_budgets(hardware: HardwareInfo) -> HardwareInfo:
    """What a plain ``whichllm`` applies to detected hardware without flags."""
    _apply_ram_bandwidth(hardware, None)
    return _apply_memory_budgets(hardware, vram_headroom="auto", ram_budget=None)


def rank_for_hardware(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    *,
    context_length: int,
    top_n: int,
    benchmark_scores: dict[str, float] | None,
    task_scores: dict[str, float] | None = None,
    profile: str = "general",
    quant: str | None = None,
    min_speed: float | None = None,
    min_params: float | None = None,
    evidence_filter: str = "any",
    fit_filter: str = "any",
) -> list[CompatibilityResult]:
    """Rank models for ``hardware`` the way ``whichllm`` shows them.

    Without ``min_params``, general rankings get a size floor that follows the
    GPU (too-small models are hidden on large cards), relaxed when it would
    leave nothing to show.
    """
    from whichllm.engine.ranker import rank_models

    size_floor = (
        _auto_min_params_for_profile(hardware, profile)
        if min_params is None
        else min_params
    )
    kwargs = dict(
        context_length=context_length,
        top_n=top_n,
        quant_filter=quant,
        min_speed=min_speed,
        benchmark_scores=benchmark_scores,
        task_scores=task_scores,
        task_profile=profile,
        require_direct_top=True,
        evidence_filter=evidence_filter,
        fit_filter=fit_filter,
    )
    ranked = rank_models(models, hardware, min_params_b=size_floor, **kwargs)
    if not ranked and size_floor is not None and min_params is None:
        ranked = rank_models(models, hardware, min_params_b=None, **kwargs)
    return ranked
