"""Stable exports for CLI validation and hardware helpers."""

from whichllm.cli_options import (
    _validate_gpu_flags,
    _validate_output_flags,
    _validate_ranking_flags,
    _validate_profile,
    _validate_evidence,
    _resolve_evidence_mode,
    _resolve_fit_filter,
    _resolve_speed_filter,
    _validate_lmstudio_path_flags,
)
from whichllm.cli_hardware import (
    _parse_memory_amount,
    _auto_vram_headroom,
    _parse_vram_headroom,
    _apply_memory_budgets,
    _format_budget_bytes,
    _apply_gpu_overrides,
    _auto_min_params_for_profile,
)
from whichllm.cli_metadata import (
    _include_vision_candidates,
    _fill_missing_published_at,
    _merge_model_eval_benchmarks,
)

__all__ = [
    "_validate_gpu_flags",
    "_validate_output_flags",
    "_validate_ranking_flags",
    "_validate_profile",
    "_validate_evidence",
    "_resolve_evidence_mode",
    "_resolve_fit_filter",
    "_resolve_speed_filter",
    "_validate_lmstudio_path_flags",
    "_parse_memory_amount",
    "_auto_vram_headroom",
    "_parse_vram_headroom",
    "_apply_memory_budgets",
    "_format_budget_bytes",
    "_apply_gpu_overrides",
    "_auto_min_params_for_profile",
    "_include_vision_candidates",
    "_fill_missing_published_at",
    "_merge_model_eval_benchmarks",
]
