"""Compatibility shim for model ranking helpers.

The implementation is split by responsibility:

- ``ranking`` for the top-level ranking orchestration
- ``ranking_filters`` for model/profile/evidence filters
- ``ranking_score`` for quality and final selection scoring
- ``ranking_variants`` for GGUF candidate variant selection

Existing imports from ``whichllm.engine.ranker`` are re-exported here.
"""

from __future__ import annotations

from whichllm.engine.ranking import rank_models
from whichllm.engine.ranking_filters import (
    _DUBIOUS_DERIVATIVE_PATTERNS,
    _EXCLUDED_NAME_PATTERNS,
    _EXCLUDED_ORGS,
    _derivative_name_penalty,
    _detect_specializations,
    _effective_params_b,
    _generation_bonus,
    _is_excluded_model,
    _is_gguf_only_backend,
    _knowledge_capacity_b,
    _matches_profile,
    _passes_evidence_filter,
)
from whichllm.engine.ranking_score import (
    _SOURCE_WEIGHTS,
    _compute_quality_score,
    _family_selection_key,
    _partial_offload_quality_factor,
)
from whichllm.engine.ranking_sources import (
    _OFFICIAL_ORGS,
    _REPACKAGER_ORGS,
    _TRUSTED_CONVERTERS,
)
from whichllm.engine.ranking_variants import (
    _PREQUANTIZED_REPO_RE,
    _SYNTHETIC_QUANTS,
    _iter_candidate_variants,
    _synthesize_variants_for_official_repo,
)

__all__ = [
    "_DUBIOUS_DERIVATIVE_PATTERNS",
    "_EXCLUDED_NAME_PATTERNS",
    "_EXCLUDED_ORGS",
    "_OFFICIAL_ORGS",
    "_PREQUANTIZED_REPO_RE",
    "_REPACKAGER_ORGS",
    "_SOURCE_WEIGHTS",
    "_SYNTHETIC_QUANTS",
    "_TRUSTED_CONVERTERS",
    "_compute_quality_score",
    "_derivative_name_penalty",
    "_detect_specializations",
    "_effective_params_b",
    "_family_selection_key",
    "_generation_bonus",
    "_is_excluded_model",
    "_is_gguf_only_backend",
    "_iter_candidate_variants",
    "_knowledge_capacity_b",
    "_matches_profile",
    "_partial_offload_quality_factor",
    "_passes_evidence_filter",
    "_synthesize_variants_for_official_repo",
    "rank_models",
]
