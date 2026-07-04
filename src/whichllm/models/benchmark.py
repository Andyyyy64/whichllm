"""Compatibility shim for benchmark fetching and lookup helpers.

The implementation is split by responsibility:

- ``benchmark_cache`` for cache I/O
- ``benchmark_fetch`` for source fetching and layered merge policy
- ``benchmark_lineage`` for frozen-score recency demotion
- ``benchmark_index`` for score indices and model-line interpolation
- ``benchmark_lookup`` for evidence lookup and inheritance rules

Existing imports from ``whichllm.models.benchmark`` are re-exported here.
"""

from __future__ import annotations

from whichllm.models import benchmark_cache as _cache_module
from whichllm.models.benchmark_cache import DEFAULT_TTL_SECONDS
from whichllm.models.benchmark_fetch import fetch_benchmark_scores
from whichllm.models.benchmark_index import (
    _extract_model_lines,
    _extract_params_b_from_id,
    _interpolate_line_score,
    build_line_bucket_index,
    build_score_index,
)
from whichllm.models.benchmark_lineage import (
    _apply_lineage_recency_demotion,
    _build_lineage_regex,
    _lineage_recency_factor,
)
from whichllm.models.benchmark_lookup import (
    _REPO_SUFFIXES,
    _append_unique,
    _generate_candidates,
    _generate_score_name_candidates,
    _params_compatible,
    _strip_repo_suffix,
    _try_lookup,
    lookup_benchmark,
    lookup_benchmark_evidence,
)
from whichllm.models.benchmark_types import BenchmarkEvidence

CACHE_DIR = _cache_module.CACHE_DIR
BENCHMARK_CACHE = _cache_module.BENCHMARK_CACHE


def _sync_cache_module_globals() -> None:
    _cache_module.CACHE_DIR = CACHE_DIR
    _cache_module.BENCHMARK_CACHE = BENCHMARK_CACHE


def load_benchmark_cache() -> dict[str, float] | None:
    """Load cached benchmark scores through the legacy shim globals."""
    _sync_cache_module_globals()
    return _cache_module.load_benchmark_cache()


def save_benchmark_cache(scores: dict[str, float]) -> None:
    """Save cached benchmark scores through the legacy shim globals."""
    _sync_cache_module_globals()
    _cache_module.save_benchmark_cache(scores)


__all__ = [
    "BENCHMARK_CACHE",
    "CACHE_DIR",
    "DEFAULT_TTL_SECONDS",
    "BenchmarkEvidence",
    "_REPO_SUFFIXES",
    "_append_unique",
    "_apply_lineage_recency_demotion",
    "_build_lineage_regex",
    "_extract_model_lines",
    "_extract_params_b_from_id",
    "_generate_candidates",
    "_generate_score_name_candidates",
    "_interpolate_line_score",
    "_lineage_recency_factor",
    "_params_compatible",
    "_strip_repo_suffix",
    "_try_lookup",
    "build_line_bucket_index",
    "build_score_index",
    "fetch_benchmark_scores",
    "load_benchmark_cache",
    "lookup_benchmark",
    "lookup_benchmark_evidence",
    "save_benchmark_cache",
]
