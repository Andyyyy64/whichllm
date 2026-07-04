"""Lineage-aware recency demotion for frozen benchmark scores."""

from __future__ import annotations

import re

_LINEAGE_DEMOTION_REGEX = None


def _build_lineage_regex():
    """Compile MODEL_LINEAGE_VERSIONS once into (family, [(re, idx)]) form."""
    global _LINEAGE_DEMOTION_REGEX
    if _LINEAGE_DEMOTION_REGEX is not None:
        return _LINEAGE_DEMOTION_REGEX
    from whichllm.constants import MODEL_LINEAGE_VERSIONS

    out = {}
    for family, entries in MODEL_LINEAGE_VERSIONS.items():
        compiled = [(re.compile(pat), idx) for pat, idx in entries]
        max_idx = max(idx for _, idx in entries)
        out[family] = (compiled, max_idx)
    _LINEAGE_DEMOTION_REGEX = out
    return out


def _lineage_recency_factor(model_id: str) -> float:
    """Return a multiplicative recency factor for frozen-only scores.

    Newest generation in a known family -> 1.0 (no demotion). Each generation
    older -> another 12% off. Unknown families -> 1.0.
    """
    if not model_id:
        return 1.0
    lower = model_id.lower()
    families = _build_lineage_regex()
    best_factor = 1.0
    for family, (patterns, max_idx) in families.items():
        for regex, idx in patterns:
            if regex.search(lower):
                gens_old = max(0, max_idx - idx)
                factor = max(0.55, 1.0 - 0.12 * gens_old)
                if factor < best_factor:
                    best_factor = factor
                break
    return best_factor


def _apply_lineage_recency_demotion(
    combined: dict[str, float],
    frozen: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    """Multiply frozen-only entries by a lineage-derived recency factor."""
    if not combined:
        return combined
    out: dict[str, float] = {}
    for k, v in combined.items():
        if k in current:
            out[k] = v
            continue
        factor = _lineage_recency_factor(k)
        out[k] = round(v * factor, 1)
    return out
