"""Shared benchmark data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Benchmark evidence with confidence.

    source values, ordered from most trusted to least:
      - "direct"        : independent leaderboard / Arena ELO hit on exact id
      - "variant"       : suffix-stripped derivative of a direct leaderboard hit
      - "base_model"    : cardData.base_model pointer to a direct hit
      - "line_interp"   : size-aware interpolation within the same model line
      - "self_reported" : evalResults reported by the uploader themselves
      - "none"          : no usable signal
    """

    score: float | None
    confidence: float
    source: str
