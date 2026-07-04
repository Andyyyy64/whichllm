"""Benchmark score indexing and model-line interpolation."""

from __future__ import annotations

import math
import re
import statistics


def _extract_params_b_from_id(model_id: str) -> float | None:
    """Extract parameter size in billions from model ID text."""
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        return max(float(v) for v in matches)
    except ValueError:
        return None


def _extract_model_lines(model_id: str) -> list[str]:
    """Extract model line candidates from a model ID, most specific first."""
    if "/" not in model_id:
        return []
    lower = model_id.lower()

    stripped = re.sub(r"-(gguf|awq|gptq|fp8|fp16|bf16|mxfp4|nvfp4)$", "", lower)
    stripped = re.sub(r"-\d{4}(-hf)?$", "", stripped)

    lines: list[str] = []
    cleaned = re.sub(
        r"-\d+(\.\d+)?b(-a\d+b)?(-[a-z][-a-z0-9]*)*$",
        "",
        stripped,
    )
    if cleaned != stripped and "/" in cleaned:
        lines.append(cleaned)

    for line in list(lines) + ([stripped] if not lines else []):
        broader = re.sub(r"(\d+)\.\d+$", r"\1", line)
        if broader != line and broader not in lines:
            lines.append(broader)

    return lines


def _interpolate_line_score(
    bucket: list[tuple[float | None, float]],
    params_b: float | None,
) -> tuple[float, float]:
    """Interpolate score from same-model-line benchmarks with confidence."""
    if not bucket:
        return 0.0, 0.0

    valid = [(p, s) for p, s in bucket if p is not None]
    if not valid:
        vals = [s for _, s in bucket]
        return statistics.median(vals), 0.25

    if params_b is None or params_b <= 0:
        vals = [s for _, s in valid]
        return statistics.median(vals), 0.30

    weighted: list[tuple[float, float, float]] = []
    for p, s in valid:
        assert p is not None
        dist = abs(math.log2(max(params_b, 0.1) / max(p, 0.1)))
        w = 1.0 / (0.35 + dist)
        weighted.append((w, s, dist))

    score = sum(w * s for w, s, _ in weighted) / sum(w for w, _, _ in weighted)
    nearest = min(d for _, _, d in weighted)
    if nearest <= 0.15:
        conf = 0.45
    elif nearest <= 0.50:
        conf = 0.34
    else:
        conf = 0.26
    return score, conf


def build_score_index(
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build case-insensitive and model-line lookup indices."""
    ci_index: dict[str, float] = {}
    line_index: dict[str, float] = {}

    for key, val in scores.items():
        lk = key.lower()
        if lk not in ci_index or val > ci_index[lk]:
            ci_index[lk] = val

        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            lines = [lk]
        for line in lines:
            if line not in line_index or val > line_index[line]:
                line_index[line] = val

    return ci_index, line_index


def build_line_bucket_index(
    scores: dict[str, float],
) -> dict[str, list[tuple[float | None, float]]]:
    """Build line -> [(params_b, score)] index for size-aware interpolation."""
    buckets: dict[str, list[tuple[float | None, float]]] = {}
    for key, val in scores.items():
        params_b = _extract_params_b_from_id(key)
        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            lines = [key.lower()]
        for line in lines:
            buckets.setdefault(line, []).append((params_b, val))
    return buckets
