"""Measurements people reported and a maintainer verified.

``data/measurements.json`` comes from ``scripts/collect_measurements.py``. Every
report is kept there, but only plain llama.cpp runs measured by ``whichllm
bench``, with one GPU holding the whole model, count as evidence for an estimate.
Ollama, LM Studio, other engines and hand-entered numbers are reference values.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from whichllm.data import read_json

# One lucky run is not evidence about a GPU.
MIN_REPORTS = 3

_MEASUREMENTS: list[dict] = read_json("measurements.json")["measurements"]
_NAME_NOISE_RE = re.compile(r"\(simulated\)|\(tm\)|\(r\)", re.IGNORECASE)
_VENDOR_PREFIX_RE = re.compile(r"^(?:nvidia|amd|intel)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class GpuEvidence:
    reports: int
    median_ratio: float  # measured over estimated


def counts_as_evidence(row: dict) -> bool:
    return (
        row["engine"] == "llama.cpp"
        and row["app"] is None
        and row["method"] == "auto"
        and row["placement"] == "gpu"
        and row["gpu_count"] == 1
        and bool(row["estimate_tok_per_sec"])
    )


def _gpu_key(name: str) -> str:
    bare = " ".join(_NAME_NOISE_RE.sub(" ", name).split())
    return _VENDOR_PREFIX_RE.sub("", bare).casefold()


def gpu_evidence(gpu_name: str) -> GpuEvidence | None:
    """What verified reports say about this GPU, once there are enough of them."""
    key = _gpu_key(gpu_name)
    ratios = [
        row["tok_per_sec"] / row["estimate_tok_per_sec"]
        for row in _MEASUREMENTS
        if row["gpu"] and _gpu_key(row["gpu"]) == key and counts_as_evidence(row)
    ]
    if len(ratios) < MIN_REPORTS:
        return None
    return GpuEvidence(len(ratios), statistics.median(ratios))
