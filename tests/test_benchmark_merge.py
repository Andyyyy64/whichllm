"""LiveBench and AA are merged on one scale, measured values first."""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pytest

from whichllm.models.benchmark_fetch import _merge_general_scores
from whichllm.models.benchmark_sources import aa_index, livebench

_SOURCES = Path(aa_index.__file__).parent


def _aa_snapshot_markers() -> dict[str, str]:
    """``{hf_id: "live" | "peer"}`` from the snapshot's entry comments."""
    text = (_SOURCES / "aa_index.py").read_text()
    block = text.split("AA_INDEX_FALLBACK_2026_10_27: dict[str, float] = {", 1)[1]
    block = block.split("\n}", 1)[0]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'^\s*"([^"]+)":\s*[-\d.]+,\s*#\s*(live|peer)\b', block, re.M
        )
    }


def test_aa_peer_ids_match_the_snapshot_comments():
    markers = _aa_snapshot_markers()
    assert markers.keys() == aa_index.AA_INDEX_FALLBACK_2026_10_27.keys()
    assert aa_index.AA_PEER_IDS == {k for k, v in markers.items() if v == "peer"}


def test_livebench_sits_on_the_aa_scale_where_both_measured():
    live = {k for k, v in _aa_snapshot_markers().items() if v == "live"}
    shared = sorted(livebench.LIVEBENCH_2026_01_08.keys() & live)
    lb = livebench.get_livebench_data()
    aa = aa_index.get_aa_curated_fallback()
    residuals = [lb[k] - aa[k] for k in shared]

    assert len(shared) >= 10
    # A refresh of either snapshot without refitting _LB_MIN/_LB_MAX shifts
    # LiveBench off AA's scale and fails here.
    assert abs(statistics.mean(residuals)) < 1.0
    assert statistics.pstdev(residuals) < 5.0


def test_measured_values_are_averaged():
    merged = _merge_general_scores(
        ({"a": 60.0, "b": 50.0}, frozenset()),
        ({"a": 50.0, "c": 40.0}, frozenset()),
    )
    assert merged == {"a": 55.0, "b": 50.0, "c": 40.0}


def test_estimates_only_count_without_a_measurement():
    merged = _merge_general_scores(
        (
            {"measured-by-aa": 80.0, "nobody-measured": 50.0},
            frozenset({"measured-by-aa", "nobody-measured"}),
        ),
        (
            {"measured-by-aa": 45.0, "nobody-measured": 40.0},
            frozenset({"nobody-measured"}),
        ),
    )
    assert merged["measured-by-aa"] == 45.0
    assert merged["nobody-measured"] == 45.0


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # LiveBench 83.5 used to win over AA 61.0 for Qwen3.6-27B.
        ("Qwen/Qwen3.6-27B", 62.2),
        # A hand-estimated LiveBench carryover no longer outvotes AA.
        ("Qwen/Qwen3-14B", 40.9),
        # AA-only models keep their AA value.
        ("Qwen/Qwen3.8-27B", 77.8),
    ],
)
def test_snapshot_merge(model_id, expected):
    merged = _merge_general_scores(
        (livebench.get_livebench_data(), livebench.LIVEBENCH_ESTIMATED_IDS),
        (aa_index.get_aa_curated_fallback(), aa_index.AA_PEER_IDS),
    )
    assert merged[model_id] == pytest.approx(expected, abs=0.15)
