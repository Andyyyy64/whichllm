"""LiveBench (livebench.ai) source: a dated snapshot.

LiveBench publishes its leaderboard as a dated CSV (e.g.
``https://livebench.ai/table_2026_01_08.csv``). The snapshot lives in
``data/livebench.json``. To refresh it, download the latest
``table_YYYY_MM_DD.csv`` from livebench.ai and run
``scripts/import_livebench_csv.py``.
"""

from __future__ import annotations

from whichllm.data import read_json

_SNAPSHOT = read_json("livebench.json")

# LiveBench global average per model. Values are raw 0-100; the normalizer
# below rescales them onto the project's shared 0-100 axis.
LIVEBENCH_SNAPSHOT: dict[str, float] = _SNAPSHOT["models"]
# The date of the LiveBench CSV the snapshot was read from.
LIVEBENCH_READ_AT: str = _SNAPSHOT["read_at"]

# Hand-estimated LiveBench-equivalents carried over from 2026-04 for older
# and smaller models LiveBench no longer lists. They sit well above what AA
# measures for the same models, so the merge uses them only for models no
# source measured.
LIVEBENCH_CARRYOVER: dict[str, float] = _SNAPSHOT["carryover"]["models"]

LIVEBENCH_RAW_DATA: dict[str, float] = {**LIVEBENCH_CARRYOVER, **LIVEBENCH_SNAPSHOT}
LIVEBENCH_ESTIMATED_IDS: frozenset[str] = frozenset(
    LIVEBENCH_CARRYOVER.keys() - LIVEBENCH_SNAPSHOT.keys()
)
# LiveBench and AA are merged per model, so LiveBench is mapped onto AA's
# normalized scale: a least-squares fit over the 15 models both measured
# (LiveBench 2026-01-08 CSV vs AA v4.3.2 "live" entries) gives
# AA = 0.871 * raw + 6.35, residual SD 3.2. These bounds are that line.
# Refit them whenever either snapshot is refreshed; a test checks the fit.
_LB_MIN = -7.3
_LB_MAX = 107.6


def _normalize_livebench(score: float) -> float:
    normalized = (score - _LB_MIN) / (_LB_MAX - _LB_MIN) * 100.0
    return max(0.0, min(100.0, round(normalized, 1)))


def get_livebench_data() -> dict[str, float]:
    """Return the LiveBench snapshot, normalized to the 0-100 scale."""
    return {
        hf_id: _normalize_livebench(raw)
        for hf_id, raw in LIVEBENCH_RAW_DATA.items()
        if _normalize_livebench(raw) > 0
    }
