"""Curated reference data: GPU specs, quantization tiers, model lineage, framework limits.

The benchmark snapshots are JSON files in this package, so a refresh rewrites
data rather than Python.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def read_json(name: str) -> dict:
    """Load one of the JSON data files in this package."""
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
