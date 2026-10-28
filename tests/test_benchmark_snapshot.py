"""The displayed benchmark snapshot date must track the curated fallbacks."""

from __future__ import annotations

import importlib
import pkgutil
import re

import whichllm.models.benchmark_sources as sources

_FALLBACK_NAME_RE = re.compile(r"_FALLBACK_(\d{4})_(\d{2})(?:_\d{2})?$")


def _fallback_months() -> list[str]:
    months = []
    for info in pkgutil.iter_modules(sources.__path__):
        module = importlib.import_module(f"{sources.__name__}.{info.name}")
        for name in vars(module):
            match = _FALLBACK_NAME_RE.search(name)
            if match:
                months.append(f"{match.group(1)}-{match.group(2)}")
    return months


def test_benchmark_snapshot_is_newest_curated_fallback():
    months = _fallback_months()

    assert months, "expected dated *_FALLBACK_YYYY_MM constants"
    assert sources.BENCHMARK_SNAPSHOT == max(months)
