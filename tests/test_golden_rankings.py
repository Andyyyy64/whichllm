"""Golden ranking set: archetype recommendations must satisfy their constraints.

Archetypes and constraints live in tests/golden/archetypes.toml; the frozen
catalog and the reviewable top-pick snapshot are refreshed with
scripts/update_golden.py.
"""

from __future__ import annotations

import pytest

from tests.golden import harness


def _golden_params():
    params = []
    for case in harness.golden_cases():
        reason = case.xfail_reason
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(case, id=case.test_id, marks=marks))
    return params


@pytest.mark.parametrize("case", _golden_params())
def test_golden_constraint(case):
    problem = harness.evaluate(case)
    assert problem is None, problem


def test_top_picks_snapshot_is_current():
    expected = harness.TOP_PICKS_PATH.read_text(encoding="utf-8")
    assert harness.render_top_picks() == expected, (
        "Recommendations changed. Run "
        "`uv run --locked python scripts/update_golden.py snapshot` "
        "and review the diff in tests/golden/top_picks.md."
    )
