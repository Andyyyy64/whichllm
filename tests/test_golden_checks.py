"""The golden checks themselves: each must fail on the row it exists to catch."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from tests.golden import harness
from tests.golden.harness import CHECKS, Archetype, Constraint

_GiB = 1024**3


def _row(
    model_id: str = "org/Model",
    *,
    speed: float | None = 30.0,
    required_gb: float = 8.0,
    available_gb: float = 12.0,
    fit_type: str = "full_gpu",
    file_gb: float | None = 6.0,
    quant: str = "Q4_K_M",
    is_moe: bool = False,
    older: tuple[str, ...] = (),
    basis: str = "general",
):
    variant = (
        None
        if file_gb is None
        else SimpleNamespace(quant_type=quant, file_size_bytes=int(file_gb * 1e9))
    )
    return SimpleNamespace(
        model=SimpleNamespace(
            id=model_id, is_moe=is_moe, parameter_count=8_000_000_000
        ),
        can_run=True,
        estimated_tok_per_sec=speed,
        vram_required_bytes=int(required_gb * _GiB),
        vram_available_bytes=int(available_gb * _GiB),
        fit_type=fit_type,
        placement=None,
        gguf_variant=variant,
        older_generations=list(older),
        benchmark_basis=basis,
        quality_score=50.0,
    )


def _constraint(kind: str, **fields) -> Constraint:
    return Constraint(id="test", kind=kind, rationale="", **fields)


_RTX3060 = Archetype(
    id="test-rtx3060",
    gpus=("RTX 3060",),
    ram_gb=16,
    os="windows",
    profiles=("general",),
    vram_gb=12,
)


def _problem(constraint: Constraint, rows) -> str | None:
    return CHECKS[constraint.kind](constraint, rows, _RTX3060)


def test_top_speed_fails_when_nothing_is_fast():
    constraint = _constraint("top_speed", top=3, min_tok_per_sec=8.0)

    assert "no candidate" in _problem(constraint, [_row(speed=1.0), _row(speed=None)])


def test_top_speed_only_claims_the_slots_the_machine_can_fill():
    constraint = _constraint("top_speed", top=3, min_tok_per_sec=8.0)
    fast, slow = _row(speed=20.0), _row(speed=2.0)

    assert _problem(constraint, [fast, slow, slow]) is None
    assert "below 8.0" in _problem(constraint, [slow, fast, fast])


def test_stale_generation_needs_a_runnable_newer_model():
    constraint = _constraint("stale_generation", top=2, pairs=(("-old$", "^org/new$"),))
    stale = _row("org/line-old")

    assert _problem(constraint, [stale]) is None
    assert "stale" in _problem(constraint, [stale, _row("org/new")])
    assert _problem(constraint, [_row("org/new"), _row("x"), stale]) is None


def test_older_noted_wants_the_row_to_name_what_it_replaced():
    constraint = _constraint("older_noted", pairs=(("^org/old$", "^org/new$"),))

    assert _problem(constraint, [_row("org/new", older=("org/old",))]) is None
    assert "not noted" in _problem(constraint, [_row("org/new")])


@pytest.mark.parametrize(
    ("speed", "required_gb", "expected"),
    [(30.0, 8.0, True), (70.0, 8.0, False), (30.0, 14.0, False), (None, 8.0, False)],
)
def test_row_within_holds_speed_and_memory(speed, required_gb, expected):
    constraint = _constraint(
        "row_within",
        patterns=("^org/Model$",),
        min_tok_per_sec=20.0,
        max_tok_per_sec=60.0,
        min_required_gb=7.0,
        max_required_gb=9.0,
    )
    row = _row(speed=speed, required_gb=required_gb)

    assert (_problem(constraint, [row]) is None) is expected


def test_row_within_fails_without_the_row():
    constraint = _constraint("row_within", patterns=("^org/Other$",))

    assert "no row" in _problem(constraint, [_row()])


def test_bandwidth_ceiling_caps_dense_full_gpu_rows_only():
    constraint = _constraint("bandwidth_ceiling")
    # The simulated RTX 3060 reads 360 GB/s, so a 6 GB file tops out at 60 tok/s.
    too_fast = dict(speed=61.0, file_gb=6.0)

    assert _problem(constraint, [_row(speed=59.0, file_gb=6.0)]) is None
    assert "GB/s" in _problem(constraint, [_row(**too_fast)])
    assert _problem(constraint, [_row(**too_fast, is_moe=True)]) is None
    assert _problem(constraint, [_row(**too_fast, fit_type="partial_offload")]) is None


def test_full_gpu_budget_uses_the_memory_on_the_box():
    constraint = _constraint("full_gpu_budget", max_gpu_fraction=0.95)

    assert _problem(constraint, [_row(required_gb=11.4)]) is None
    assert "past 11.4" in _problem(constraint, [_row(required_gb=11.5)])
    assert (
        _problem(constraint, [_row(required_gb=30.0, fit_type="partial_offload")])
        is None
    )


def test_ram_budget_counts_what_does_not_fit_the_gpu():
    constraint = _constraint("ram_budget", max_ram_fraction=0.85)

    assert _problem(constraint, [_row(required_gb=25.0, available_gb=12.0)]) is None
    assert "in RAM" in _problem(constraint, [_row(required_gb=26.0, available_gb=12.0)])


def test_gguf_only_rejects_rows_without_a_gguf():
    constraint = _constraint("gguf_only")

    assert _problem(constraint, [_row()]) is None
    assert "non-GGUF" in _problem(constraint, [_row(file_gb=None)])


def test_quant_in_rejects_other_quants():
    constraint = _constraint("quant_in", quants=("Q5_K_M", "Q6_K"))

    assert _problem(constraint, [_row(quant="Q6_K")]) is None
    assert "outside" in _problem(constraint, [_row(quant="Q3_K_M")])


def test_task_evidence_counts_rows_ranked_on_the_task():
    constraint = _constraint("task_evidence", top=3, min_rows=2)
    coding, general = _row(basis="coding"), _row()

    assert _problem(constraint, [coding, coding, general]) is None
    assert "1 of top 3" in _problem(constraint, [coding, general, general])


def test_every_pattern_still_names_something_in_the_catalog():
    """A renamed repo must not leave a constraint silently matching nothing."""
    ids = [model.id for model in harness.catalog_models()]
    _, constraints = harness.load_spec()

    dead = [
        (constraint.id, pattern)
        for constraint in constraints
        for pattern in (*constraint.patterns, *sum(constraint.pairs, ()))
        if not any(re.search(pattern, model_id) for model_id in ids)
    ]

    assert not dead, dead


def test_every_constraint_applies_somewhere():
    applied = {case.constraint.id for case in harness.golden_cases()}
    _, constraints = harness.load_spec()

    assert {constraint.id for constraint in constraints} == applied


def test_metal_gating_uses_the_size_on_the_box():
    archetypes, _ = harness.load_spec()
    by_id = {archetype.id: archetype for archetype in archetypes}

    assert by_id["m4-16gb-mac"].gpu_memory_gb == 16
    assert by_id["cpu-32gb-linux"].gpu_memory_gb == 0
    assert by_id["mixed-1660s-5060ti-win"].gpu_memory_bytes == (6 * _GiB, 16 * _GiB)
