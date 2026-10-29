"""Shared golden-set plumbing: frozen catalog, archetypes, and the ranking call.

The golden set checks recommendations for fixed hardware archetypes against a
frozen model catalog and frozen benchmark scores, so a ranking change shows up
as a test diff instead of a surprise in someone's terminal. Everything here is
offline; only ``scripts/update_golden.py`` touches the network.
"""

from __future__ import annotations

import gzip
import json
import re
import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import cached_property, lru_cache
from pathlib import Path

from whichllm.cli_hardware import (
    _apply_memory_budgets,
    _apply_ram_bandwidth,
    _auto_min_params_for_profile,
)
from whichllm.engine.ranker import rank_models
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.gpu_simulator import create_synthetic_gpu
from whichllm.hardware.types import HardwareInfo
from whichllm.models.fetcher import dicts_to_models
from whichllm.models.grouper import group_models
from whichllm.models.types import ModelInfo

GOLDEN_DIR = Path(__file__).resolve().parent
CATALOG_PATH = GOLDEN_DIR / "catalog.json.gz"
ARCHETYPES_PATH = GOLDEN_DIR / "archetypes.toml"
TOP_PICKS_PATH = GOLDEN_DIR / "top_picks.md"
SNAPSHOT_TOP_N = 5

# Mirrors the defaults of a plain `whichllm` run.
CONTEXT_LENGTH = 4096
DISPLAY_TOP_N = 10
# Rank deeper than the display so constraints can ask "is a newer model
# runnable at all?" without changing the order of the first rows.
RANK_DEPTH = 50

_GiB = 1024**3


@dataclass(frozen=True)
class Archetype:
    id: str
    gpus: tuple[str, ...]
    ram_gb: float
    os: str
    profiles: tuple[str, ...]
    vram_gb: float | None = None
    # None means the default (dual-channel DDR4-3200), or the unified
    # bandwidth for Apple GPUs.
    ram_bandwidth_gbps: float | None = None
    cpu_cores: int = 16
    context_length: int = CONTEXT_LENGTH
    note: str = ""

    def hardware(self) -> HardwareInfo:
        gpus = [create_synthetic_gpu(name, self.vram_gb) for name in self.gpus]
        hardware = HardwareInfo(
            gpus=gpus,
            cpu_name="Golden CPU",
            cpu_cores=self.cpu_cores,
            has_avx2=True,
            ram_bytes=int(self.ram_gb * _GiB),
            disk_free_bytes=2048 * _GiB,
            os=self.os,
        )
        _apply_ram_bandwidth(hardware, self.ram_bandwidth_gbps)
        return _apply_memory_budgets(hardware, vram_headroom="auto", ram_budget=None)

    @cached_property
    def gpu_memory_gb(self) -> float:
        """Largest single-GPU memory, the figure people shop by."""
        if not self.gpus:
            return 0.0
        return max(g.vram_bytes for g in self.hardware().gpus) / _GiB


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    rationale: str
    draft: bool = True
    top: int = DISPLAY_TOP_N
    patterns: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    min_tok_per_sec: float | None = None
    archetypes: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("general",)
    min_gpu_memory_gb: float | None = None
    gpu: bool | None = None
    xfail: dict[str, str] = field(default_factory=dict)

    def applies_to(self, archetype: Archetype, profile: str) -> bool:
        if self.archetypes and archetype.id not in self.archetypes:
            return False
        if profile not in self.profiles:
            return False
        if self.gpu is not None and bool(archetype.gpus) != self.gpu:
            return False
        if self.min_gpu_memory_gb is not None:
            return archetype.gpu_memory_gb >= self.min_gpu_memory_gb
        return True


@dataclass(frozen=True)
class GoldenCase:
    archetype: Archetype
    profile: str
    constraint: Constraint

    @property
    def case_id(self) -> str:
        return f"{self.archetype.id}/{self.profile}"

    @property
    def test_id(self) -> str:
        return f"{self.archetype.id}.{self.profile}.{self.constraint.id}"

    @property
    def xfail_reason(self) -> str | None:
        """Known-failure reason; xfail keys are fnmatch patterns on case_id."""
        for pattern, reason in self.constraint.xfail.items():
            if fnmatch(self.case_id, pattern):
                return reason
        return None


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with gzip.open(CATALOG_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def catalog_models() -> tuple[ModelInfo, ...]:
    """Grouped and flattened models, exactly as `whichllm` feeds the ranker."""
    models = dicts_to_models(load_catalog()["models"])
    flattened: list[ModelInfo] = []
    for family in group_models(models):
        flattened.append(family.base_model)
        flattened.extend(family.variants)
    return tuple(flattened)


def catalog_benchmarks() -> dict[str, float]:
    return dict(load_catalog()["benchmark_scores"])


@lru_cache(maxsize=1)
def load_spec() -> tuple[tuple[Archetype, ...], tuple[Constraint, ...]]:
    with ARCHETYPES_PATH.open("rb") as f:
        raw = tomllib.load(f)
    archetypes = tuple(
        Archetype(
            id=a["id"],
            gpus=tuple(a.get("gpus", ())),
            ram_gb=float(a["ram_gb"]),
            os=a["os"],
            profiles=tuple(a.get("profiles", ("general",))),
            vram_gb=a.get("vram_gb"),
            ram_bandwidth_gbps=a.get("ram_bandwidth_gbps"),
            cpu_cores=a.get("cpu_cores", 16),
            context_length=a.get("context_length", CONTEXT_LENGTH),
            note=a.get("note", ""),
        )
        for a in raw["archetype"]
    )
    constraints = tuple(
        Constraint(
            id=c["id"],
            kind=c["kind"],
            rationale=c["rationale"],
            draft=c.get("draft", True),
            top=c.get("top", DISPLAY_TOP_N),
            patterns=tuple(c.get("patterns", ())),
            pairs=tuple((p["old"], p["new"]) for p in c.get("pairs", ())),
            min_tok_per_sec=c.get("min_tok_per_sec"),
            archetypes=tuple(c.get("archetypes", ())),
            profiles=tuple(c.get("profiles", ("general",))),
            min_gpu_memory_gb=c.get("min_gpu_memory_gb"),
            gpu=c.get("gpu"),
            xfail=dict(c.get("xfail", {})),
        )
        for c in raw["constraint"]
    )
    _check_spec(archetypes, constraints)
    return archetypes, constraints


def _check_spec(
    archetypes: tuple[Archetype, ...], constraints: tuple[Constraint, ...]
) -> None:
    ids = [a.id for a in archetypes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate archetype id")
    constraint_ids = [c.id for c in constraints]
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError("duplicate constraint id")
    for c in constraints:
        if c.kind not in CHECKS:
            raise ValueError(f"{c.id}: unknown kind {c.kind!r}")
        for archetype_id in c.archetypes:
            if archetype_id not in ids:
                raise ValueError(f"{c.id}: unknown archetype {archetype_id!r}")
        applicable = [
            f"{a.id}/{p}" for a in archetypes for p in a.profiles if c.applies_to(a, p)
        ]
        for pattern in c.xfail:
            if not any(fnmatch(case_id, pattern) for case_id in applicable):
                raise ValueError(f"{c.id}: xfail {pattern!r} matches no case")


def golden_cases() -> list[GoldenCase]:
    archetypes, constraints = load_spec()
    return [
        GoldenCase(archetype, profile, constraint)
        for archetype in archetypes
        for profile in archetype.profiles
        for constraint in constraints
        if constraint.applies_to(archetype, profile)
    ]


_ranking_cache: dict[tuple[str, str], tuple[CompatibilityResult, ...]] = {}


def rank_case(archetype: Archetype, profile: str) -> tuple[CompatibilityResult, ...]:
    """Rank the frozen catalog for one archetype the way `whichllm` does."""
    key = (archetype.id, profile)
    if key not in _ranking_cache:
        _ranking_cache[key] = tuple(
            _rank(
                archetype.hardware(),
                profile,
                list(catalog_models()),
                archetype.context_length,
            )
        )
    return _ranking_cache[key]


def _rank(
    hardware: HardwareInfo,
    profile: str,
    models: list[ModelInfo],
    context_length: int = CONTEXT_LENGTH,
) -> list[CompatibilityResult]:
    benchmark_scores = catalog_benchmarks()
    auto_min_params = _auto_min_params_for_profile(hardware, profile)
    kwargs = dict(
        context_length=context_length,
        top_n=RANK_DEPTH,
        benchmark_scores=benchmark_scores,
        task_profile=profile,
        require_direct_top=True,
        evidence_filter="any",
        fit_filter="any",
    )
    results = rank_models(models, hardware, min_params_b=auto_min_params, **kwargs)
    # `whichllm` relaxes the automatic size floor when it removes everything.
    if not results and auto_min_params is not None:
        results = rank_models(models, hardware, min_params_b=None, **kwargs)
    return results


def describe(result: CompatibilityResult) -> str:
    speed = result.estimated_tok_per_sec
    speed_text = "?" if speed is None else f"{speed:.1f} tok/s"
    quant = result.gguf_variant.quant_type if result.gguf_variant else "-"
    return (
        f"{result.model.id} [{quant}, {result.fit_type}, {speed_text}, "
        f"score {result.quality_score:.1f}]"
    )


def _matches(model_id: str, patterns: tuple[str, ...]) -> list[str]:
    return [p for p in patterns if re.search(p, model_id)]


def _check_nonempty(c: Constraint, results) -> str | None:
    if not results:
        return "no recommendations"
    bad = [describe(r) for r in results[: c.top] if not r.can_run]
    if bad:
        return f"unrunnable rows in top {c.top}: {bad}"
    return None


def _check_never_in_top(c: Constraint, results) -> str | None:
    hits = [
        f"#{i} {describe(r)} matches {_matches(r.model.id, c.patterns)}"
        for i, r in enumerate(results[: c.top], start=1)
        if _matches(r.model.id, c.patterns)
    ]
    return f"forbidden models in top {c.top}: {hits}" if hits else None


def _check_any_in_top(c: Constraint, results) -> str | None:
    if any(_matches(r.model.id, c.patterns) for r in results[: c.top]):
        return None
    top = [describe(r) for r in results[: c.top]]
    return f"none of {list(c.patterns)} in top {c.top}: {top}"


def _check_stale_generation(c: Constraint, results) -> str | None:
    problems = []
    for old, new in c.pairs:
        newer = [r for r in results if re.search(new, r.model.id)]
        if not newer:
            continue
        stale = [
            f"#{i} {describe(r)}"
            for i, r in enumerate(results[: c.top], start=1)
            if re.search(old, r.model.id)
        ]
        if stale:
            problems.append(
                f"{stale} while newer {describe(newer[0])} is runnable "
                f"({old!r} vs {new!r})"
            )
    return f"stale generation in top {c.top}: {problems}" if problems else None


def _check_top1_offload_speed(c: Constraint, results) -> str | None:
    if not results or results[0].fit_type == "full_gpu":
        return None
    top = results[0]
    speed = top.estimated_tok_per_sec
    if speed is None or speed < (c.min_tok_per_sec or 0.0):
        return (
            f"#1 needs {top.fit_type} but runs below "
            f"{c.min_tok_per_sec} tok/s: {describe(top)}"
        )
    return None


CHECKS = {
    "nonempty": _check_nonempty,
    "never_in_top": _check_never_in_top,
    "any_in_top": _check_any_in_top,
    "stale_generation": _check_stale_generation,
    "top1_offload_speed": _check_top1_offload_speed,
}


def evaluate(case: GoldenCase) -> str | None:
    """Return None when the constraint holds, else a failure description."""
    results = rank_case(case.archetype, case.profile)
    return CHECKS[case.constraint.kind](case.constraint, results)


def render_top_picks() -> str:
    """Render the reviewable top-pick snapshot stored in ``top_picks.md``."""
    meta = load_catalog()["meta"]
    archetypes, _ = load_spec()
    lines = [
        "# Golden top picks",
        "",
        "Generated by `scripts/update_golden.py`; do not edit by hand.",
        "A diff here is a ranking change: review it like code.",
        "",
        f"Catalog captured {meta['captured_at']} with whichllm "
        f"{meta['whichllm_version']} ({meta['git_commit']}): "
        f"{meta['model_count']} models, {meta['benchmark_count']} benchmark ids.",
    ]
    for archetype in archetypes:
        for profile in archetype.profiles:
            context = (
                ""
                if archetype.context_length == CONTEXT_LENGTH
                else f" (context {archetype.context_length})"
            )
            lines += ["", f"## {archetype.id} / {profile}{context}", ""]
            results = rank_case(archetype, profile)[:SNAPSHOT_TOP_N]
            if not results:
                lines.append("(no recommendations)")
            for i, result in enumerate(results, start=1):
                lines.append(f"{i}. {describe(result)}")
    return "\n".join(lines) + "\n"
