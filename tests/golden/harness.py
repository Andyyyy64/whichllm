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

from whichllm.cli_hardware import _apply_memory_budgets, _apply_ram_bandwidth
from whichllm.commands.ranking import rank_for_hardware
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.gpu_simulator import create_synthetic_gpu
from whichllm.hardware.types import HardwareInfo
from whichllm.models.artifacts import resolve_ranked_gguf_artifact
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
    # One size for every GPU, or one per GPU for a mixed set.
    vram_gb: float | tuple[float, ...] | None = None
    # None means the default (dual-channel DDR4-3200), or the unified
    # bandwidth for Apple GPUs.
    ram_bandwidth_gbps: float | None = None
    cpu_cores: int = 16
    context_length: int = CONTEXT_LENGTH
    note: str = ""

    def hardware(self) -> HardwareInfo:
        sizes = (
            self.vram_gb
            if isinstance(self.vram_gb, tuple)
            else (self.vram_gb,) * len(self.gpus)
        )
        gpus = [
            create_synthetic_gpu(name, size)
            for name, size in zip(self.gpus, sizes, strict=True)
        ]
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
    def gpu_memory_bytes(self) -> tuple[int, ...]:
        """Each GPU's full memory, before headroom or Metal's working set."""
        return tuple(
            g.unified_memory_bytes or g.vram_bytes for g in self.hardware().gpus
        )

    @cached_property
    def gpu_memory_gb(self) -> float:
        """Largest single-GPU memory, the figure people shop by."""
        return max(self.gpu_memory_bytes, default=0) / _GiB


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    rationale: str
    top: int = DISPLAY_TOP_N
    patterns: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    quant: str | None = None
    quants: tuple[str, ...] = ()
    fit_type: str | None = None
    placement: str | None = None
    min_tok_per_sec: float | None = None
    max_tok_per_sec: float | None = None
    min_required_gb: float | None = None
    max_required_gb: float | None = None
    max_gpu_fraction: float | None = None
    max_ram_fraction: float | None = None
    min_params_b: float | None = None
    min_rows: int | None = None
    # fnmatch patterns on archetype ids.
    archetypes: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("general",)
    min_gpu_memory_gb: float | None = None
    max_context_length: int | None = None
    gpu: bool | None = None
    xfail: dict[str, str] = field(default_factory=dict)

    def applies_to(self, archetype: Archetype, profile: str) -> bool:
        if self.archetypes and not any(
            fnmatch(archetype.id, pattern) for pattern in self.archetypes
        ):
            return False
        if profile not in self.profiles:
            return False
        if self.gpu is not None and bool(archetype.gpus) != self.gpu:
            return False
        if (
            self.max_context_length is not None
            and archetype.context_length > self.max_context_length
        ):
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


def catalog_task_scores(profile: str) -> dict[str, float]:
    """The profile's task scores; empty for general or an older catalog."""
    return dict((load_catalog().get("task_scores") or {}).get(profile) or {})


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
            vram_gb=(
                tuple(a["vram_gb"])
                if isinstance(a.get("vram_gb"), list)
                else a.get("vram_gb")
            ),
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
            top=c.get("top", DISPLAY_TOP_N),
            patterns=tuple(c.get("patterns", ())),
            pairs=tuple((p["old"], p["new"]) for p in c.get("pairs", ())),
            quant=c.get("quant"),
            quants=tuple(c.get("quants", ())),
            fit_type=c.get("fit_type"),
            placement=c.get("placement"),
            min_tok_per_sec=c.get("min_tok_per_sec"),
            max_tok_per_sec=c.get("max_tok_per_sec"),
            min_required_gb=c.get("min_required_gb"),
            max_required_gb=c.get("max_required_gb"),
            max_gpu_fraction=c.get("max_gpu_fraction"),
            max_ram_fraction=c.get("max_ram_fraction"),
            min_params_b=c.get("min_params_b"),
            min_rows=c.get("min_rows"),
            archetypes=tuple(c.get("archetypes", ())),
            profiles=tuple(c.get("profiles", ("general",))),
            min_gpu_memory_gb=c.get("min_gpu_memory_gb"),
            max_context_length=c.get("max_context_length"),
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
        for pattern in c.archetypes:
            if not any(fnmatch(archetype_id, pattern) for archetype_id in ids):
                raise ValueError(f"{c.id}: archetypes {pattern!r} matches nothing")
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


_ranking_cache: dict[tuple[str, str, str | None], tuple[CompatibilityResult, ...]] = {}


def rank_case(
    archetype: Archetype, profile: str, quant: str | None = None
) -> tuple[CompatibilityResult, ...]:
    """Rank the frozen catalog for one archetype the way `whichllm` does."""
    key = (archetype.id, profile, quant)
    if key not in _ranking_cache:
        _ranking_cache[key] = tuple(
            _rank(
                archetype.hardware(),
                profile,
                list(catalog_models()),
                archetype.context_length,
                quant,
            )
        )
    return _ranking_cache[key]


def _rank(
    hardware: HardwareInfo,
    profile: str,
    models: list[ModelInfo],
    context_length: int = CONTEXT_LENGTH,
    quant: str | None = None,
) -> list[CompatibilityResult]:
    return rank_for_hardware(
        models,
        hardware,
        context_length=context_length,
        top_n=RANK_DEPTH,
        benchmark_scores=catalog_benchmarks(),
        task_scores=catalog_task_scores(profile),
        profile=profile,
        quant=quant,
    )


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


def _every_row(c: Constraint, results, ok, what: str) -> str | None:
    bad = [
        f"#{i} {describe(r)}"
        for i, r in enumerate(results[: c.top], start=1)
        if not ok(r)
    ]
    return f"{what} in top {c.top}: {bad}" if bad else None


def _check_nonempty(c: Constraint, results, archetype: Archetype) -> str | None:
    if not results:
        return "no recommendations"
    return _every_row(c, results, lambda r: r.can_run, "unrunnable rows")


def _check_never_in_top(c: Constraint, results, archetype: Archetype) -> str | None:
    hits = [
        f"#{i} {describe(r)} matches {_matches(r.model.id, c.patterns)}"
        for i, r in enumerate(results[: c.top], start=1)
        if _matches(r.model.id, c.patterns)
    ]
    return f"forbidden models in top {c.top}: {hits}" if hits else None


def _row_fits_filters(c: Constraint, r: CompatibilityResult) -> bool:
    return (
        (not c.patterns or bool(_matches(r.model.id, c.patterns)))
        and (c.fit_type is None or r.fit_type == c.fit_type)
        and (
            c.placement is None
            or (r.placement is not None and r.placement.kind == c.placement)
        )
        and (
            c.min_required_gb is None
            or r.vram_required_bytes >= c.min_required_gb * _GiB
        )
    )


def _check_any_in_top(c: Constraint, results, archetype: Archetype) -> str | None:
    if any(_row_fits_filters(c, r) for r in results[: c.top]):
        return None
    top = [describe(r) for r in results[: c.top]]
    return f"no row like {list(c.patterns)} {c.fit_type} {c.placement} in top {c.top}: {top}"


def _check_stale_generation(c: Constraint, results, archetype: Archetype) -> str | None:
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


def _check_older_noted(c: Constraint, results, archetype: Archetype) -> str | None:
    problems = [
        f"{describe(r)} does not list {old!r}: {r.older_generations}"
        for old, new in c.pairs
        for r in results
        if re.search(new, r.model.id)
        and not any(re.search(old, model_id) for model_id in r.older_generations)
    ]
    return f"older generation not noted: {problems}" if problems else None


def _check_top_speed(c: Constraint, results, archetype: Archetype) -> str | None:
    bar = c.min_tok_per_sec or 0.0

    def fast(r) -> bool:
        return (r.estimated_tok_per_sec or 0.0) >= bar

    # Only claim the slots the machine can fill with fast candidates.
    slots = min(c.top, sum(1 for r in results if fast(r)))
    if slots == 0:
        return f"no candidate reaches {bar} tok/s"
    slow = [describe(r) for r in results[:slots] if not fast(r)]
    if slow:
        return f"top {slots} has picks below {bar} tok/s: {slow}"
    return None


def _check_row_within(c: Constraint, results, archetype: Archetype) -> str | None:
    row = next((r for r in results if _matches(r.model.id, c.patterns)), None)
    if row is None:
        return f"no row for {list(c.patterns)}"
    speed = row.estimated_tok_per_sec or 0.0
    required = row.vram_required_bytes / _GiB
    outside = (
        (c.min_tok_per_sec is not None and speed < c.min_tok_per_sec)
        or (c.max_tok_per_sec is not None and speed > c.max_tok_per_sec)
        or (c.min_required_gb is not None and required < c.min_required_gb)
        or (c.max_required_gb is not None and required > c.max_required_gb)
    )
    if outside:
        return (
            f"{describe(row)} needs {required:.1f} GB, outside "
            f"{c.min_tok_per_sec}-{c.max_tok_per_sec} tok/s, "
            f"{c.min_required_gb}-{c.max_required_gb} GB"
        )
    return None


def _check_bandwidth_ceiling(
    c: Constraint, results, archetype: Archetype
) -> str | None:
    bandwidth = max(g.memory_bandwidth_gbps or 0 for g in archetype.hardware().gpus)

    def under_ceiling(r: CompatibilityResult) -> bool:
        if r.fit_type != "full_gpu" or r.model.is_moe or r.gguf_variant is None:
            return True
        file_gb = r.gguf_variant.file_size_bytes / 1e9
        return (r.estimated_tok_per_sec or 0.0) <= bandwidth / file_gb

    return _every_row(
        c, results, under_ceiling, f"rows past what {bandwidth:.0f} GB/s can read"
    )


def _check_full_gpu_budget(c: Constraint, results, archetype: Archetype) -> str | None:
    budget = sum(archetype.gpu_memory_bytes) * c.max_gpu_fraction
    return _every_row(
        c,
        results,
        lambda r: r.fit_type != "full_gpu" or r.vram_required_bytes <= budget,
        f"full-GPU rows past {budget / _GiB:.1f} GB",
    )


def _check_ram_budget(c: Constraint, results, archetype: Archetype) -> str | None:
    budget = archetype.ram_gb * _GiB * c.max_ram_fraction
    return _every_row(
        c,
        results,
        lambda r: r.vram_required_bytes - r.vram_available_bytes <= budget,
        f"rows putting more than {budget / _GiB:.1f} GB in RAM",
    )


def _check_gguf_only(c: Constraint, results, archetype: Archetype) -> str | None:
    return _every_row(c, results, lambda r: r.gguf_variant, "non-GGUF rows")


def _check_runnable(c: Constraint, results, archetype: Archetype) -> str | None:
    models = list(catalog_models())

    def runnable(r: CompatibilityResult) -> bool:
        return r.gguf_variant is None or bool(
            resolve_ranked_gguf_artifact(
                r.model, r.gguf_variant, models, quant_filter=None
            )
        )

    return _every_row(c, results, runnable, "rows `run` cannot launch")


def _check_quant_in(c: Constraint, results, archetype: Archetype) -> str | None:
    return _every_row(
        c,
        results,
        lambda r: r.gguf_variant is None or r.gguf_variant.quant_type in c.quants,
        f"rows outside {list(c.quants)}",
    )


def _check_min_params(c: Constraint, results, archetype: Archetype) -> str | None:
    return _every_row(
        c,
        results,
        lambda r: r.model.parameter_count >= c.min_params_b * 1e9,
        f"rows under {c.min_params_b}B",
    )


def _check_task_evidence(c: Constraint, results, archetype: Archetype) -> str | None:
    rows = results[: c.top]
    scored = sum(1 for r in rows if r.benchmark_basis != "general")
    if scored >= c.min_rows:
        return None
    return f"{scored} of top {c.top} ranked on task scores, want {c.min_rows}"


CHECKS = {
    "nonempty": _check_nonempty,
    "never_in_top": _check_never_in_top,
    "any_in_top": _check_any_in_top,
    "stale_generation": _check_stale_generation,
    "older_noted": _check_older_noted,
    "top_speed": _check_top_speed,
    "row_within": _check_row_within,
    "bandwidth_ceiling": _check_bandwidth_ceiling,
    "full_gpu_budget": _check_full_gpu_budget,
    "ram_budget": _check_ram_budget,
    "gguf_only": _check_gguf_only,
    "runnable": _check_runnable,
    "quant_in": _check_quant_in,
    "min_params": _check_min_params,
    "task_evidence": _check_task_evidence,
}


def evaluate(case: GoldenCase) -> str | None:
    """Return None when the constraint holds, else a failure description."""
    c = case.constraint
    results = rank_case(case.archetype, case.profile, c.quant)
    return CHECKS[c.kind](c, results, case.archetype)


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
        f"{meta['model_count']} models, {meta['benchmark_count']} benchmark ids"
        + (
            f" (refreshed {meta['benchmarks_refreshed_at']})."
            if meta.get("benchmarks_refreshed_at")
            else "."
        ),
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
