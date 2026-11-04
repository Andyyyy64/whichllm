"""Rows for the models a fixed-model engine runs here.

The ranking asks whether llama.cpp can place a model. A fixed-model engine
such as Strata needs less memory for its own models, because it keeps a large
n-gram table on the SSD or reads cold experts from it, so it can run a model
the ranking had to leave out. These rows show that, scored the way the ranking
scores its own.

A pack held in memory is timed the way llama.cpp would run the same file with
its experts in RAM. That is a floor: the engines report faster runs (Strata
about three times llama.cpp's on its own packs), and their docs' numbers come
with the row. A pack that reads experts from the SSD is not timed.
"""

from __future__ import annotations

from dataclasses import dataclass

from whichllm.constants import QUANT_QUALITY_PENALTY
from whichllm.engine.moe import routed_expert_fraction
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.engine.placement import Placement
from whichllm.engine.quantization import quant_quality_penalty, resident_weight_bytes
from whichllm.engine.ranking import _family_stats, _lookup_evidence, _score_indices
from whichllm.engine.ranking_filters import (
    _knowledge_capacity_b,
    _matches_profile,
    _passes_evidence_filter,
)
from whichllm.engine.ranking_score import USABLE_SPEED_TOK_S, _compute_quality_score
from whichllm.engine.types import CompatibilityResult
from whichllm.engines.cards import EngineCard, Pack
from whichllm.engines.detect import Installed
from whichllm.engines.packs import PackOption, pack_options
from whichllm.hardware.types import HardwareInfo
from whichllm.models.benchmark import BenchmarkEvidence
from whichllm.models.gguf import quant_table_key
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3
# Pack rows shown under the ranking at most.
PACK_ROW_LIMIT = 3
# A pack that reads experts from the SSD is not timed. It is scored as if it
# ran at the usable bar, with the factor for half its layers in RAM, which is
# what "much slower" in the engines' docs comes to.
_SSD_OFFLOAD_RATIO = 0.5
_ALTERNATIVES_PER_ROW = 2


@dataclass(frozen=True)
class EngineAlternative:
    """A fixed-model engine that also runs a ranked model here, from memory."""

    engine: str
    pack_name: str
    launch: str
    docs_speed: tuple[float, str] | None


def _docs_speed(pack: Pack) -> tuple[float, str] | None:
    if pack.docs_speed is None:
        return None
    return pack.docs_speed.tok_per_sec, pack.docs_speed.hardware


def _floor_speed(
    model: ModelInfo, variant: GGUFVariant, hardware: HardwareInfo, on_cpu: bool
) -> float:
    """llama.cpp's speed for the same file with every routed expert in RAM."""
    bandwidth = hardware.ram_bandwidth_gbps
    if on_cpu or not hardware.gpus:
        return estimate_tok_per_sec(
            model, variant, None, "cpu_only", ram_bandwidth_gbps=bandwidth
        )
    gpu = max(hardware.gpus, key=lambda g: g.vram_bytes)
    resident = resident_weight_bytes(model, variant)
    experts = int(resident * routed_expert_fraction(model))
    if experts <= 0:
        return estimate_tok_per_sec(model, variant, gpu, "full_gpu")
    placement = Placement(
        "expert_offload",
        gpu_weight_bytes=resident - experts,
        cpu_weight_bytes=experts,
        num_layers=model.shape.num_layers if model.shape else None,
    )
    return estimate_tok_per_sec(
        model,
        variant,
        gpu,
        "partial_offload",
        placement=placement,
        ram_bandwidth_gbps=bandwidth,
    )


def _pack_variant(pack: Pack, artifact: ModelInfo | None) -> GGUFVariant:
    """The pack's file in the catalog, else one sized from the card."""
    key = quant_table_key(pack.quant, QUANT_QUALITY_PENALTY)
    if artifact is not None:
        for variant in artifact.gguf_variants:
            if quant_table_key(variant.quant_type, QUANT_QUALITY_PENALTY) == key:
                return variant
    return GGUFVariant(
        filename=f"{pack.id}.gguf",
        quant_type=pack.quant,
        file_size_bytes=int(pack.requires.disk_gb * _GiB),
        is_estimated=True,
    )


def _status(evidence: BenchmarkEvidence) -> str:
    if evidence.score is None:
        return "none"
    if evidence.source in ("direct", "self_reported"):
        return evidence.source
    return "estimated"


def _row(
    option: PackOption,
    base: ModelInfo,
    artifact: ModelInfo | None,
    evidence: BenchmarkEvidence,
    basis: str,
    hardware: HardwareInfo,
    family_downloads: int,
    family_likes: int,
    installed: dict[str, Installed] | None,
) -> CompatibilityResult:
    pack = option.pack
    variant = _pack_variant(pack, artifact)
    bench_avg = None
    if evidence.score is not None:
        bench_avg = evidence.score
        if evidence.source not in ("direct", "self_reported"):
            confidence = max(0.0, min(1.0, evidence.confidence))
            bench_avg = evidence.score * (0.75 + 0.25 * confidence)
    in_memory = option.fit.tier == "memory"
    # An engine that runs here on the CPU only (WARP) is a CPU-only fit, with
    # the ranking's discount for one.
    on_cpu = option.platform is not None and option.platform.accel == "cpu"
    fit_type = "cpu_only" if on_cpu else "partial_offload"
    speed = None
    if in_memory:
        # 0.0 when the GPU's bandwidth is unknown, which is no estimate either.
        speed = _floor_speed(artifact or base, variant, hardware, on_cpu) or None
    docs = _docs_speed(pack)
    notes = []
    if speed is not None:
        notes.append("llama.cpp's speed for this file, a floor for the engine")
    if docs is not None:
        notes.append(f"{option.card.name}'s docs: {docs[0]:g} tok/s on {docs[1]}")
    score = _compute_quality_score(
        base,
        variant,
        speed or USABLE_SPEED_TOK_S,
        fit_type,
        offload_ratio=0.0 if in_memory else _SSD_OFFLOAD_RATIO,
        placement="expert_offload" if in_memory else None,
        family_downloads=family_downloads,
        family_likes=family_likes,
        benchmark_avg=bench_avg,
        benchmark_source=evidence.source,
        artifact_repo=pack.repo,
    )
    largest = max((gpu.vram_bytes for gpu in hardware.gpus), default=0)
    card: EngineCard = option.card
    return CompatibilityResult(
        model=base,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=int((pack.requires.vram_gb or 0) * _GiB),
        vram_available_bytes=largest,
        estimated_tok_per_sec=speed,
        speed_confidence="low",
        speed_notes=notes,
        warnings=[option.fit.reason] if option.fit.reason else [],
        quality_score=score,
        fit_type=fit_type,
        benchmark_status=_status(evidence),
        benchmark_source=evidence.source,
        benchmark_confidence=evidence.confidence,
        benchmark_basis=basis,
        engine=card.id,
        engines=(card.id,),
        engine_installed=None if installed is None else card.id in installed,
        artifact_model=artifact,
        artifact_variant=variant if artifact is not None else None,
        pack_tier=option.fit.tier,
        pack_name=pack.name,
        pack_launch=pack.launch,
        pack_docs_speed=docs,
    )


def attach_engine_alternatives(
    results: list[CompatibilityResult],
    models: list[ModelInfo],
    hardware: HardwareInfo,
    cards: tuple[EngineCard, ...],
    *,
    wsl: bool = False,
) -> None:
    """Name the fixed-model engines that also run a ranked model here.

    For each row that is not itself an engine's pack, up to two engines whose
    pack of the same model family fits in memory are listed, those with a
    speed in their docs first, each with its best-rated pack: the smallest
    quality penalty, then the larger file.
    """
    catalog = {m.id.casefold(): m for m in models}
    best: dict[tuple[str, str], tuple[tuple[float, int], PackOption]] = {}
    for option in pack_options(cards, hardware, wsl=wsl):
        if option.fit.tier != "memory":
            continue
        base = catalog.get(option.model.base.casefold())
        if base is None:
            continue
        repo = option.pack.repo
        artifact = catalog.get(repo.casefold()) if repo else None
        variant = _pack_variant(option.pack, artifact)
        rank = (
            -quant_quality_penalty(base, variant, repo),
            variant.file_size_bytes,
        )
        key = (base.family_id, option.card.id)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, option)
    for result in results:
        if result.pack_launch:
            continue
        family = result.model.family_id
        alternatives = [
            EngineAlternative(
                engine=option.card.id,
                pack_name=option.pack.name,
                launch=option.pack.launch,
                docs_speed=_docs_speed(option.pack),
            )
            for (fam, _), (_, option) in sorted(best.items())
            if fam == family
        ]
        alternatives.sort(key=lambda a: a.docs_speed is None)
        result.engine_alternatives = alternatives[:_ALTERNATIVES_PER_ROW]


def rank_pack_rows(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    cards: tuple[EngineCard, ...],
    *,
    shown: list[CompatibilityResult],
    benchmark_scores: dict[str, float] | None,
    task_scores: dict[str, float] | None = None,
    task_profile: str = "general",
    installed: dict[str, Installed] | None = None,
    wsl: bool = False,
    quant_filter: str | None = None,
    min_speed: float | None = None,
    min_params_b: float | None = None,
    evidence_filter: str = "any",
    fit_filter: str = "any",
    limit: int = PACK_ROW_LIMIT,
) -> list[CompatibilityResult]:
    """Packs that run here and would earn a place in ``shown``, the ranking.

    A row qualifies when it outscores the ranking's last row and the ranking
    has no row of its family. A model llama.cpp already ranks keeps that row,
    and the engine is named beside it instead (attach_engine_alternatives),
    since both are timed alike. Each engine contributes its best pack per
    model, one that fits in memory before one that reads the SSD.
    """
    # They never fit the GPU alone.
    if fit_filter == "full_gpu":
        return []
    catalog = {m.id.casefold(): m for m in models}
    downloads, likes, dominant_params = _family_stats(models)
    general_indices = _score_indices(benchmark_scores)
    task_indices = _score_indices(task_scores)
    floor = min((r.quality_score for r in shown), default=0.0)
    ranked_families = {result.model.family_id for result in shown}

    best: dict[tuple[str, str], CompatibilityResult] = {}
    for option in pack_options(cards, hardware, wsl=wsl):
        if not option.fit.runs:
            continue
        base = catalog.get(option.model.base.casefold())
        if base is None or base.family_id in ranked_families:
            continue
        if quant_filter and option.pack.quant != quant_filter.upper():
            continue
        if not _matches_profile(base, task_profile):
            continue
        if min_params_b is not None and _knowledge_capacity_b(base) < min_params_b:
            continue
        evidence = BenchmarkEvidence(score=None, confidence=0.0, source="none")
        if benchmark_scores:
            evidence = _lookup_evidence(
                base, benchmark_scores, general_indices, dominant_params, None
            )
        basis = "general"
        if task_scores:
            task = _lookup_evidence(
                base, task_scores, task_indices, dominant_params, None
            )
            if task.source in ("direct", "variant", "base_model"):
                evidence, basis = task, task_profile.lower()
        if not _passes_evidence_filter(evidence.source, evidence_filter):
            continue
        repo = option.pack.repo
        artifact = catalog.get(repo.casefold()) if repo else None
        row = _row(
            option,
            base,
            artifact,
            evidence,
            basis,
            hardware,
            downloads.get(base.family_id, 0),
            likes.get(base.family_id, 0),
            installed,
        )
        if row.quality_score <= floor and shown:
            continue
        # A pack read from the SSD is not timed, so a speed floor leaves it out.
        if min_speed is not None and (row.estimated_tok_per_sec or 0) < min_speed:
            continue
        key = (option.card.id, option.model.id)
        current = best.get(key)
        if current is None or _preference(row) > _preference(current):
            best[key] = row
    rows = sorted(best.values(), key=lambda r: r.quality_score, reverse=True)
    return rows[: max(limit, 0)]


def _preference(row: CompatibilityResult) -> tuple[bool, float, int]:
    """Memory before SSD, then score; between packs rated alike, the larger
    file, which keeps more bits."""
    size = row.gguf_variant.file_size_bytes if row.gguf_variant else 0
    return row.pack_tier == "memory", row.quality_score, size


def merge_pack_rows(
    results: list[CompatibilityResult],
    pack_rows: list[CompatibilityResult],
    top_n: int,
) -> list[CompatibilityResult]:
    """The ranking with each pack row placed by its score, cut back to top_n.

    The ranking keeps its own order. A pack row goes in front of the first
    ranked row that scores lower, so it takes a rank like any other model.
    """
    merged = list(results)
    for row in sorted(pack_rows, key=lambda r: r.quality_score, reverse=True):
        at = next(
            (i for i, r in enumerate(merged) if r.quality_score < row.quality_score),
            len(merged),
        )
        merged.insert(at, row)
    return merged[: max(top_n, 0)]


def pack_tiers(row: CompatibilityResult) -> list[str]:
    """Where a pack row's weights live: GPU and RAM, or RAM alone for a CPU
    engine, then the SSD when experts are read from it."""
    tiers = ["RAM"] if row.fit_type == "cpu_only" else ["GPU", "RAM"]
    if row.pack_tier == "ssd":
        tiers.append("SSD")
    return tiers


__all__ = [
    "EngineAlternative",
    "PACK_ROW_LIMIT",
    "attach_engine_alternatives",
    "merge_pack_rows",
    "pack_tiers",
    "rank_pack_rows",
]
