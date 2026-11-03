"""Rows for the models a fixed-model engine runs here.

The ranking asks whether llama.cpp can place a model. A fixed-model engine
such as Strata needs less memory for its own models, because it keeps a large
n-gram table on the SSD or reads cold experts from it, so it can run a model
the ranking had to leave out. These rows show that, scored the way the ranking
scores its own. Their speed is not estimated: the engines publish their own
numbers, and whichllm has not measured them yet.
"""

from __future__ import annotations

from whichllm.constants import QUANT_QUALITY_PENALTY
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
# Speed is unknown, so a pack that fits in memory is scored as if it ran at
# the usable bar, neither rewarded nor held back for speed. One that reads
# experts from the SSD is scored like a model with half its layers in RAM,
# which is what "much slower" in the engines' docs comes to.
_SSD_OFFLOAD_RATIO = 0.5


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
    score = _compute_quality_score(
        base,
        variant,
        USABLE_SPEED_TOK_S,
        fit_type,
        offload_ratio=0.0 if in_memory else _SSD_OFFLOAD_RATIO,
        placement="expert_offload" if in_memory else None,
        family_downloads=family_downloads,
        family_likes=family_likes,
        benchmark_avg=bench_avg,
        benchmark_source=evidence.source,
    )
    largest = max((gpu.vram_bytes for gpu in hardware.gpus), default=0)
    card: EngineCard = option.card
    return CompatibilityResult(
        model=base,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=int((pack.requires.vram_gb or 0) * _GiB),
        vram_available_bytes=largest,
        estimated_tok_per_sec=None,
        speed_confidence="low",
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
    )


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
    has nothing better from the same family. Each engine contributes its best
    pack per model, one that fits in memory before one that reads the SSD.
    """
    # Their speed is unknown and they never fit the GPU alone.
    if min_speed is not None or fit_filter == "full_gpu":
        return []
    catalog = {m.id.casefold(): m for m in models}
    downloads, likes, dominant_params = _family_stats(models)
    general_indices = _score_indices(benchmark_scores)
    task_indices = _score_indices(task_scores)
    floor = min((r.quality_score for r in shown), default=0.0)
    best_in_family: dict[str, float] = {}
    for result in shown:
        family = result.model.family_id
        best_in_family[family] = max(
            best_in_family.get(family, 0.0), result.quality_score
        )

    best: dict[tuple[str, str], CompatibilityResult] = {}
    for option in pack_options(cards, hardware, wsl=wsl):
        if not option.fit.runs:
            continue
        base = catalog.get(option.model.base.casefold())
        if base is None:
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
        if row.quality_score <= best_in_family.get(base.family_id, 0.0):
            continue
        key = (option.card.id, option.model.id)
        current = best.get(key)
        if current is None or _preference(row) > _preference(current):
            best[key] = row
    rows = sorted(best.values(), key=lambda r: r.quality_score, reverse=True)
    return rows[: max(limit, 0)]


def _preference(row: CompatibilityResult) -> tuple[bool, float]:
    return row.pack_tier == "memory", row.quality_score


def pack_tiers(row: CompatibilityResult) -> list[str]:
    """Where a pack row's weights live: GPU and RAM, or RAM alone for a CPU
    engine, then the SSD when experts are read from it."""
    tiers = ["RAM"] if row.fit_type == "cpu_only" else ["GPU", "RAM"]
    if row.pack_tier == "ssd":
        tiers.append("SSD")
    return tiers


__all__ = ["PACK_ROW_LIMIT", "pack_tiers", "rank_pack_rows"]
