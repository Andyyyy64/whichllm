from __future__ import annotations

from dataclasses import dataclass, field

from whichllm.engine.placement import Placement
from whichllm.models.types import GGUFVariant, ModelInfo


@dataclass
class CompatibilityResult:
    model: ModelInfo
    gguf_variant: GGUFVariant | None
    can_run: bool
    vram_required_bytes: int
    vram_available_bytes: int
    offload_ratio: float = 0.0  # 0.0-1.0 fraction of weights spilled to CPU RAM
    estimated_tok_per_sec: float | None = None
    speed_confidence: str = "medium"  # "high" | "medium" | "low"
    speed_range_tok_per_sec: tuple[float, float] | None = None
    speed_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100 for ranking
    fit_type: str = "full_gpu"  # "full_gpu" | "partial_offload" | "cpu_only"
    # Where the weights live; expert offload and layer offload are both
    # "partial_offload" fits but run at very different speeds.
    placement: Placement | None = None
    benchmark_status: str = "none"  # "direct" | "estimated" | "self_reported" | "none"
    benchmark_source: str = "none"  # granular: "direct" | "variant" | "base_model" | "line_interp" | "self_reported" | "none"
    benchmark_confidence: float = 0.0  # 0.0-1.0 from BenchmarkEvidence
    # Which scores the quality used: "general", or the profile ("coding", ...)
    benchmark_basis: str = "general"
    # Older generations of this model line that ranked below it and were
    # left out of the list.
    older_generations: list[str] = field(default_factory=list)
    context_fits: bool = True  # False when known model max context < requested
    # Longest context with the same placement, set for the rows shown.
    max_context_tokens: int | None = None
    # The engine the recipe uses, the engines that can run this row here,
    # and whether the chosen one was found installed (None when unknown).
    engine: str | None = None
    engines: tuple[str, ...] = ()
    engine_installed: bool | None = None
    # The installed binary's name, when it changes the command: the
    # llama.cpp installer ships one `llama` binary that serves with
    # `llama serve` instead of `llama-server`.
    engine_binary: str | None = None
    uses_multi_gpu: bool = False
    multi_gpu_effective_vram_bytes: int | None = None
    # Decode speed of the layer split relative to the best single card.
    multi_gpu_speed_factor: float = 1.0
    multi_gpu_identical: bool = False
    artifact_model: ModelInfo | None = None
    artifact_variant: GGUFVariant | None = None
    local_path: str | None = None
