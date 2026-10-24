from __future__ import annotations

from dataclasses import dataclass, field

from whichllm.engine.placement import DevicePlacement
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
    benchmark_status: str = "none"  # "direct" | "estimated" | "self_reported" | "none"
    benchmark_source: str = "none"  # granular: "direct" | "variant" | "base_model" | "line_interp" | "self_reported" | "none"
    benchmark_confidence: float = 0.0  # 0.0-1.0 from BenchmarkEvidence
    context_fits: bool = True  # False when known model max context < requested
    uses_multi_gpu: bool = False
    multi_gpu_effective_vram_bytes: int | None = None
    # Per-device share of a layer split, populated only for pooled dedicated
    # GPUs. None means the fit used a single device or a shared-memory pool.
    multi_gpu_placement: list[DevicePlacement] | None = None
    artifact_model: ModelInfo | None = None
    artifact_variant: GGUFVariant | None = None
    local_path: str | None = None
