from __future__ import annotations

from dataclasses import dataclass, field

from local_llm_checker.models.types import GGUFVariant, ModelInfo


@dataclass
class CompatibilityResult:
    model: ModelInfo
    gguf_variant: GGUFVariant | None
    can_run: bool
    vram_required_bytes: int
    vram_available_bytes: int
    estimated_tok_per_sec: float | None = None
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100 for ranking
    fit_type: str = "full_gpu"  # "full_gpu" | "partial_offload" | "cpu_only"
