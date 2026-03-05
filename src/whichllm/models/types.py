from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GGUFVariant:
    filename: str
    quant_type: str  # "Q4_K_M", "Q8_0" etc
    file_size_bytes: int


@dataclass
class ModelInfo:
    id: str  # HF repo ID
    family_id: str  # grouping key
    name: str
    parameter_count: int  # total parameters
    parameter_count_active: int | None = None  # MoE active params
    architecture: str = ""  # "llama", "qwen2", "mixtral" etc
    is_moe: bool = False
    context_length: int | None = None
    license: str | None = None
    published_at: str | None = None
    downloads: int = 0
    likes: int = 0
    gguf_variants: list[GGUFVariant] = field(default_factory=list)
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    base_model: str | None = None  # cardData.base_model


@dataclass
class ModelFamily:
    family_id: str
    display_name: str
    base_model: ModelInfo
    variants: list[ModelInfo] = field(default_factory=list)
    best_benchmark: dict[str, float] = field(default_factory=dict)
