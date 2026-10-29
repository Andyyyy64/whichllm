from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    vendor: str  # "nvidia" | "amd" | "apple" | "intel"
    vram_bytes: int
    usable_vram_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None  # NVIDIA only
    cuda_version: str | None = None
    rocm_version: str | None = None
    memory_bandwidth_gbps: float | None = None  # from lookup table
    shared_memory: bool = False
    vram_overridden: bool = False
    # Apple Silicon: the whole unified memory, when vram_bytes is only the
    # part Metal lets the GPU use.
    unified_memory_bytes: int | None = None


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    ram_bytes: int = 0
    ram_budget_bytes: int | None = None
    # System RAM bandwidth in GB/s. It bounds CPU-side decode: CPU-only runs,
    # MoE experts kept in RAM and layers that do not fit on the GPU.
    ram_bandwidth_gbps: float | None = None
    # "detected" (memory modules read), "estimated" (from the CPU name),
    # "unified" (Apple or another unified-memory GPU), "override" (flag)
    # or "default" (nothing known; dual-channel DDR4-3200 is assumed).
    ram_bandwidth_source: str = "default"
    ram_bandwidth_detail: str | None = None
    disk_free_bytes: int = 0
    os: str = "linux"  # "linux" | "darwin" | "windows"
    budget_notes: list[str] = field(default_factory=list)
