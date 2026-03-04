from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    vendor: str  # "nvidia" | "amd" | "apple" | "intel"
    vram_bytes: int
    compute_capability: tuple[int, int] | None = None  # NVIDIA only
    cuda_version: str | None = None
    rocm_version: str | None = None
    memory_bandwidth_gbps: float | None = None  # from lookup table


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    ram_bytes: int = 0
    disk_free_bytes: int = 0
    os: str = "linux"  # "linux" | "darwin" | "windows"
