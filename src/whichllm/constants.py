"""Compatibility shim: curated registries now live under ``whichllm.data``.

This module re-exports the same names so existing imports
(``from whichllm.constants import ...``) keep working. New code should import
from the specific ``whichllm.data.*`` submodule instead.
"""

from whichllm.data.framework import (
    FRAMEWORK_OVERHEAD_BYTES,
    MIN_COMPUTE_CAPABILITY_OLLAMA,
    MIN_COMPUTE_CAPABILITY_VLLM,
)
from whichllm.data.gpu import (
    _GiB,
    AMD_SHARED_MEMORY_APU_MARKERS,
    CURATED_GPU_SPECS,
    CuratedGPUSpec,
    GPU_BANDWIDTH,
    GPU_MEMORY_CLOCK_VARIANTS,
    INTEL_PCI_DEVICE_NAMES,
    NVIDIA_COMPUTE_CAPABILITY,
    VULKAN_ONLY_GPUS,
)
from whichllm.data.lineage import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
)
from whichllm.data.quantization import (
    QUANT_BYTES_PER_WEIGHT,
    QUANT_PREFERENCE_ORDER,
    QUANT_QUALITY_PENALTY,
)

__all__ = [
    "_GiB",
    "AMD_SHARED_MEMORY_APU_MARKERS",
    "CURATED_GPU_SPECS",
    "CuratedGPUSpec",
    "FRAMEWORK_OVERHEAD_BYTES",
    "GPU_BANDWIDTH",
    "GPU_MEMORY_CLOCK_VARIANTS",
    "INTEL_PCI_DEVICE_NAMES",
    "MIN_COMPUTE_CAPABILITY_OLLAMA",
    "MIN_COMPUTE_CAPABILITY_VLLM",
    "MODEL_GENERATION_BONUS_MAX",
    "MODEL_GENERATION_PENALTY_MAX",
    "MODEL_LINEAGE_VERSIONS",
    "NVIDIA_COMPUTE_CAPABILITY",
    "QUANT_BYTES_PER_WEIGHT",
    "QUANT_PREFERENCE_ORDER",
    "QUANT_QUALITY_PENALTY",
    "VULKAN_ONLY_GPUS",
]
