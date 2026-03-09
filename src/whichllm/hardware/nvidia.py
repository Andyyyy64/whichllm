"""NVIDIA GPU detection via pynvml."""

from __future__ import annotations

import logging

from whichllm.constants import GPU_BANDWIDTH, NVIDIA_COMPUTE_CAPABILITY
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)


def _lookup_compute_capability(name: str) -> tuple[int, int] | None:
    name_upper = name.upper()
    for key, cc in NVIDIA_COMPUTE_CAPABILITY.items():
        if key.upper() in name_upper:
            return cc
    return None


def _lookup_bandwidth(name: str) -> float | None:
    name_upper = name.upper()
    # Try longer matches first (e.g. "RTX 4080 SUPER" before "RTX 4080")
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in name_upper:
            return GPU_BANDWIDTH[key]
    return None


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using pynvml. Returns empty list on failure."""
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not installed, skipping NVIDIA detection")
        return []

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("NVML init failed, no NVIDIA driver available")
        return []

    gpus: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        # Get CUDA driver version
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except Exception:
            cuda_str = None

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append(
                GPUInfo(
                    name=name,
                    vendor="nvidia",
                    vram_bytes=mem_info.total,
                    compute_capability=_lookup_compute_capability(name),
                    cuda_version=cuda_str,
                    memory_bandwidth_gbps=_lookup_bandwidth(name),
                )
            )
    except pynvml.NVMLError as e:
        logger.debug(f"Error enumerating NVIDIA GPUs: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return gpus
