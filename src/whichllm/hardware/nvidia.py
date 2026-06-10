"""NVIDIA GPU detection via NVML with nvidia-smi fallback."""

from __future__ import annotations

import logging
import re
import subprocess

from whichllm.constants import NVIDIA_COMPUTE_CAPABILITY, _GiB
from whichllm.hardware.gpu_db import _static_bandwidth, resolve_detected_bandwidth
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_NVIDIA_UNIFIED_MEMORY_MARKERS = ("GB10", "DGX SPARK")


def _lookup_compute_capability(name: str) -> tuple[int, int] | None:
    name_upper = name.upper()
    for key, cc in NVIDIA_COMPUTE_CAPABILITY.items():
        if key.upper() in name_upper:
            return cc
    return None


def _lookup_bandwidth(name: str) -> float | None:
    """Curated GPU_BANDWIDTH lookup. Kept for regression tests; live detection
    goes through ``resolve_detected_bandwidth``, which also consults dbgpu."""
    return _static_bandwidth(name)


def _is_unified_memory_nvidia_gpu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in _NVIDIA_UNIFIED_MEMORY_MARKERS)


def _system_memory_bytes() -> int:
    from whichllm.hardware.memory import detect_ram_bytes

    ram_bytes = detect_ram_bytes()
    if ram_bytes > 0:
        return ram_bytes
    return 128 * _GiB


def _make_nvidia_gpu(
    name: str,
    vram_bytes: int | None,
    cuda_version: str | None = None,
) -> GPUInfo:
    shared_memory = _is_unified_memory_nvidia_gpu(name)
    if shared_memory and (vram_bytes is None or vram_bytes <= 0):
        vram_bytes = _system_memory_bytes()
    elif vram_bytes is None:
        vram_bytes = 0

    return GPUInfo(
        name=name,
        vendor="nvidia",
        vram_bytes=vram_bytes,
        compute_capability=_lookup_compute_capability(name),
        cuda_version=cuda_version,
        memory_bandwidth_gbps=resolve_detected_bandwidth(name, vram_bytes),
        shared_memory=shared_memory,
    )


def _detect_nvidia_gpus_via_smi() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi when Python NVML cannot load."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        logger.debug(f"nvidia-smi fallback failed: {e}")
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2 or not parts[0]:
            continue

        name, memory_mib_text = parts
        match = re.search(r"\d+", memory_mib_text)
        if not match:
            if not _is_unified_memory_nvidia_gpu(name):
                logger.debug(f"Could not parse nvidia-smi memory value: {line!r}")
                continue
            gpus.append(_make_nvidia_gpu(name, None))
            continue

        memory_mib = int(match.group(0))
        gpus.append(_make_nvidia_gpu(name, memory_mib * 1024**2))

    return gpus


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs. Returns empty list on failure."""
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not installed, trying nvidia-smi fallback")
        return _detect_nvidia_gpus_via_smi()

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("NVML init failed, trying nvidia-smi fallback")
        return _detect_nvidia_gpus_via_smi()

    gpus: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        # Get CUDA driver version
        try:
            pynvml.nvmlSystemGetDriverVersion()  # ensure driver is accessible
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except Exception:
            cuda_str = None

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_bytes = mem_info.total
            except pynvml.NVMLError:
                if not _is_unified_memory_nvidia_gpu(name):
                    raise
                logger.debug(f"NVML did not report dedicated memory for {name}")
                vram_bytes = None

            gpus.append(_make_nvidia_gpu(name, vram_bytes, cuda_str))
    except pynvml.NVMLError as e:
        logger.debug(f"Error enumerating NVIDIA GPUs: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if gpus:
        return gpus

    logger.debug("NVML returned no NVIDIA GPUs, trying nvidia-smi fallback")
    return _detect_nvidia_gpus_via_smi()
