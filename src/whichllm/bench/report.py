"""The saved result of a benchmark.

A report holds what the measurement needs to be useful to someone else, and
nothing that identifies the machine's owner: no username, hostname, file path,
serial number, IP address or generated text. Fields are listed one by one for
that reason, so a new field on a hardware object cannot leak in unnoticed.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from whichllm.bench.backends import Backend, Sample
from whichllm.constants import _GiB
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.utils import _cache_dir

SCHEMA_VERSION = 1


def reports_dir() -> Path:
    return _cache_dir() / "reports"


def _hardware_section(hardware: HardwareInfo) -> dict:
    return {
        "os": hardware.os,
        "cpu": hardware.cpu_name,
        "cpu_cores": hardware.cpu_cores,
        "ram_gb": round(hardware.ram_bytes / _GiB, 1),
        "ram_bandwidth_gbps": hardware.ram_bandwidth_gbps,
        "ram_bandwidth_source": hardware.ram_bandwidth_source,
        "gpus": [
            {
                "name": gpu.name,
                "vendor": gpu.vendor,
                "vram_gb": round(gpu.vram_bytes / _GiB, 1),
                "bandwidth_gbps": gpu.memory_bandwidth_gbps,
            }
            for gpu in hardware.gpus
        ],
    }


def _model_section(
    model: ModelInfo | None,
    variant: GGUFVariant | None,
    served_model: str | None,
    context_length: int,
) -> dict:
    return {
        "repo": model.id if model is not None else served_model,
        "filename": variant.filename if variant is not None else None,
        "quant": variant.quant_type if variant is not None else None,
        "file_size_bytes": (
            variant.file_size_bytes
            if variant is not None and not variant.is_estimated
            else None
        ),
        "context_length": context_length,
    }


def _placement_section(picked: CompatibilityResult | None) -> dict | None:
    if picked is None or picked.placement is None:
        return None
    placement = picked.placement
    return {
        "kind": placement.kind,
        "n_cpu_moe": placement.n_cpu_moe,
        "n_gpu_layers": placement.n_gpu_layers,
    }


def _estimate_section(picked: CompatibilityResult | None) -> dict | None:
    if picked is None:
        return None
    speed = picked.estimated_tok_per_sec
    return {
        "tok_per_sec": round(speed, 2) if speed else None,
        "confidence": picked.speed_confidence,
        "required_memory_gb": round(picked.vram_required_bytes / _GiB, 2),
        "fit": picked.fit_type,
    }


def build_report(
    *,
    version: str,
    hardware: HardwareInfo,
    backend: Backend,
    model: ModelInfo | None,
    variant: GGUFVariant | None,
    served_model: str | None,
    context_length: int,
    picked: CompatibilityResult | None,
    samples: list[Sample],
    vram_used_mb: int | None,
) -> dict:
    """The report for one benchmark. Placement and estimate need a ranked pick."""
    speeds = [sample.tok_per_sec for sample in samples]
    return {
        "schema_version": SCHEMA_VERSION,
        "whichllm_version": version,
        "method": "auto",
        "hardware": _hardware_section(hardware),
        "runtime": {
            "engine": backend.engine,
            "app": backend.app,
            "version": backend.version,
        },
        "model": _model_section(model, variant, served_model, context_length),
        "placement": _placement_section(picked),
        "measurement": {
            "tok_per_sec": [round(speed, 2) for speed in speeds],
            "median_tok_per_sec": round(statistics.median(speeds), 2),
            "tokens": samples[0].tokens,
            "vram_used_mb": vram_used_mb,
        },
        "estimate": _estimate_section(picked),
    }


def save_report(report: dict, directory: Path | None = None) -> Path:
    """Write a report under a timestamped name. It stays on this machine."""
    target = directory if directory is not None else reports_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = target / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
