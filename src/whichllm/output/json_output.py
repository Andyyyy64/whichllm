"""Machine-readable JSON output for ranking, plan, and upgrade surfaces."""

from __future__ import annotations

import json

from whichllm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output import _console
from whichllm.output.upgrade import _summarize_row


def display_json(results: list[CompatibilityResult], hardware: HardwareInfo) -> None:
    """Output ranking results as JSON."""
    output = {
        "hardware": {
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_bytes": g.vram_bytes,
                    "memory_bandwidth_gbps": g.memory_bandwidth_gbps,
                    "shared_memory": g.shared_memory,
                }
                for g in hardware.gpus
            ],
            "cpu": hardware.cpu_name,
            "cpu_cores": hardware.cpu_cores,
            "ram_bytes": hardware.ram_bytes,
            "os": hardware.os,
        },
        "models": [
            {
                "rank": i,
                "model_id": r.model.id,
                "parameter_count": r.model.parameter_count,
                "published_at": r.model.published_at,
                "downloads": r.model.downloads,
                "quant_type": effective_quant_type(r.model, r.gguf_variant),
                "file_size_bytes": (
                    r.gguf_variant.file_size_bytes
                    if r.gguf_variant
                    else estimate_weight_bytes(r.model, None)
                ),
                "vram_required_bytes": r.vram_required_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "speed_confidence": r.speed_confidence,
                "speed_range_tok_per_sec": (
                    list(r.speed_range_tok_per_sec)
                    if r.speed_range_tok_per_sec
                    else None
                ),
                "speed_notes": r.speed_notes,
                "quality_score": round(r.quality_score, 2),
                "benchmark_status": r.benchmark_status,
                "benchmark_source": r.benchmark_source,
                "benchmark_confidence": round(r.benchmark_confidence, 2),
                "fit_type": r.fit_type,
                "can_run": r.can_run,
                "warnings": r.warnings,
                "license": r.model.license,
            }
            for i, r in enumerate(results, 1)
        ],
    }
    _console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_plan_json(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
) -> None:
    """Output plan results as JSON."""
    from whichllm.constants import (
        GPU_BANDWIDTH,
        QUANT_BYTES_PER_WEIGHT,
        QUANT_QUALITY_PENALTY,
    )
    from whichllm.engine.performance import estimate_tok_per_sec
    from whichllm.engine.vram import estimate_vram
    from whichllm.hardware.types import GPUInfo

    _GiB = 1024**3

    quant_levels = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    vram_by_quant = {}
    for qt in quant_levels:
        bpw = QUANT_BYTES_PER_WEIGHT.get(qt)
        if bpw is None:
            continue
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=qt, file_size_bytes=fake_size
        )
        vram_bytes = estimate_vram(model, fake_variant, context_length)
        vram_by_quant[qt] = {
            "vram_bytes": vram_bytes,
            "quality_loss": QUANT_QUALITY_PENALTY.get(qt, 0.0),
        }

    target_vram = vram_by_quant.get(target_quant.upper(), {}).get("vram_bytes", 0)
    if target_vram == 0:
        bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=target_quant, file_size_bytes=fake_size
        )
        target_vram = estimate_vram(model, fake_variant, context_length)

    _PLAN_GPUS: list[tuple[str, int]] = [
        ("RTX 4060", 8),
        ("RTX 3060", 12),
        ("RTX 4070", 12),
        ("RTX 4080", 16),
        ("RTX 4090", 24),
        ("RX 7900 XTX", 24),
        ("RTX 5090", 32),
        ("A100 40GB", 40),
        ("L40S", 48),
        ("A100 80GB", 80),
        ("H100", 80),
        ("H200", 141),
    ]

    bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
    fake_size = int(model.parameter_count * bpw)
    fake_variant = GGUFVariant(
        filename="", quant_type=target_quant, file_size_bytes=fake_size
    )

    gpus = []
    for gpu_name, vram_gb in _PLAN_GPUS:
        vram_bytes = int(vram_gb * _GiB)
        bandwidth = GPU_BANDWIDTH.get(gpu_name)
        gpu_info = GPUInfo(
            name=gpu_name,
            vendor="nvidia",
            vram_bytes=vram_bytes,
            memory_bandwidth_gbps=bandwidth,
        )
        if vram_bytes >= target_vram:
            fit_type = "full_gpu"
        elif vram_bytes >= target_vram * 0.4:
            fit_type = "partial_offload"
        else:
            fit_type = "too_small"

        speed = None
        if fit_type != "too_small" and bandwidth:
            speed = round(
                estimate_tok_per_sec(model, fake_variant, gpu_info, fit_type), 1
            )

        gpus.append(
            {
                "name": gpu_name,
                "vram_gb": vram_gb,
                "fit_type": fit_type,
                "estimated_tok_per_sec": speed,
            }
        )

    output = {
        "model": {
            "id": model.id,
            "parameter_count": model.parameter_count,
            "architecture": model.architecture,
            "context_length": model.context_length,
            "license": model.license,
        },
        "target_quant": target_quant,
        "context_length": context_length,
        "vram_by_quant": vram_by_quant,
        "gpu_compatibility": gpus,
    }
    _console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_upgrade_json(
    current_hw: HardwareInfo,
    current_results: list,
    target_results: list[tuple[str, HardwareInfo, list]],
) -> None:
    """Emit the upgrade comparison as JSON for scripting."""
    current_row = _summarize_row("Current", current_hw, current_results)
    rows = []
    for name, hw, res in target_results:
        row = _summarize_row(name, hw, res)
        row["delta_quality"] = row["top_quality"] - current_row["top_quality"]
        row["delta_tok_s"] = row["top_tok_s"] - current_row["top_tok_s"]
        rows.append(row)
    _console.console.print_json(
        json.dumps(
            {"current": current_row, "targets": rows},
            ensure_ascii=False,
        )
    )
