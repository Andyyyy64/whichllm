"""Rich output formatting for CLI display."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from local_llm_checker.engine.types import CompatibilityResult
from local_llm_checker.hardware.types import HardwareInfo

console = Console()


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


def _format_params(count: int) -> str:
    """Format parameter count."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return str(count)


def display_hardware(hw: HardwareInfo) -> None:
    """Display hardware information panel."""
    lines: list[str] = []

    # GPUs
    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            vram = _format_bytes(gpu.vram_bytes)
            bw = f"{gpu.memory_bandwidth_gbps:.0f} GB/s" if gpu.memory_bandwidth_gbps else "N/A"
            cc = (
                f"CC {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                if gpu.compute_capability
                else ""
            )
            extra = []
            if cc:
                extra.append(cc)
            if gpu.cuda_version:
                extra.append(f"CUDA {gpu.cuda_version}")
            if gpu.rocm_version:
                extra.append(f"ROCm {gpu.rocm_version}")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"[bold green]GPU {i}:[/] {gpu.name} — {vram}{extra_str} — BW: {bw}")
    else:
        lines.append("[yellow]No GPU detected[/] — CPU-only mode")

    # CPU
    avx_flags = []
    if hw.has_avx2:
        avx_flags.append("AVX2")
    if hw.has_avx512:
        avx_flags.append("AVX-512")
    avx_str = f" ({', '.join(avx_flags)})" if avx_flags else ""
    lines.append(f"[bold blue]CPU:[/] {hw.cpu_name} — {hw.cpu_cores} cores{avx_str}")

    # Memory
    lines.append(f"[bold blue]RAM:[/] {_format_bytes(hw.ram_bytes)}")
    lines.append(f"[bold blue]Disk free:[/] {_format_bytes(hw.disk_free_bytes)}")
    lines.append(f"[bold blue]OS:[/] {hw.os}")

    panel = Panel("\n".join(lines), title="[bold]Hardware Info[/]", border_style="blue")
    console.print(panel)


def display_ranking(results: list[CompatibilityResult]) -> None:
    """Display ranked model table."""
    if not results:
        console.print("[yellow]No compatible models found for your hardware.[/]")
        return

    table = Table(title="Recommended Models", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan", min_width=30)
    table.add_column("Params", justify="right", width=8)
    table.add_column("Quant", justify="center", width=8)
    table.add_column("VRAM", justify="right", width=10)
    table.add_column("Speed", justify="right", width=10)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Fit", justify="center", width=10)
    table.add_column("License", width=12)

    for i, r in enumerate(results, 1):
        quant = r.gguf_variant.quant_type if r.gguf_variant else "FP16"
        vram_str = _format_bytes(r.vram_required_bytes)
        speed_str = f"{r.estimated_tok_per_sec:.1f} tok/s" if r.estimated_tok_per_sec else "N/A"
        score_str = f"{r.quality_score:.1f}"

        fit_style = {
            "full_gpu": "[green]Full GPU[/]",
            "partial_offload": "[yellow]Partial[/]",
            "cpu_only": "[red]CPU only[/]",
        }
        fit_str = fit_style.get(r.fit_type, r.fit_type)

        params_str = _format_params(r.model.parameter_count)
        if r.model.is_moe and r.model.parameter_count_active:
            params_str += f" ({_format_params(r.model.parameter_count_active)}a)"

        license_str = r.model.license or "—"

        table.add_row(
            str(i),
            r.model.id,
            params_str,
            quant,
            vram_str,
            speed_str,
            score_str,
            fit_str,
            license_str,
        )

    console.print(table)

    # Show warnings for top results
    for i, r in enumerate(results[:3], 1):
        if r.warnings:
            for w in r.warnings:
                console.print(f"  [yellow]⚠ #{i} {r.model.name}:[/] {w}")


def display_json(results: list[CompatibilityResult], hardware: HardwareInfo) -> None:
    """Output results as JSON."""
    output = {
        "hardware": {
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_bytes": g.vram_bytes,
                    "memory_bandwidth_gbps": g.memory_bandwidth_gbps,
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
                "quant_type": r.gguf_variant.quant_type if r.gguf_variant else "FP16",
                "file_size_bytes": (
                    r.gguf_variant.file_size_bytes if r.gguf_variant else None
                ),
                "vram_required_bytes": r.vram_required_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "quality_score": round(r.quality_score, 2),
                "fit_type": r.fit_type,
                "can_run": r.can_run,
                "warnings": r.warnings,
                "license": r.model.license,
            }
            for i, r in enumerate(results, 1)
        ],
    }
    console.print_json(json.dumps(output, ensure_ascii=False))
