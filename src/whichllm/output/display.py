"""Rich output formatting for CLI display."""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from whichllm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo

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


def _detect_specializations(model_id: str) -> list[str]:
    """Detect task-specialized model hints from repository name."""
    lower = model_id.lower()
    tags: list[str] = []
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.append("coding")
    if re.search(r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)", lower):
        tags.append("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.append("math")
    return tags


def _top_pick_confidence(results: list[CompatibilityResult]) -> tuple[str, str]:
    """Return confidence level and explanation for top pick."""
    top = results[0]
    gap = (top.quality_score - results[1].quality_score) if len(results) > 1 else 999.0
    fit_note = ""
    if top.fit_type == "partial_offload":
        fit_note = ", partial offload"
    elif top.fit_type == "cpu_only":
        fit_note = ", CPU-only"

    if top.benchmark_status == "none":
        return "Low", f"no benchmark data, gap +{gap:.1f}{fit_note}"
    if top.benchmark_status == "estimated":
        if gap >= 2.0:
            return "Medium", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
        return "Low", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
    # direct benchmark
    if gap >= 2.5:
        confidence = "High"
        reason = f"direct benchmark, gap +{gap:.1f}{fit_note}"
    elif gap >= 1.0:
        confidence = "Medium"
        reason = f"direct benchmark, gap +{gap:.1f}{fit_note}"
    else:
        confidence = "Low"
        reason = f"direct benchmark but very close (+{gap:.1f}){fit_note}"

    # オフロード/CPU-onlyの1位は実運用で不確実性が高いため信頼度を1段階下げる
    if top.fit_type != "full_gpu":
        if confidence == "High":
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"
    return confidence, reason


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


def display_ranking(results: list[CompatibilityResult], *, has_gpu: bool = True) -> None:
    """Display ranked model table."""
    if not results:
        console.print("[yellow]No compatible models found for your hardware.[/]")
        return

    mem_label = "VRAM" if has_gpu else "RAM"

    table = Table(title="Recommended Models", show_lines=True, expand=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan", min_width=18, overflow="fold")
    table.add_column("Params", justify="right", width=7)
    table.add_column("Quant", justify="center", width=7)
    table.add_column(mem_label, justify="right", width=9)
    table.add_column("Speed", justify="right", width=9)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Fit", justify="center", width=8)
    table.add_column("License", width=10)

    for i, r in enumerate(results, 1):
        quant = effective_quant_type(r.model, r.gguf_variant)
        vram_str = _format_bytes(r.vram_required_bytes)
        speed_str = f"{r.estimated_tok_per_sec:.1f} tok/s" if r.estimated_tok_per_sec else "N/A"

        # Score with benchmark status indicator
        score_val = f"{r.quality_score:.1f}"
        if r.benchmark_status == "none":
            score_str = f"[red]{score_val} ?[/red]"
        elif r.benchmark_status == "estimated":
            score_str = f"[yellow]{score_val} ~[/yellow]"
        else:
            score_str = f"[green]{score_val}[/green]"

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

        model_link = r.model.id

        row_style = ""
        if r.benchmark_status == "estimated":
            row_style = "yellow"
        elif r.benchmark_status == "none":
            row_style = "red"

        table.add_row(
            str(i),
            model_link,
            params_str,
            quant,
            vram_str,
            speed_str,
            score_str,
            fit_str,
            license_str,
            style=row_style,
        )

    console.print(table)

    # Score legend
    has_estimated = any(r.benchmark_status == "estimated" for r in results)
    has_none = any(r.benchmark_status == "none" for r in results)
    if has_estimated or has_none:
        parts = []
        if has_estimated:
            parts.append("[yellow]Estimated / ~[/yellow] = inferred from model line")
        if has_none:
            parts.append("[red]None / ?[/red] = no benchmark data")
        console.print(f"  [dim]Score:[/dim]  {',  '.join(parts)}")

    has_direct = any(r.benchmark_status == "direct" for r in results)
    if not has_direct:
        console.print(
            "  [red]No confirmed winner:[/] direct benchmark data is missing for current candidates."
        )

    confidence, reason = _top_pick_confidence(results)
    confidence_style = {
        "High": "green",
        "Medium": "yellow",
        "Low": "red",
    }[confidence]
    console.print(
        f"  Top pick confidence: [{confidence_style}]{confidence}[/{confidence_style}] ({reason})"
    )

    # 上位が僅差なら「断定しすぎない」ための注意を表示する
    if len(results) >= 2:
        gap = results[0].quality_score - results[1].quality_score
        if gap < 1.5:
            console.print(
                f"  [yellow]Note:[/] Top candidates are very close (#{1} vs #{2}: {gap:.1f} pts)."
            )

    # 上位に根拠が弱い候補がある場合は目立つ注意を出す
    weak_top = [idx + 1 for idx, r in enumerate(results[:3]) if r.benchmark_status != "direct"]
    if weak_top:
        joined = ", ".join(f"#{i}" for i in weak_top)
        console.print(f"  [yellow]Caution:[/] Weaker benchmark evidence in top ranks: {joined}")

    specialized: list[str] = []
    for idx, r in enumerate(results[:10], 1):
        tags = _detect_specializations(r.model.id)
        if tags:
            joined_tags = "/".join(tags)
            specialized.append(f"#{idx} {joined_tags}")
    if specialized:
        console.print(
            "  [yellow]Task hint:[/] Specialized models detected in ranking: "
            + ", ".join(specialized)
        )

    # Show warnings for top results
    for i, r in enumerate(results[:3], 1):
        if r.warnings:
            for w in r.warnings:
                console.print(f"  [yellow]Warning #{i} {r.model.name}:[/] {w}")


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
                "quant_type": effective_quant_type(r.model, r.gguf_variant),
                "file_size_bytes": (
                    r.gguf_variant.file_size_bytes
                    if r.gguf_variant
                    else estimate_weight_bytes(r.model, None)
                ),
                "vram_required_bytes": r.vram_required_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "quality_score": round(r.quality_score, 2),
                "benchmark_status": r.benchmark_status,
                "fit_type": r.fit_type,
                "can_run": r.can_run,
                "warnings": r.warnings,
                "license": r.model.license,
            }
            for i, r in enumerate(results, 1)
        ],
    }
    console.print_json(json.dumps(output, ensure_ascii=False))
