"""Ranking and hardware Rich output surfaces."""

from __future__ import annotations

import re
from math import log10

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.output import _console
from whichllm.output.formatting import (
    _downloads_style,
    _format_bytes,
    _format_downloads,
    _format_params,
    _format_published_at,
    _format_speed,
    _parse_published_at,
    _published_style,
)


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
    if top.benchmark_status == "self_reported":
        return (
            "Low",
            f"uploader-reported benchmark only (unverified), gap +{gap:.1f}{fit_note}",
        )
    if top.benchmark_status == "estimated":
        if gap >= 2.0:
            return "Medium", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
        return "Low", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
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

    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            if gpu.shared_memory:
                vram = (
                    f"{_format_bytes(gpu.vram_bytes)} shared"
                    if gpu.vram_bytes > 0
                    else "shared memory"
                )
            else:
                vram = (
                    "shared memory"
                    if gpu.vendor in ("amd", "intel") and gpu.vram_bytes == 0
                    else _format_bytes(gpu.vram_bytes)
                )
            bw = (
                f"{gpu.memory_bandwidth_gbps:.0f} GB/s"
                if gpu.memory_bandwidth_gbps
                else "N/A"
            )
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
            if (
                gpu.vendor in ("amd", "intel")
                and gpu.vram_bytes > 0
                and not gpu.shared_memory
            ):
                extra.append("shared memory")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            lines.append(
                f"[bold green]GPU {i}:[/] {gpu.name} — {vram}{extra_str} — BW: {bw}"
            )
    else:
        lines.append("[yellow]No GPU detected[/] — CPU-only mode")

    avx_flags = []
    if hw.has_avx2:
        avx_flags.append("AVX2")
    if hw.has_avx512:
        avx_flags.append("AVX-512")
    avx_str = f" ({', '.join(avx_flags)})" if avx_flags else ""
    lines.append(f"[bold blue]CPU:[/] {hw.cpu_name} — {hw.cpu_cores} cores{avx_str}")

    lines.append(f"[bold blue]RAM:[/] {_format_bytes(hw.ram_bytes)}")
    lines.append(f"[bold blue]Disk free:[/] {_format_bytes(hw.disk_free_bytes)}")
    lines.append(f"[bold blue]OS:[/] {hw.os}")

    panel = Panel("\n".join(lines), title="[bold]Hardware Info[/]", border_style="blue")
    _console.console.print(panel)


def display_ranking(
    results: list[CompatibilityResult],
    *,
    has_gpu: bool = True,
    show_status: bool = False,
) -> None:
    """Display ranked model table."""
    if not results:
        _console.console.print(
            "[yellow]No compatible models found for your hardware.[/]"
        )
        return

    mem_label = "VRAM" if has_gpu else "RAM"

    table = Table(title="Recommended Models", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan", min_width=14, overflow="fold")
    table.add_column("Params", justify="right", width=6)
    table.add_column("Quant", justify="center", width=6)
    if show_status:
        table.add_column(mem_label, justify="right", width=8)
        table.add_column("Speed", justify="right", width=8)
        table.add_column("Fit", justify="center", width=7)
    else:
        table.add_column("Published", justify="center", width=10)
        table.add_column("Downloads", justify="right", width=9)
    table.add_column("Score", justify="right", width=5)
    table.add_column("License", width=8)

    download_logs = [
        log10(max(r.model.downloads, 1)) for r in results if r.model.downloads > 0
    ]
    min_download_log = min(download_logs) if download_logs else 0.0
    max_download_log = max(download_logs) if download_logs else 1.0
    published_dates = [_parse_published_at(r.model.published_at) for r in results]
    published_valid = [d for d in published_dates if d is not None]
    oldest_ts = min((d.timestamp() for d in published_valid), default=None)
    newest_ts = max((d.timestamp() for d in published_valid), default=None)

    for i, r in enumerate(results, 1):
        quant = effective_quant_type(r.model, r.gguf_variant)
        vram_str = _format_bytes(r.vram_required_bytes)
        speed_str = _format_speed(r)

        score_val = f"{r.quality_score:.1f}"
        if r.benchmark_status == "none":
            score_str = f"[red]{score_val} ?[/red]"
        elif r.benchmark_status == "self_reported":
            score_str = f"[bright_yellow]{score_val} !sr[/bright_yellow]"
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
        published_dt = _parse_published_at(r.model.published_at)
        published_str = Text(
            _format_published_at(r.model.published_at),
            style=_published_style(published_dt, oldest_ts, newest_ts),
        )
        downloads_str = Text(
            _format_downloads(r.model.downloads),
            style=_downloads_style(
                r.model.downloads, min_download_log, max_download_log
            ),
        )

        params_str = _format_params(r.model.parameter_count)
        if r.model.is_moe and r.model.parameter_count_active:
            params_str += f" ({_format_params(r.model.parameter_count_active)}a)"

        license_str = r.model.license or "—"

        model_link = Text(r.model.id, style="cyan")
        model_link.stylize(f"link https://huggingface.co/{r.model.id}")

        row_cells = [
            str(i),
            model_link,
            params_str,
            quant,
        ]
        if show_status:
            row_cells.extend([vram_str, speed_str, fit_str])
        else:
            row_cells.extend([published_str, downloads_str])
        row_cells.extend([score_str, license_str])
        table.add_row(*row_cells)

    _console.console.print(table)

    has_estimated = any(r.benchmark_status == "estimated" for r in results)
    has_self = any(r.benchmark_status == "self_reported" for r in results)
    has_none = any(r.benchmark_status == "none" for r in results)
    if has_estimated or has_none or has_self:
        parts = []
        if has_self:
            parts.append(
                "[bright_yellow]!sr[/bright_yellow] = uploader-reported only (unverified)"
            )
        if has_estimated:
            parts.append("[yellow]Estimated / ~[/yellow] = inferred from model line")
        if has_none:
            parts.append("[red]None / ?[/red] = no benchmark data")
        _console.console.print(f"  [dim]Score:[/dim]  {',  '.join(parts)}")

    if show_status:
        has_speed_medium = any(r.speed_confidence == "medium" for r in results)
        has_speed_low = any(r.speed_confidence == "low" for r in results)
        if has_speed_medium or has_speed_low:
            parts = []
            if has_speed_medium:
                parts.append("[yellow]~[/yellow] = estimated tok/s range")
            if has_speed_low:
                parts.append("[red]?[/red] = low-confidence/backend-sensitive tok/s")
            _console.console.print(f"  [dim]Speed:[/dim]  {',  '.join(parts)}")

    has_direct = any(r.benchmark_status == "direct" for r in results)
    if not has_direct:
        _console.console.print(
            "  [red]No confirmed winner:[/] direct benchmark data is missing for current candidates."
        )

    confidence, reason = _top_pick_confidence(results)
    confidence_style = {
        "High": "green",
        "Medium": "yellow",
        "Low": "red",
    }[confidence]
    _console.console.print(
        f"  Top pick confidence: [{confidence_style}]{confidence}[/{confidence_style}] ({reason})"
    )

    from whichllm.models.benchmark_sources import BENCHMARK_SNAPSHOT

    _console.console.print(
        f"  [dim]Benchmark reference: {BENCHMARK_SNAPSHOT} curated snapshot; "
        "live AA / LiveBench / Aider merged when reachable.[/dim]"
    )

    # 上位が僅差なら「断定しすぎない」ための注意を表示する
    if len(results) >= 2:
        gap = results[0].quality_score - results[1].quality_score
        if gap < 1.5:
            _console.console.print(
                f"  [yellow]Note:[/] Top candidates are very close (#{1} vs #{2}: {gap:.1f} pts)."
            )

    # 上位に根拠が弱い候補がある場合は目立つ注意を出す
    weak_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.benchmark_status != "direct"
    ]
    if weak_top:
        joined = ", ".join(f"#{i}" for i in weak_top)
        _console.console.print(
            f"  [yellow]Caution:[/] Weaker benchmark evidence in top ranks: {joined}"
        )

    weak_speed_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.speed_confidence == "low"
    ]
    if weak_speed_top:
        joined = ", ".join(f"#{i}" for i in weak_speed_top)
        _console.console.print(
            f"  [yellow]Speed caution:[/] Low-confidence speed estimates in top ranks: {joined}"
        )

    specialized: list[str] = []
    for idx, r in enumerate(results[:10], 1):
        tags = _detect_specializations(r.model.id)
        if tags:
            joined_tags = "/".join(tags)
            specialized.append(f"#{idx} {joined_tags}")
    if specialized:
        _console.console.print(
            "  [yellow]Task hint:[/] Specialized models detected in ranking: "
            + ", ".join(specialized)
        )

    for i, r in enumerate(results[:3], 1):
        if r.warnings:
            for w in r.warnings:
                _console.console.print(f"  [yellow]Warning #{i} {r.model.name}:[/] {w}")
