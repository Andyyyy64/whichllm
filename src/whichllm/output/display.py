"""Rich output formatting for CLI display."""

from __future__ import annotations

import json
import re
from datetime import datetime
from math import log10

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whichllm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo

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


def _format_downloads(downloads: int) -> str:
    """Format download count for compact table display."""
    if downloads >= 1_000_000:
        return f"{downloads / 1_000_000:.1f}M"
    if downloads >= 1_000:
        return f"{downloads / 1_000:.1f}K"
    return str(downloads)


def _format_published_at(value: str | None) -> str:
    """Format published datetime into YYYY-MM-DD."""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return value[:10] if len(value) >= 10 else value


def _format_speed(result: CompatibilityResult) -> str:
    speed = result.estimated_tok_per_sec
    if speed is None:
        return "N/A"
    base = f"{speed:.1f} tok/s"
    if result.speed_confidence == "low":
        return f"[red]{base} ?[/red]"
    if result.speed_confidence == "medium":
        return f"[yellow]{base} ~[/yellow]"
    return base


def _parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _blend_hex(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = _lerp_channel(a[0], b[0], t)
    g = _lerp_channel(a[1], b[1], t)
    bch = _lerp_channel(a[2], b[2], t)
    return f"#{r:02x}{g:02x}{bch:02x}"


def _downloads_style(downloads: int, min_log: float, max_log: float) -> str:
    if downloads <= 0:
        return "grey50"
    dlog = log10(max(downloads, 1))
    span = max(max_log - min_log, 1e-6)
    t = (dlog - min_log) / span
    return _blend_hex((145, 80, 80), (55, 190, 120), t)


def _published_style(
    published: datetime | None,
    oldest_ts: float | None,
    newest_ts: float | None,
) -> str:
    if published is None or oldest_ts is None or newest_ts is None:
        return "grey50"
    pts = published.timestamp()
    span = max(newest_ts - oldest_ts, 1e-6)
    t = (pts - oldest_ts) / span
    return _blend_hex((190, 85, 85), (80, 190, 110), t)


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
        # Uploader-reported eval — never above Low, regardless of gap.
        return (
            "Low",
            f"uploader-reported benchmark only (unverified), gap +{gap:.1f}{fit_note}",
        )
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


def display_ranking(
    results: list[CompatibilityResult],
    *,
    has_gpu: bool = True,
    show_status: bool = False,
) -> None:
    """Display ranked model table."""
    if not results:
        console.print("[yellow]No compatible models found for your hardware.[/]")
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

        # Score with benchmark status indicator
        score_val = f"{r.quality_score:.1f}"
        if r.benchmark_status == "none":
            score_str = f"[red]{score_val} ?[/red]"
        elif r.benchmark_status == "self_reported":
            # Distinct marker so users can spot uploader-claimed numbers.
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

    console.print(table)

    # Score legend
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
        console.print(f"  [dim]Score:[/dim]  {',  '.join(parts)}")

    if show_status:
        has_speed_medium = any(r.speed_confidence == "medium" for r in results)
        has_speed_low = any(r.speed_confidence == "low" for r in results)
        if has_speed_medium or has_speed_low:
            parts = []
            if has_speed_medium:
                parts.append("[yellow]~[/yellow] = estimated tok/s range")
            if has_speed_low:
                parts.append("[red]?[/red] = low-confidence/backend-sensitive tok/s")
            console.print(f"  [dim]Speed:[/dim]  {',  '.join(parts)}")

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

    from whichllm.models.benchmark_sources import BENCHMARK_SNAPSHOT

    console.print(
        f"  [dim]Benchmark reference: {BENCHMARK_SNAPSHOT} curated snapshot; "
        "live AA / LiveBench / Aider merged when reachable.[/dim]"
    )

    # 上位が僅差なら「断定しすぎない」ための注意を表示する
    if len(results) >= 2:
        gap = results[0].quality_score - results[1].quality_score
        if gap < 1.5:
            console.print(
                f"  [yellow]Note:[/] Top candidates are very close (#{1} vs #{2}: {gap:.1f} pts)."
            )

    # 上位に根拠が弱い候補がある場合は目立つ注意を出す
    weak_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.benchmark_status != "direct"
    ]
    if weak_top:
        joined = ", ".join(f"#{i}" for i in weak_top)
        console.print(
            f"  [yellow]Caution:[/] Weaker benchmark evidence in top ranks: {joined}"
        )

    weak_speed_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.speed_confidence == "low"
    ]
    if weak_speed_top:
        joined = ", ".join(f"#{i}" for i in weak_speed_top)
        console.print(
            f"  [yellow]Speed caution:[/] Low-confidence speed estimates in top ranks: {joined}"
        )

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


def display_plan(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
) -> None:
    """Display hardware requirements for a specific model."""
    from whichllm.constants import (
        GPU_BANDWIDTH,
        QUANT_BYTES_PER_WEIGHT,
        QUANT_QUALITY_PENALTY,
    )
    from whichllm.engine.performance import estimate_tok_per_sec
    from whichllm.engine.vram import estimate_vram
    from whichllm.hardware.types import GPUInfo

    _GiB = 1024**3

    # -- Model info panel --
    params = _format_params(model.parameter_count)
    active = ""
    if model.is_moe and model.parameter_count_active:
        active = f" ({_format_params(model.parameter_count_active)} active)"
    ctx = str(model.context_length) if model.context_length else "unknown"

    lines = [
        f"[bold cyan]Model:[/]  {model.id}",
        f"[bold cyan]Params:[/] {params}{active} | Arch: {model.architecture} | Context: {ctx}",
    ]
    if model.license:
        lines.append(f"[bold cyan]License:[/] {model.license}")
    panel = Panel("\n".join(lines), title="[bold]Model Info[/]", border_style="cyan")
    console.print(panel)

    # -- VRAM requirements by quantization --
    quant_levels = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    vram_table = Table(
        title=f"VRAM Required (context: {context_length})", show_lines=True
    )
    vram_table.add_column("Quant", style="bold", width=8)
    vram_table.add_column("VRAM", justify="right", width=10)
    vram_table.add_column("Quality Loss", justify="right", width=12)

    target_vram = 0
    for qt in quant_levels:
        bpw = QUANT_BYTES_PER_WEIGHT.get(qt)
        if bpw is None:
            continue
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=qt, file_size_bytes=fake_size
        )
        vram_bytes = estimate_vram(model, fake_variant, context_length)
        penalty = QUANT_QUALITY_PENALTY.get(qt, 0.0)
        penalty_str = f"-{penalty * 100:.0f}%" if penalty > 0 else "0%"
        marker = " ★" if qt.upper() == target_quant.upper() else ""
        style = "bold green" if qt.upper() == target_quant.upper() else ""
        vram_table.add_row(
            f"{qt}{marker}", _format_bytes(vram_bytes), penalty_str, style=style
        )
        if qt.upper() == target_quant.upper():
            target_vram = vram_bytes

    console.print(vram_table)

    # Ensure target_vram is set
    if target_vram == 0:
        bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=target_quant, file_size_bytes=fake_size
        )
        target_vram = estimate_vram(model, fake_variant, context_length)

    # -- GPU compatibility table --
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

    gpu_table = Table(
        title=f"GPU Compatibility ({target_quant}, {_format_bytes(target_vram)} required)",
        show_lines=True,
    )
    gpu_table.add_column("GPU", style="bold", min_width=14)
    gpu_table.add_column("VRAM", justify="right", width=8)
    gpu_table.add_column("Fit", justify="center", width=12)
    gpu_table.add_column("Est. Speed", justify="right", width=10)

    bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
    fake_size = int(model.parameter_count * bpw)
    fake_variant = GGUFVariant(
        filename="", quant_type=target_quant, file_size_bytes=fake_size
    )

    min_full_gpu = None
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
            fit = "[green]✓ Full GPU[/]"
            fit_type = "full_gpu"
            if min_full_gpu is None:
                min_full_gpu = (gpu_name, vram_gb)
        elif vram_bytes >= target_vram * 0.4:
            fit = "[yellow]~ Partial[/]"
            fit_type = "partial_offload"
        else:
            fit = "[red]✗ Too small[/]"
            fit_type = None

        if fit_type and bandwidth:
            speed = estimate_tok_per_sec(model, fake_variant, gpu_info, fit_type)
            speed_str = f"{speed:.1f} tok/s"
        else:
            speed_str = "—"

        gpu_table.add_row(gpu_name, f"{vram_gb} GB", fit, speed_str)

    console.print(gpu_table)

    if min_full_gpu:
        console.print(
            f"  [green]★[/] Minimum GPU for full offload: "
            f"[bold]{min_full_gpu[0]}[/] ({min_full_gpu[1]} GB) at {target_quant}"
        )
    else:
        console.print(
            f"  [yellow]Note:[/] No single GPU can fully load this model at {target_quant}. "
            "Consider a lower quantization or multi-GPU setup."
        )


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
    console.print_json(json.dumps(output, ensure_ascii=False))


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
    console.print_json(json.dumps(output, ensure_ascii=False))


def _summarize_row(name: str, hw: HardwareInfo, results: list) -> dict:
    """Reduce a (hardware, ranking) pair to one row for the upgrade table."""
    gpu_label = "CPU-only"
    vram_gb = 0.0
    if hw.gpus:
        g = max(hw.gpus, key=lambda x: x.vram_bytes)
        gpu_label = g.name
        vram_gb = g.vram_bytes / 1024**3
    if not results:
        return {
            "name": name,
            "gpu": gpu_label,
            "vram_gb": vram_gb,
            "top_model": "—",
            "top_quality": 0.0,
            "top_tok_s": 0.0,
            "top_speed_confidence": "low",
            "top_speed_range_tok_per_sec": None,
            "top_fit": "—",
            "top_quant": "—",
        }
    r = results[0]
    return {
        "name": name,
        "gpu": gpu_label,
        "vram_gb": vram_gb,
        "top_model": r.model.id,
        "top_quality": float(r.quality_score),
        "top_tok_s": float(r.estimated_tok_per_sec),
        "top_speed_confidence": r.speed_confidence,
        "top_speed_range_tok_per_sec": (
            list(r.speed_range_tok_per_sec) if r.speed_range_tok_per_sec else None
        ),
        "top_fit": r.fit_type,
        "top_quant": (
            r.gguf_variant.quant_type
            if r.gguf_variant
            else effective_quant_type(r.model, None)
        ),
    }


def _upgrade_verdict(delta_q: float, delta_speed: float) -> str:
    """Return a short verdict for an upgrade row."""
    if delta_q >= 12 and delta_speed >= 10:
        return "[bold green]worth it[/]"
    if delta_q >= 8 or delta_speed >= 20:
        return "[green]meaningful[/]"
    if delta_q >= 3 or delta_speed >= 5:
        return "[yellow]marginal[/]"
    if delta_q <= -3 or delta_speed <= -5:
        return "[red]downgrade[/]"
    return "[dim]flat[/]"


def display_upgrade(
    current_hw: HardwareInfo,
    current_results: list,
    target_results: list[tuple[str, HardwareInfo, list]],
) -> None:
    """Render the GPU-upgrade comparison table."""
    current_row = _summarize_row("Current", current_hw, current_results)
    target_rows = [_summarize_row(name, hw, res) for name, hw, res in target_results]

    table = Table(
        title="GPU upgrade comparison",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Setup", style="bold")
    table.add_column("GPU", overflow="fold")
    table.add_column("VRAM", justify="right")
    table.add_column("Best model", overflow="fold")
    table.add_column("Quant")
    table.add_column("Quality", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("ΔQ", justify="right")
    table.add_column("Δtok/s", justify="right")
    table.add_column("Verdict")

    table.add_row(
        current_row["name"],
        current_row["gpu"],
        f"{current_row['vram_gb']:.0f} GB" if current_row["vram_gb"] is not None else "—",
        current_row["top_model"],
        current_row["top_quant"],
        f"{current_row['top_quality']:.1f}",
        f"{current_row['top_tok_s']:.0f}",
        "—",
        "—",
        "—",
    )
    for row in target_rows:
        dq = row["top_quality"] - current_row["top_quality"]
        ds = row["top_tok_s"] - current_row["top_tok_s"]
        table.add_row(
            row["name"],
            row["gpu"],
            f"{row['vram_gb']:.0f} GB" if row["vram_gb"] is not None else "—",
            row["top_model"],
            row["top_quant"],
            f"{row['top_quality']:.1f}",
            f"{row['top_tok_s']:.0f}",
            f"{dq:+.1f}",
            f"{ds:+.0f}",
            _upgrade_verdict(dq, ds),
        )

    console.print(table)
    console.print(
        "[dim]Verdict: worth it (≥12pt Q & ≥10 tok/s lift) · meaningful (≥8pt Q or "
        "≥20 tok/s) · marginal · flat (no change) · downgrade.[/]"
    )


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
    console.print_json(
        json.dumps(
            {"current": current_row, "targets": rows},
            ensure_ascii=False,
        )
    )
