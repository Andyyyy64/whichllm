"""Ranking and hardware Rich output surfaces."""

from __future__ import annotations

import platform
import re
from math import log10

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whichllm.engine.context import context_cell
from whichllm.engine.pack_ranking import pack_tiers
from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.recipe import build_recipe, build_server_recipe
from whichllm.engine.types import CompatibilityResult
from whichllm.engines.cards import card_by_id, install_hint
from whichllm.hardware import ram_bandwidth
from whichllm.hardware.types import HardwareInfo
from whichllm.output import _console
from whichllm.output.formatting import (
    _downloads_style,
    _format_bytes,
    _format_downloads,
    _format_params,
    _format_published_at,
    _format_speed,
    _format_weights,
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


def _artifact_model_id(result: CompatibilityResult) -> str:
    if result.artifact_model:
        return result.artifact_model.id
    return result.model.id


def _top_pick_confidence(results: list[CompatibilityResult]) -> tuple[str, str]:
    """Return confidence level and explanation for top pick."""
    top = results[0]
    gap = (top.quality_score - results[1].quality_score) if len(results) > 1 else 999.0
    notes: list[str] = []
    if top.placement is not None and top.placement.kind == "expert_offload":
        notes.append("MoE experts in RAM")
    elif top.fit_type == "partial_offload":
        notes.append("partial offload")
    elif top.fit_type == "cpu_only":
        notes.append("CPU-only")
    if top.speed_confidence == "low":
        notes.append("low-confidence speed")
    risk_note = f", {', '.join(notes)}" if notes else ""

    if top.benchmark_status == "none":
        return "Low", f"no benchmark data, gap +{gap:.1f}{risk_note}"
    if top.benchmark_status == "self_reported":
        return (
            "Low",
            f"uploader-reported benchmark only (unverified), gap +{gap:.1f}{risk_note}",
        )
    if top.benchmark_status == "estimated":
        if gap >= 2.0:
            confidence = "Medium"
        else:
            confidence = "Low"
        if top.speed_confidence == "low" and confidence == "Medium":
            confidence = "Low"
        return confidence, f"estimated benchmark, gap +{gap:.1f}{risk_note}"
    if gap >= 2.5:
        confidence = "High"
        reason = f"direct benchmark, gap +{gap:.1f}{risk_note}"
    elif gap >= 1.0:
        confidence = "Medium"
        reason = f"direct benchmark, gap +{gap:.1f}{risk_note}"
    else:
        confidence = "Low"
        reason = f"direct benchmark but very close (+{gap:.1f}){risk_note}"

    # オフロード/CPU-only/低信頼speedの1位は実運用で不確実性が高いため信頼度を1段階下げる
    if top.fit_type != "full_gpu" or top.speed_confidence == "low":
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
                vram = _format_bytes(gpu.vram_bytes)
            if (
                gpu.usable_vram_bytes is not None
                and gpu.usable_vram_bytes < gpu.vram_bytes
            ):
                vram += f" (budget {_format_bytes(gpu.usable_vram_bytes)})"
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
            if gpu.pcie_generation and gpu.pcie_lanes:
                extra.append(f"PCIe {gpu.pcie_generation}.0 x{gpu.pcie_lanes}")
            # NVML is the usual path; a fallback says which one found the GPU.
            if gpu.detected_by in ("nvidia-smi", "device-tree"):
                extra.append(f"via {gpu.detected_by}")
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

    ram = _format_bytes(hw.ram_bytes)
    ram_notes = []
    if hw.ram_available_bytes is not None:
        ram_notes.append(f"{_format_bytes(hw.ram_available_bytes)} free")
    if hw.ram_budget_bytes is not None and hw.ram_budget_bytes < hw.ram_bytes:
        ram_notes.append(f"budget {_format_bytes(hw.ram_budget_bytes)}")
    if ram_notes:
        ram += f" ({', '.join(ram_notes)})"
    if hw.ram_bandwidth_gbps:
        how = {
            "detected": "",
            "unified": "",
            "override": "set",
            "estimated": "estimated",
        }.get(hw.ram_bandwidth_source, "assumed")
        measured = (
            f"measured {hw.ram_bandwidth_measured_gbps:.0f} GB/s"
            if hw.ram_bandwidth_measured_gbps
            else None
        )
        detail = ", ".join(
            part for part in (hw.ram_bandwidth_detail, how, measured) if part
        )
        ram += f" — BW: {hw.ram_bandwidth_gbps:.0f} GB/s"
        if detail:
            ram += f" [dim]({detail})[/dim]"
    lines.append(f"[bold blue]RAM:[/] {ram}")
    lines.append(f"[bold blue]Disk free:[/] {_format_bytes(hw.disk_free_bytes)}")
    lines.append(f"[bold blue]OS:[/] {hw.os}")
    for note in hw.budget_notes:
        lines.append(f"[dim]{note}[/dim]")

    panel = Panel("\n".join(lines), title="[bold]Hardware Info[/]", border_style="blue")
    _console.console.print(panel)


def _min_table_width(table: Table) -> int:
    """Width the table needs with every column at its fixed or minimum width."""
    cells = sum((column.width or column.min_width or 0) + 2 for column in table.columns)
    return cells + len(table.columns) + 1


def _context_text(result: CompatibilityResult) -> str:
    """The model's limit, then the longest context this placement still holds."""
    return context_cell(result).replace("→", "\n→")


def _score_text(result: CompatibilityResult) -> str:
    score_val = f"{result.quality_score:.1f}"
    if result.benchmark_status == "none":
        return f"[red]{score_val} ?[/red]"
    if result.benchmark_status == "self_reported":
        return f"[bright_yellow]{score_val} !sr[/bright_yellow]"
    if result.benchmark_status == "estimated":
        return f"[yellow]{score_val} ~[/yellow]"
    return f"[green]{score_val}[/green]"


def _params_text(result: CompatibilityResult) -> str:
    params = _format_params(result.model.parameter_count)
    if result.model.is_moe and result.model.parameter_count_active:
        params += f" ({_format_params(result.model.parameter_count_active)}a)"
    return params


def _engine_name(engine: str | None) -> str:
    card = card_by_id(engine) if engine else None
    return card.name if card else (engine or "")


def _pack_cells(
    result: CompatibilityResult, rank: int, show_status: bool, show_context: bool
) -> list:
    """A row that runs with a fixed-model engine from its own pack."""
    model_link = Text(result.model.id, style="cyan")
    model_link.stylize(f"link https://huggingface.co/{_artifact_model_id(result)}")
    model_link.append(
        f"\nwith {_engine_name(result.engine)}, {result.pack_name}", style="magenta"
    )
    cells: list = [
        str(rank),
        model_link,
        effective_quant_type(result.model, result.gguf_variant),
    ]
    published = Text(_format_published_at(result.model.published_at))
    if show_status:
        model_link.append(f"\n{_params_text(result)}", style="dim")
        tiers = pack_tiers(result)
        if result.pack_tier == "memory":
            fit = f"[green]{'+'.join(tiers)}[/]"
        else:
            fit = f"[yellow]{'+'.join(tiers[:-1])}[/]\n[yellow]+SSD[/]"
        if result.vram_required_bytes:
            fit += f"\n[dim]{_format_bytes(result.vram_required_bytes)}[/dim]"
        if result.estimated_tok_per_sec:
            speed = _format_speed(result)
        else:
            speed = "[grey50]?[/]"
        if result.pack_docs_speed:
            speed += f"\n[dim]docs {result.pack_docs_speed[0]:g}[/dim]"
        cells += [
            _format_weights(result)[0],
            fit,
            speed,
            *([_context_text(result)] if show_context else []),
            published,
        ]
    else:
        cells += [_params_text(result), published, Text("-")]
    cells.append(_score_text(result))
    return cells


def display_ranking(
    results: list[CompatibilityResult],
    *,
    has_gpu: bool = True,
    show_status: bool = False,
    empty_message: str | None = None,
    context_length: int | None = None,
) -> None:
    """Display ranked model table.

    A row with ``pack_launch`` set runs with a fixed-model engine from its own
    pack. It takes its rank like the others, and names the engine and pack.
    """
    if not results:
        _console.console.print(
            f"[yellow]{empty_message or 'No compatible models found for your hardware.'}[/]"
        )
        return

    mem_label = "VRAM" if has_gpu else "RAM"

    table = Table(title="Recommended Models", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan", min_width=14, overflow="fold")
    table.add_column("Quant", justify="center", width=6)
    if show_status:
        table.add_column("Weights", justify="right", width=8)
        table.add_column(f"Fit / {mem_label}", justify="center", width=8)
        table.add_column("Speed", justify="right", width=14)
        table.add_column("Context", justify="right", width=9)
        context_column = table.columns[-1]
        table.add_column("Published", justify="center", width=10)
    else:
        table.add_column("Params", justify="right", width=6)
        table.add_column("Published", justify="center", width=10)
        table.add_column("Downloads", justify="right", width=9)
    table.add_column("Score", justify="right", width=5)
    # Past the terminal's width Rich squeezes the model name to nothing, so a
    # narrow terminal goes without Context instead.
    show_context = show_status and _min_table_width(table) <= _console.console.width
    if show_status and not show_context:
        table.columns.remove(context_column)

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
        if r.pack_launch:
            table.add_row(*_pack_cells(r, i, show_status, show_context))
            continue
        quant = effective_quant_type(r.model, r.gguf_variant)
        vram_str = _format_bytes(r.vram_required_bytes)
        speed_str = _format_speed(r)

        score_str = _score_text(r)

        fit_style = {
            "full_gpu": "[green]Full GPU[/]",
            "partial_offload": "[yellow]Partial[/]",
            "cpu_only": "[red]CPU only[/]",
        }
        fit_str = fit_style.get(r.fit_type, r.fit_type)
        if r.placement is not None and r.placement.kind == "expert_offload":
            fit_str = "[green]GPU+RAM[/]"
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

        params_str = _params_text(r)

        model_link = Text(r.model.id, style="cyan")
        model_link.stylize(f"link https://huggingface.co/{_artifact_model_id(r)}")
        if r.local_path:
            model_link.append("\nInstalled", style="bold green")
        if show_status:
            model_link.append(f"\n{params_str}", style="dim")

        row_cells = [
            str(i),
            model_link,
            quant,
        ]
        if show_status:
            weights_str, _ = _format_weights(r)
            row_cells.extend(
                [
                    weights_str,
                    f"{fit_str}\n[dim]{vram_str}[/dim]",
                    speed_str,
                    *([_context_text(r)] if show_context else []),
                    published_str,
                ]
            )
        else:
            row_cells.append(params_str)
            row_cells.extend([published_str, downloads_str])
        row_cells.append(score_str)
        table.add_row(*row_cells)

    _console.console.print(table)
    pack_rows = [(i, r) for i, r in enumerate(results, 1) if r.pack_launch]
    if pack_rows:
        _console.console.print(
            "  [dim]with <engine> = runs with that engine from its own pack, as its "
            "docs size it. Speed is llama.cpp's for the same file, a floor; docs = "
            "the engine's own measurement.[/dim]"
        )

    if show_status:
        has_estimated_weights = any(_format_weights(r)[1] for r in results)
        if has_estimated_weights:
            _console.console.print(
                "  [dim]Weights:[/dim]  [yellow]~[/yellow] = estimated size"
            )

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
        if show_context and any(
            r.max_context_tokens is not None and "→" in context_cell(r) for r in results
        ):
            _console.console.print(
                "  [dim]Context:[/dim]  the model's limit (? if unknown) → the longest "
                "context this fit holds on your hardware"
            )

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
        "live AA results merged when reachable.[/dim]"
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
    for i, r in enumerate(results[:5], 1):
        if r.older_generations:
            _console.console.print(
                f"  [dim]#{i} {r.model.name} also stands in for older "
                f"generations of its line: {_older_names(r)}[/dim]"
            )

    if context_length is not None and not results[0].pack_launch:
        _print_recipe(results[0], context_length)
    _print_pack_recipes(pack_rows)
    _print_engine_alternatives(results)


def _older_names(result: CompatibilityResult, limit: int = 3) -> str:
    """The publisher's own older releases first, then uploads and finetunes."""
    org = result.model.id.split("/", 1)[0]
    own = [i for i in result.older_generations if i.split("/", 1)[0] == org]
    others = [i for i in result.older_generations if i not in own]
    names = [i.rsplit("/", 1)[-1] for i in (own + others)[:limit]]
    extra = len(result.older_generations) - len(names)
    return ", ".join(names) + (f" and {extra} more" if extra > 0 else "")


def _host_os() -> str:
    name = platform.system().lower()
    return name if name in ("linux", "windows", "darwin") else "linux"


def _print_server_recipe(result: CompatibilityResult) -> None:
    """The run line of a row of Hugging Face weights: vLLM's or SGLang's."""
    recipe = build_server_recipe(result)
    if recipe is None:
        return
    console = _console.console
    card = card_by_id(recipe.engine)
    name = card.name if card else recipe.engine
    installed = " (installed)" if result.engine_installed else ""
    console.print()
    console.print(f"  [bold]Run #1 with {escape(name)}{installed}:[/]")
    console.print(Text(f"    {recipe.command}", style="bold cyan"))
    for engine, command in recipe.others:
        other = card_by_id(engine)
        console.print(
            Text(f"  {other.name if other else engine}: {command}", style="dim")
        )
    # Neither has a Windows build, so under WSL nothing installed on the
    # Windows side can be missed.
    if result.engine_installed is False and card:
        hint = install_hint(card, _host_os())
        if hint:
            console.print(Text(f"  {name} is not installed here; {hint}", style="dim"))


def _print_recipe(result: CompatibilityResult, context_length: int) -> None:
    recipe = build_recipe(result, context_length)
    if recipe is None:
        _print_server_recipe(result)
        return
    console = _console.console
    card = card_by_id(recipe.engine)
    name = card.name if card else recipe.engine
    installed = " (installed)" if result.engine_installed else ""
    console.print()
    if recipe.engine == "lm-studio":
        console.print(f"  [bold]Run #1 in LM Studio{installed}:[/]")
        console.print(
            Text(
                f"    {recipe.gguf_repo} {recipe.quant}, {recipe.lm_studio}",
                style="bold cyan",
            )
        )
    else:
        console.print(f"  [bold]Run #1 with {escape(name)}{installed}:[/]")
        console.print(Text(f"    {recipe.command}", style="bold cyan"))
    if recipe.engine != "llama.cpp":
        console.print(Text(f"  llama.cpp: {recipe.llama_cpp}", style="dim"))
    if recipe.engine != "lm-studio":
        console.print(Text(f"  LM Studio: {recipe.lm_studio}", style="dim"))
    # Under WSL the engines may well be installed on the Windows side, which
    # detection does not look at.
    if result.engine_installed is False and card and not ram_bandwidth._is_wsl():
        hint = install_hint(card, _host_os())
        if hint:
            console.print(Text(f"  {name} is not installed here; {hint}", style="dim"))
    for note in recipe.notes:
        console.print(Text(f"  {note}", style="dim"))


def _print_engine_alternatives(results: list[CompatibilityResult]) -> None:
    """The fixed-model engines that also run a top row's model, with their
    command and the speed their docs report."""
    console = _console.console
    for rank, result in enumerate(results[:3], 1):
        for alternative in result.engine_alternatives:
            name = _engine_name(alternative.engine)
            docs = ""
            if alternative.docs_speed:
                tok, hardware = alternative.docs_speed
                docs = f", {tok:g} tok/s in its docs on {hardware}"
            console.print()
            console.print(
                f"  [bold]#{rank} also runs with {escape(name)}[/] "
                f"[dim]({escape(alternative.pack_name)}{escape(docs)}):[/dim]"
            )
            console.print(Text(f"    {alternative.launch}", style="cyan"))


def _print_pack_recipes(rows: list[tuple[int, CompatibilityResult]]) -> None:
    """How to start each row that runs with a fixed-model engine, by rank, and
    how to install an engine that is missing."""
    console = _console.console
    hinted: set[str] = set()
    for rank, result in rows:
        name = _engine_name(result.engine)
        installed = " (installed)" if result.engine_installed else ""
        console.print()
        console.print(f"  [bold]Run #{rank} with {escape(name)}{installed}:[/]")
        console.print(Text(f"    {result.pack_launch}", style="bold cyan"))
        card = card_by_id(result.engine) if result.engine else None
        if (
            result.engine_installed is False
            and card
            and card.id not in hinted
            and not ram_bandwidth._is_wsl()
        ):
            hint = install_hint(card, _host_os())
            if hint:
                hinted.add(card.id)
                console.print(
                    Text(f"  {name} is not installed here; {hint}", style="dim")
                )
