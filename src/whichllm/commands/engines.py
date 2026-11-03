"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

from contextlib import nullcontext

from whichllm.cli_shared import console
from whichllm.cli_validation import _apply_gpu_overrides, _validate_gpu_flags
from whichllm.engines.cards import EVIDENCE


def engines_command(
    *,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    gpu_index: int | None,
    json_output: bool = False,
) -> None:
    """Show which inference engines run on this machine."""
    _validate_gpu_flags(cpu_only, gpu, vram, None, gpu_index, None)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engines import availability, load_cards
    from whichllm.engines.detect import detect_installed
    from whichllm.hardware import ram_bandwidth
    from whichllm.hardware.detector import detect_hardware
    from whichllm.output.engines import display_engines, display_engines_json

    # The spinner writes to stdout, where the JSON goes.
    spinner = (
        nullcontext()
        if json_output
        else Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        )
    )
    cards = load_cards()
    with spinner as progress:
        task = (
            progress.add_task("Detecting hardware...", total=None) if progress else None
        )
        hw = detect_hardware()
        host_os = hw.os
        _apply_gpu_overrides(hw, cpu_only, gpu, vram, None, gpu_index)
        # What is installed belongs to this machine, not to a simulated one.
        installed = None
        if not gpu:
            if progress:
                progress.update(task, description="Looking for installed engines...")
            installed = detect_installed(cards, host_os)
        if progress:
            progress.remove_task(task)

    # WSL is a fact about this machine, so a simulated GPU leaves it out.
    wsl = installed is not None and ram_bandwidth._is_wsl()
    rows = sorted(
        (availability(card, hw, wsl=wsl) for card in cards),
        key=lambda row: (
            not row.runs,
            installed is not None and row.card.id not in installed,
            EVIDENCE.index(row.card.evidence),
            row.card.name.lower(),
        ),
    )
    if json_output:
        display_engines_json(rows, hw, installed)
        return
    console.print()
    display_engines(rows, hw, installed, wsl=wsl)
    console.print()
