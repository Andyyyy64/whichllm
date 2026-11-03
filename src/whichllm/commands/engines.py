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
    with spinner as progress:
        task = (
            progress.add_task("Detecting hardware...", total=None) if progress else None
        )
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram, None, gpu_index)
        if progress:
            progress.remove_task(task)

    rows = sorted(
        (availability(card, hw) for card in load_cards()),
        key=lambda row: (
            not row.runs,
            EVIDENCE.index(row.card.evidence),
            row.card.name.lower(),
        ),
    )
    if json_output:
        display_engines_json(rows, hw)
        return
    console.print()
    display_engines(rows, hw)
    console.print()
