"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations


from whichllm.cli_shared import console
from whichllm.cli_validation import (
    _apply_gpu_overrides,
    _validate_gpu_flags,
)


def hardware_command(
    *,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None,
    gpu_index: int | None,
) -> None:
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram, bandwidth, gpu_index)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.hardware.detector import detect_hardware
    from whichllm.output.display import display_hardware

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram, bandwidth, gpu_index)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()
