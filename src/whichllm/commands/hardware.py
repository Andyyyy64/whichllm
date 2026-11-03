"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

from contextlib import nullcontext

from whichllm.cli_shared import console
from whichllm.cli_validation import (
    _apply_gpu_overrides,
    _apply_ram_bandwidth,
    _validate_gpu_flags,
)


def hardware_command(
    *,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None,
    gpu_index: int | None,
    ram_bandwidth: float | None = None,
    json_output: bool = False,
) -> None:
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram, bandwidth, gpu_index, ram_bandwidth)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.hardware.detector import detect_hardware
    from whichllm.hardware.ram_measure import measure_ram_bandwidth_gbps
    from whichllm.output.display import display_hardware
    from whichllm.output.json_output import display_hardware_json

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
        _apply_gpu_overrides(hw, cpu_only, gpu, vram, bandwidth, gpu_index)
        _apply_ram_bandwidth(hw, ram_bandwidth, gpu_bandwidth_set=bandwidth is not None)
        # The CPU cannot reach what a unified-memory GPU reads, so a copy
        # test there would only contradict the figure shown.
        if hw.ram_bandwidth_source != "unified":
            if progress:
                progress.update(task, description="Measuring RAM bandwidth...")
            hw.ram_bandwidth_measured_gbps = measure_ram_bandwidth_gbps(
                hw.ram_available_bytes
            )
        if progress:
            progress.remove_task(task)

    if json_output:
        display_hardware_json(hw)
        return
    console.print()
    display_hardware(hw)
    console.print()
