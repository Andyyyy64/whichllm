"""Plan-command Rich output."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output import _console
from whichllm.output.formatting import _format_bytes, _format_params


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
    _console.console.print(panel)

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

    _console.console.print(vram_table)

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

    _console.console.print(gpu_table)

    if min_full_gpu:
        _console.console.print(
            f"  [green]★[/] Minimum GPU for full offload: "
            f"[bold]{min_full_gpu[0]}[/] ({min_full_gpu[1]} GB) at {target_quant}"
        )
    else:
        _console.console.print(
            f"  [yellow]Note:[/] No single GPU can fully load this model at {target_quant}. "
            "Consider a lower quantization or multi-GPU setup."
        )
