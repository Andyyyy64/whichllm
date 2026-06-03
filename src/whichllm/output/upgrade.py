"""Upgrade-command Rich output."""

from __future__ import annotations

from rich.table import Table

from whichllm.engine.quantization import effective_quant_type
from whichllm.hardware.types import HardwareInfo
from whichllm.output import _console


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
        f"{current_row['vram_gb']:.0f} GB" if current_row["vram_gb"] else "—",
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
            f"{row['vram_gb']:.0f} GB" if row["vram_gb"] else "—",
            row["top_model"],
            row["top_quant"],
            f"{row['top_quality']:.1f}",
            f"{row['top_tok_s']:.0f}",
            f"{dq:+.1f}",
            f"{ds:+.0f}",
            _upgrade_verdict(dq, ds),
        )

    _console.console.print(table)
    _console.console.print(
        "[dim]Verdict: worth it (≥12pt Q & ≥10 tok/s lift) · meaningful (≥8pt Q or "
        "≥20 tok/s) · marginal · flat (no change) · downgrade.[/]"
    )
