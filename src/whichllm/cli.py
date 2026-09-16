"""CLI entry point using typer.

The CLI implementation is split by responsibility:

- ``cli_shared`` for shared console, async, version, and error helpers
- ``cli_validation`` for option validation, hardware overrides, and filters
- ``cli_models`` for model loading, lookup, dependency, and script helpers
- ``cli_commands`` for command execution bodies

This module only defines the Typer app and command signatures.
"""

from __future__ import annotations

from typing import Optional

import typer

from whichllm.cli_commands import (
    hardware_command,
    main_command,
    plan_command,
    run_command,
    snippet_command,
    upgrade_command,
)
from whichllm.cli_shared import _print_version
from whichllm.utils import CONTEXT_LENGTH

app = typer.Typer(
    name="llm-checker",
    help="Find the best LLM that runs on your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=_print_version,
        is_eager=True,
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"
    ),
    min_speed: Optional[float] = typer.Option(
        None, "--min-speed", help="Exact minimum tok/s filter"
    ),
    speed: str = typer.Option(
        "any",
        "--speed",
        help="Speed preset filter: any | usable | fast",
    ),
    fit: str = typer.Option(
        "any",
        "--fit",
        help="Runtime fit filter: any | gpu | full-gpu",
    ),
    gpu_only: bool = typer.Option(
        False,
        "--gpu-only",
        help="Only show models that fit fully in GPU VRAM",
    ),
    evidence: str = typer.Option(
        "any",
        "--evidence",
        help="Benchmark evidence filter: strict | base | any",
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Alias of --evidence strict",
    ),
    status: bool = typer.Option(
        False,
        "--status",
        help="Show runtime columns (default; kept for compatibility)",
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show Downloads metadata instead of runtime columns",
    ),
    min_params: Optional[float] = typer.Option(
        None,
        "--min-params",
        help="Minimum effective parameter size in billions (e.g. 7)",
    ),
    profile: str = typer.Option(
        "general",
        "--profile",
        help="Ranking profile: general | coding | vision | math | any",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    markdown_output: bool = typer.Option(
        False,
        "--markdown",
        "-m",
        help="Output as GitHub-Flavored Markdown",
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[list[str]] = typer.Option(
        None,
        "--gpu",
        help="Simulate GPU(s), e.g. 'RTX 4090', '2x RTX 4090', or repeat --gpu",
    ),
    vram: Optional[float] = typer.Option(
        None,
        "--vram",
        help="Override simulated GPU VRAM or detected GPU usable VRAM in GB",
    ),
    bandwidth: Optional[float] = typer.Option(
        None,
        "--bandwidth",
        "--ram-bandwidth",
        help="Override GPU/RAM bandwidth in GB/s",
    ),
    gpu_index: Optional[int] = typer.Option(
        None,
        "--gpu-index",
        help="Detected GPU index to override when multiple GPUs are present",
    ),
    vram_headroom: str = typer.Option(
        "auto",
        "--vram-headroom",
        help="Reserve GPU memory for runtime overhead: auto | none | 1GB | 10%",
    ),
    ram_budget: Optional[str] = typer.Option(
        None,
        "--ram-budget",
        help="RAM budget for offload: available | 8GB | 50%",
    ),
):
    """Detect hardware and recommend the best local LLMs."""
    return main_command(
        ctx,
        refresh=refresh,
        top=top,
        context_length=context_length,
        quant=quant,
        min_speed=min_speed,
        speed=speed,
        fit=fit,
        gpu_only=gpu_only,
        evidence=evidence,
        direct=direct,
        status=status,
        details=details,
        min_params=min_params,
        profile=profile,
        json_output=json_output,
        markdown_output=markdown_output,
        cpu_only=cpu_only,
        gpu=gpu,
        vram=vram,
        bandwidth=bandwidth,
        gpu_index=gpu_index,
        vram_headroom=vram_headroom,
        ram_budget=ram_budget,
    )


@app.command()
def plan(
    model_name: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Target quantization (default: Q4_K_M)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
):
    """Show what GPU you need to run a specific model."""
    return plan_command(
        model_name=model_name,
        context_length=context_length,
        quant=quant,
        json_output=json_output,
        refresh=refresh,
    )


@app.command()
def upgrade(
    target_gpus: list[str] = typer.Argument(
        ...,
        help="GPUs to compare against (e.g. 'RTX 4090' 'RTX 5090' 'H100')",
    ),
    context_length: int = typer.Option(
        8192,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for ranking (e.g. 8192, 64k, 128k)",
    ),
    top: int = typer.Option(3, "--top", "-n", help="Best-N models to compare per GPU"),
    profile: str = typer.Option("general", "--profile", help="Ranking profile"),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Compare against a CPU-only baseline"
    ),
    json_output: bool = typer.Option(False, "--json"),
    refresh: bool = typer.Option(False, "--refresh"),
):
    """Compare the current machine against potential GPU upgrades.

    For each GPU passed on the command line, simulate a system with the same
    CPU/RAM but that GPU, run the ranker, and show the best-N models you'd
    be able to run. Useful for answering "is upgrading from a 3090 to a 4090
    worth it?" — the table shows the quality jump and the speed jump for
    each option.
    """
    return upgrade_command(
        target_gpus=target_gpus,
        context_length=context_length,
        top=top,
        profile=profile,
        cpu_only=cpu_only,
        json_output=json_output,
        refresh=refresh,
    )


@app.command()
def run(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to run (default: auto-pick best)"
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode"),
):
    """Download and chat with a model. Picks the best one if none specified."""
    return run_command(
        model_name=model_name,
        context_length=context_length,
        quant=quant,
        refresh=refresh,
        cpu_only=cpu_only,
    )


@app.command()
def snippet(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to show snippet for (default: auto-pick best)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
):
    """Print a ready-to-run Python script for a model."""
    return snippet_command(
        model_name=model_name,
        quant=quant,
        refresh=refresh,
    )


@app.command()
def hardware(
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[list[str]] = typer.Option(
        None,
        "--gpu",
        help="Simulate GPU(s), e.g. 'RTX 4090', '2x RTX 4090', or repeat --gpu",
    ),
    vram: Optional[float] = typer.Option(
        None,
        "--vram",
        help="Override simulated GPU VRAM or detected GPU usable VRAM in GB",
    ),
    bandwidth: Optional[float] = typer.Option(
        None,
        "--bandwidth",
        "--ram-bandwidth",
        help="Override GPU/RAM bandwidth in GB/s",
    ),
    gpu_index: Optional[int] = typer.Option(
        None,
        "--gpu-index",
        help="Detected GPU index to override when multiple GPUs are present",
    ),
):
    """Show detected hardware information only."""
    return hardware_command(
        cpu_only=cpu_only,
        gpu=gpu,
        vram=vram,
        bandwidth=bandwidth,
        gpu_index=gpu_index,
    )


__all__ = [
    "app",
    "hardware",
    "main",
    "plan",
    "run",
    "snippet",
    "upgrade",
]


if __name__ == "__main__":
    app()
