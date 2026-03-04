"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="llm-checker",
    help="Find the best LLM that runs on your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache and re-fetch models"),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(4096, "--context-length", "-c", help="Context length for KV cache estimation"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed", help="Minimum tok/s filter"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from local_llm_checker.engine.ranker import rank_models
    from local_llm_checker.hardware.detector import detect_hardware
    from local_llm_checker.models.cache import load_cache, save_cache
    from local_llm_checker.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from local_llm_checker.models.grouper import group_models
    from local_llm_checker.output.display import display_hardware, display_json, display_ranking

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Detect hardware
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        progress.update(task, description="Hardware detected")

        # Step 2: Fetch models
        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
            progress.update(task, description=f"Loaded {len(models)} models from cache")
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models())
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"Fetched {len(models)} models")
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                sys.exit(1)

        # Step 3: Group and rank
        progress.update(task, description="Ranking models...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=min_speed,
        )

    # Display results
    if json_output:
        display_json(results, hardware)
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(results)
        console.print()


@app.command()
def hardware():
    """Show detected hardware information only."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from local_llm_checker.hardware.detector import detect_hardware
    from local_llm_checker.output.display import display_hardware

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        hw = detect_hardware()
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


if __name__ == "__main__":
    app()
