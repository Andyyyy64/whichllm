"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

import sys

import typer

from whichllm.cli_models import (
    _search_model,
)
from whichllm.cli_shared import _format_fetch_error, _run_async, console


from whichllm.cli_models import _looks_like_hf_repo_id, _raise_repo_fetch_error


def plan_command(
    *,
    model_name: str,
    context_length: int,
    quant: str | None,
    json_output: bool,
    refresh: bool,
) -> None:
    """Show what GPU you need to run a specific model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichllm.models.hf import fetch_model_by_id
    from whichllm.output.display import display_plan, display_plan_json

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        cached_data = None if refresh else load_cache()
        models = dicts_to_models(cached_data) if cached_data is not None else []
        query_lower = model_name.lower()
        model = next((m for m in models if m.id.lower() == query_lower), None)

        if model is None and _looks_like_hf_repo_id(model_name):
            progress.update(task, description="Fetching repository from HuggingFace...")
            try:
                model = _run_async(fetch_model_by_id(model_name))
            except Exception as e:
                _raise_repo_fetch_error(model_name, e)
            if model is None:
                console.print(
                    f"[red]Hugging Face repository '{model_name}' does not expose "
                    "enough model metadata to estimate memory.[/]"
                )
                raise typer.Exit(code=1)
        elif cached_data is None:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models(include_vision=True))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                sys.exit(1)

    if model is None:
        model = _search_model(models, model_name)

    target_quant = quant.upper() if quant else "Q4_K_M"

    if json_output:
        display_plan_json(model, context_length, target_quant)
    else:
        console.print()
        display_plan(model, context_length, target_quant)
        console.print()
