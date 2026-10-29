"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import typer

from whichllm.cli_models import (
    _generate_chat_script,
    _load_models,
    _pick_gguf_variant,
    _resolve_model_deps,
    _search_model,
)
from whichllm.cli_shared import console
from whichllm.commands.autopick import pick_top_runnable
from whichllm.engine.recipe import build_recipe, python_gpu_layers


def run_command(
    *,
    model_name: str | None,
    context_length: int,
    quant: str | None,
    refresh: bool,
    cpu_only: bool,
) -> None:
    """Download and chat with a model. Picks the best one if none specified."""

    if not shutil.which("uv"):
        console.print("[red]uv is required.[/]")
        console.print(
            "Install: [bold]curl -LsSf https://astral.sh/uv/install.sh | sh[/]"
        )
        raise typer.Exit(code=1)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    variant = None
    picked = None
    if model_name:
        model = _search_model(models, model_name)
    else:
        # The same pick as `snippet`: ranked with the index's benchmark scores
        # and cached shapes, then resolved to a real GGUF repository.
        model, variant, picked = pick_top_runnable(
            models,
            context_length=context_length,
            quant=quant,
            cpu_only=cpu_only,
        )
        if variant is not None:
            quant = variant.quant_type

    if variant is None:
        variant = _pick_gguf_variant(model, quant)
    deps, script_type = _resolve_model_deps(model, variant)
    n_gpu_layers = None
    if picked is not None and variant is not None and not cpu_only:
        n_gpu_layers = python_gpu_layers(picked)
        _print_expert_offload_hint(picked, context_length, n_gpu_layers)
    script = _generate_chat_script(
        model, variant, context_length, cpu_only, n_gpu_layers=n_gpu_layers
    )

    fmt = variant.quant_type if variant else script_type.upper()
    console.print(f"\n[bold green]Running {model.id}[/] [dim]({fmt})[/]")
    console.print(f"[dim]Setting up isolated env with: {', '.join(deps)}[/]\n")

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whichllm_run_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        cmd = ["uv", "run", "--no-project"]
        for dep in deps:
            cmd.extend(["--with", dep])
        cmd.append(script_path)
        result = subprocess.run(cmd)
        raise typer.Exit(code=result.returncode)
    finally:
        os.unlink(script_path)


def _print_expert_offload_hint(picked, context_length: int, n_gpu_layers: int) -> None:
    placement = picked.placement
    if placement is None or placement.kind != "expert_offload":
        return
    recipe = build_recipe(picked, context_length)
    console.print(
        "[yellow]Note:[/] llama-cpp-python cannot keep only the MoE experts in "
        f"RAM, so this chat runs {n_gpu_layers} layers on the GPU and is slower "
        "than the estimate."
    )
    if recipe is not None:
        console.print("[dim]For the estimated speed, serve it with llama.cpp:[/]")
        console.print(f"  {recipe.llama_cpp}", markup=False, highlight=False)
