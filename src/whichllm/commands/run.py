"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import typer

from whichllm.cli_models import (
    _generate_chat_script,
    _load_models,
    _pick_gguf_variant,
    _resolve_model_deps,
    _search_model,
)
from whichllm.cli_options import _validate_lmstudio_path_flags
from whichllm.cli_shared import console


from whichllm.models.artifacts import (
    resolve_ranked_gguf_artifact,
)
from whichllm.models.lmstudio import (
    LMStudioPathError,
    LocalGGUF,
    discover_lmstudio_ggufs,
    find_local_artifact,
    verify_local_artifact,
)


def _discover_local_models(lm_studio_path: list[Path] | None) -> list[LocalGGUF]:
    """Scan LM Studio libraries, failing clearly on an unreadable explicit path."""
    try:
        return discover_lmstudio_ggufs(lm_studio_path or ())
    except LMStudioPathError as error:
        console.print(f"[red]Error:[/] {error}")
        raise typer.Exit(code=1) from error


def _verified_local_path(
    model_id: str,
    artifact_path: str,
    lm_studio_path: list[Path] | None,
) -> str | None:
    """Return the local artifact to load, re-checking the match before launch."""
    local_models = _discover_local_models(lm_studio_path)
    match = find_local_artifact(model_id, artifact_path, local_models)
    if match is None:
        return None

    verified = verify_local_artifact(match, model_id, artifact_path, local_models)
    if verified is None:
        console.print(
            "[yellow]Warning:[/] Local LM Studio file for "
            f"{model_id} is missing or incomplete; downloading instead."
        )
        return None

    console.print(f"[dim]Using local LM Studio file: {verified}[/]")
    return str(verified)


def run_command(
    *,
    model_name: str | None,
    context_length: int,
    quant: str | None,
    refresh: bool,
    cpu_only: bool,
    lm_studio_path: list[Path] | None = None,
) -> None:
    """Download and chat with a model. Picks the best one if none specified."""

    _validate_lmstudio_path_flags(lm_studio_path)

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
    if model_name:
        model = _search_model(models, model_name)
    else:
        from whichllm.engine.ranker import rank_models
        from whichllm.hardware.detector import detect_hardware
        from whichllm.models.benchmark import load_benchmark_cache
        from whichllm.models.grouper import group_models

        hardware = detect_hardware()
        if cpu_only:
            hardware.gpus = []
        bench_scores = load_benchmark_cache() or {}
        families = group_models(models)
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=5,
            quant_filter=quant,
            benchmark_scores=bench_scores,
        )
        if not results:
            console.print("[red]No runnable model found for your hardware.[/]")
            raise typer.Exit(code=1)
        skipped_gguf: list[str] = []
        model = None
        for ranked in results:
            if ranked.gguf_variant:
                resolved = resolve_ranked_gguf_artifact(
                    ranked.model,
                    ranked.gguf_variant,
                    all_models,
                    quant_filter=quant,
                )
                if resolved:
                    resolved_model, variant = resolved
                    if resolved_model.id != ranked.model.id:
                        console.print(
                            "[dim]Resolved GGUF runtime: "
                            f"{ranked.model.id} -> {resolved_model.id} "
                            f"({variant.quant_type})[/]"
                        )
                    model = resolved_model
                    quant = variant.quant_type
                    break
                skipped_gguf.append(ranked.model.id)
                continue

            model = ranked.model
            break

        if skipped_gguf:
            skipped = ", ".join(skipped_gguf[:3])
            suffix = "..." if len(skipped_gguf) > 3 else ""
            console.print(
                "[yellow]Warning:[/] Skipped GGUF-ranked candidate(s) without "
                f"a matching runnable GGUF repo: {skipped}{suffix}"
            )
        if model is None:
            console.print(
                "[red]Error:[/] Top recommendations require GGUF builds, "
                "but no matching GGUF repos were found."
            )
            console.print(
                "[dim]Try specifying a GGUF model explicitly, for example "
                '`whichllm run "qwen gguf"`.[/]'
            )
            raise typer.Exit(code=1)

    if variant is None:
        variant = _pick_gguf_variant(model, quant)

    local_path = (
        _verified_local_path(model.id, variant.filename, lm_studio_path)
        if variant is not None
        else None
    )
    deps, script_type = _resolve_model_deps(model, variant)
    if local_path is not None:
        # A verified local artifact is opened directly, so the isolated run does
        # not need the Hugging Face client.
        deps = [dep for dep in deps if dep != "huggingface-hub"]
    script = _generate_chat_script(
        model, variant, context_length, cpu_only, local_path=local_path
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
