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
from whichllm.commands.autopick import pick_top_runnable
from whichllm.engine.recipe import build_recipe, python_gpu_layers
from whichllm.models.lmstudio import (
    LMStudioPathError,
    LocalGGUF,
    discover_lmstudio_ggufs,
    find_incomplete_local_artifact,
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


def _local_artifact_target(model, variant, picked) -> tuple[str, str] | None:
    """Return the repository and artifact the launch would load.

    The auto-pick path can resolve a synthetic variant to a different GGUF
    repository, so the resolved artifact wins over the ranked candidate.
    """
    if (
        picked is not None
        and picked.artifact_model is not None
        and picked.artifact_variant is not None
    ):
        return picked.artifact_model.id, picked.artifact_variant.filename
    if variant is not None:
        return model.id, variant.filename
    return None


def _warn_incomplete_local_set(
    model_id: str,
    artifact_path: str,
    local_models: list[LocalGGUF],
) -> None:
    """Say why a local set is not used, when one was found but is incomplete.

    A library without the artifact at all stays quiet: only an explicit local
    set that cannot load is worth a warning.
    """
    incomplete = find_incomplete_local_artifact(model_id, artifact_path, local_models)
    if incomplete is None:
        return
    local, missing, total = incomplete
    parts = ", ".join(str(part) for part in missing)
    console.print(
        "[yellow]Warning:[/] Local LM Studio set for "
        f"{model_id} in {local.path.parent} is missing part {parts} of {total}; "
        "downloading instead."
    )


def _verified_local_path(
    model_id: str,
    artifact_path: str,
    lm_studio_path: list[Path] | None,
) -> str | None:
    """Return the local artifact to load, re-checking the match before launch."""
    local_models = _discover_local_models(lm_studio_path)
    match = find_local_artifact(model_id, artifact_path, local_models)
    if match is None:
        _warn_incomplete_local_set(model_id, artifact_path, local_models)
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
    """Download and chat with a model. Picks the best one if none specified.

    A GGUF model runs in llama.cpp's llama-cli when it is installed: it
    downloads the file itself, uses the GPU and keeps MoE experts in RAM the
    way the ranking estimated. Otherwise the chat runs on llama-cpp-python (or
    transformers) in an isolated uv environment.

    A GGUF artifact already installed in an LM Studio library is opened from
    there instead of being downloaded again, for either backend.
    """

    _validate_lmstudio_path_flags(lm_studio_path)

    llama_cli = shutil.which("llama-cli")
    if not llama_cli and not shutil.which("uv"):
        _print_uv_required()
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

    target = _local_artifact_target(model, variant, picked)
    local_path = (
        _verified_local_path(target[0], target[1], lm_studio_path)
        if target is not None
        else None
    )

    if llama_cli and variant is not None:
        args = _llama_cli_args(
            model,
            variant,
            picked,
            context_length,
            cpu_only=cpu_only,
            local_path=local_path,
        )
        console.print(
            f"\n[bold green]Running {model.id}[/] [dim]({variant.quant_type}) "
            "with llama.cpp[/]"
        )
        console.print(
            " ".join(["llama-cli", *args]), style="dim", markup=False, highlight=False
        )
        result = subprocess.run([llama_cli, *args])
        raise typer.Exit(code=result.returncode)

    if not shutil.which("uv"):
        _print_uv_required()
        raise typer.Exit(code=1)
    deps, script_type = _resolve_model_deps(model, variant)
    if script_type == "gguf":
        _print_llama_cpp_python_note()
    n_gpu_layers = None
    if picked is not None and variant is not None and not cpu_only:
        n_gpu_layers = python_gpu_layers(picked)
        _print_expert_offload_hint(picked, context_length, n_gpu_layers)
    if local_path is not None:
        # The generated script opens the local file, so the isolated run does
        # not need the Hugging Face client.
        deps = [dep for dep in deps if dep != "huggingface-hub"]
    script = _generate_chat_script(
        model,
        variant,
        context_length,
        cpu_only,
        n_gpu_layers=n_gpu_layers,
        local_path=local_path,
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


def _llama_cli_args(
    model,
    variant,
    picked,
    context_length: int,
    *,
    cpu_only: bool,
    local_path: str | None = None,
) -> list[str]:
    """llama-cli arguments: the printed recipe for a ranked pick.

    A named model has no placement, so llama.cpp's --fit (on by default)
    sizes the GPU layers and MoE experts to the free memory. A verified local
    artifact replaces the repository download: llama.cpp opens the file with
    -m instead of downloading the -hf source.
    """
    recipe = build_recipe(picked, context_length) if picked is not None else None
    if recipe is not None:
        args = recipe.llama_cpp.split()[1:]
    else:
        args = ["-hf", f"{model.id}:{variant.quant_type}", "-c", str(context_length)]
        if cpu_only:
            args += ["-ngl", "0"]
    if local_path is not None and args[:1] == ["-hf"]:
        return ["-m", local_path, *args[2:]]
    return args


def _print_uv_required() -> None:
    console.print("[red]uv is required[/] (or llama.cpp's llama-cli for GGUF models).")
    console.print("Install: [bold]curl -LsSf https://astral.sh/uv/install.sh | sh[/]")


def _print_llama_cpp_python_note() -> None:
    console.print(
        "[yellow]Note:[/] llama-cpp-python is compiled on this machine the first "
        "time, which can take 20 minutes or more, and that build runs on the CPU "
        "only. With llama.cpp installed (brew install llama.cpp, winget install "
        "ggml.llamacpp, or https://github.com/ggml-org/llama.cpp), `whichllm run` "
        "chats through its llama-cli on the GPU instead."
    )


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
