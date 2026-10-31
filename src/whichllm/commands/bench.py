"""`whichllm bench`: measure a model and record it next to the estimate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer
from rich.markup import escape

from whichllm.bench.backends import (
    Backend,
    BenchError,
    LlamaServer,
    LMStudio,
    Ollama,
    OpenAICompatible,
    Sample,
    auto_backend,
    measure,
)
from whichllm.bench.report import build_report, save_report
from whichllm.cli_models import _load_models, _pick_gguf_variant, _search_model
from whichllm.cli_options import _validate_lmstudio_path_flags
from whichllm.cli_shared import console
from whichllm.commands.autopick import pick_top_runnable
from whichllm.commands.ranking import apply_default_budgets, rank_for_hardware
from whichllm.commands.run import (
    _llama_cli_args,
    _local_artifact_target,
    _verified_local_path,
)
from whichllm.engine.recipe import build_recipe
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.artifacts import resolve_ranked_gguf_artifact
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.utils import _current_version

BACKENDS = ("auto", "llama-server", "ollama", "lm-studio", "openai")
_MIN_RUNS = 1
_MIN_TOKENS = 8
_GB = 10**9


def _validate_options(
    backend: str, url: str | None, served_model: str | None, runs: int, tokens: int
) -> None:
    if backend not in BACKENDS:
        console.print(f"[red]Error:[/] --backend must be one of {', '.join(BACKENDS)}")
        raise typer.Exit(code=2)
    if backend == "openai" and not (url and served_model):
        console.print("[red]Error:[/] --backend openai needs --url and --model")
        raise typer.Exit(code=2)
    if runs < _MIN_RUNS or tokens < _MIN_TOKENS:
        console.print(
            f"[red]Error:[/] --runs must be at least {_MIN_RUNS} and --tokens at "
            f"least {_MIN_TOKENS}"
        )
        raise typer.Exit(code=2)


def _detected_hardware(cpu_only: bool) -> HardwareInfo:
    from whichllm.hardware.detector import detect_hardware

    hardware = detect_hardware()
    if cpu_only:
        hardware.gpus = []
    return apply_default_budgets(hardware)


def _rank_named_model(
    models: list[ModelInfo],
    model: ModelInfo,
    hardware: HardwareInfo,
    *,
    context_length: int,
    quant: str | None,
) -> CompatibilityResult | None:
    """The estimate for a model the user named, resolved to a real GGUF."""
    ranked = rank_for_hardware(
        [model],
        hardware,
        context_length=context_length,
        top_n=1,
        quant=quant,
        benchmark_scores={},
        min_params=0.0,
    )
    if not ranked:
        return None
    picked = ranked[0]
    if picked.gguf_variant is not None:
        resolved = resolve_ranked_gguf_artifact(
            picked.model, picked.gguf_variant, models, quant_filter=quant
        )
        if resolved is not None:
            picked.artifact_model, picked.artifact_variant = resolved
    return picked


def _resolve_model(
    models: list[ModelInfo],
    model_name: str | None,
    hardware: HardwareInfo,
    *,
    context_length: int,
    quant: str | None,
    cpu_only: bool,
) -> tuple[ModelInfo, GGUFVariant | None, CompatibilityResult | None]:
    """What to measure: the model and file, and the pick that estimated them."""
    if not model_name:
        return pick_top_runnable(
            models, context_length=context_length, quant=quant, cpu_only=cpu_only
        )
    model = _search_model(models, model_name)
    picked = _rank_named_model(
        models, model, hardware, context_length=context_length, quant=quant
    )
    if picked is not None and picked.artifact_model is not None:
        return picked.artifact_model, picked.artifact_variant, picked
    return model, _pick_gguf_variant(model, quant), picked


def _confirm(prompt: str, yes: bool) -> bool:
    return yes or typer.confirm(prompt, default=False)


def _describe_file(variant: GGUFVariant) -> str:
    return f"{variant.filename} ({variant.file_size_bytes / _GB:.1f} GB)"


def _llama_server_backend(
    model: ModelInfo,
    variant: GGUFVariant,
    picked: CompatibilityResult | None,
    *,
    context_length: int,
    cpu_only: bool,
    lm_studio_path: list[Path] | None,
    yes: bool,
) -> LlamaServer:
    binary = shutil.which("llama-server")
    if binary is None:
        raise BenchError(
            "llama-server is not installed. Try `brew install llama.cpp` or "
            "`winget install ggml.llamacpp`, or pick another --backend."
        )
    target = _local_artifact_target(model, variant, picked)
    local = (
        _verified_local_path(target[0], target[1], lm_studio_path)
        if target is not None
        else None
    )
    local_path = local[0] if local is not None else None
    if local_path is None and not _confirm(
        f"llama.cpp downloads {_describe_file(variant)} from {model.id} unless it "
        "is already cached. Continue?",
        yes,
    ):
        raise typer.Abort()
    args = _llama_cli_args(
        model,
        variant,
        picked,
        context_length,
        cpu_only=cpu_only,
        local_path=local_path,
    )
    return LlamaServer([binary, *args])


def _ollama_name(model: ModelInfo, variant: GGUFVariant, picked) -> str:
    recipe = build_recipe(picked, 0) if picked is not None else None
    if recipe is not None:
        return recipe.ollama.removeprefix("ollama run ")
    return f"hf.co/{model.id}:{variant.quant_type}"


def _make_backend(
    backend: str,
    model: ModelInfo | None,
    variant: GGUFVariant | None,
    picked: CompatibilityResult | None,
    *,
    url: str | None,
    engine: str | None,
    served_model: str | None,
    context_length: int,
    cpu_only: bool,
    lm_studio_path: list[Path] | None,
    yes: bool,
) -> Backend:
    if backend == "openai":
        return OpenAICompatible(url, served_model, engine or "unknown")
    if model is None or variant is None:
        raise BenchError(f"--backend {backend} needs a GGUF build of this model.")
    if backend == "llama-server":
        return _llama_server_backend(
            model,
            variant,
            picked,
            context_length=context_length,
            cpu_only=cpu_only,
            lm_studio_path=lm_studio_path,
            yes=yes,
        )
    if backend == "ollama":
        return Ollama(
            _ollama_name(model, variant, picked),
            context_length,
            confirm_pull=lambda name: _confirm(
                f"Ollama will download {name} ({_describe_file(variant)}). Continue?",
                yes,
            ),
        )
    return LMStudio(variant.filename, context_length, lms=shutil.which("lms") or "lms")


def _gpu_memory_used_mb() -> int | None:
    """Memory in use across NVIDIA GPUs, the only vendor that reports it simply."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return sum(int(line) for line in result.stdout.split())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _run_benchmark(
    backend: Backend, runs: int, tokens: int
) -> tuple[list[Sample], int | None]:
    before = _gpu_memory_used_mb()
    with backend:
        console.print(f"[dim]Warming up, then {runs} runs of {tokens} tokens...[/]")
        samples = measure(backend, runs=runs, tokens=tokens)
        during = _gpu_memory_used_mb()
    used = during - before if before is not None and during is not None else None
    return samples, used


def _print_result(report: dict, path: Path) -> None:
    measured = report["measurement"]
    runs = ", ".join(f"{speed:.1f}" for speed in measured["tok_per_sec"])
    console.print(
        f"\n[bold]Measured[/]  {measured['median_tok_per_sec']:.1f} tok/s "
        f"[dim]({runs})[/]"
    )
    estimate = report["estimate"]
    if estimate is not None and estimate["tok_per_sec"]:
        console.print(
            f"[bold]Estimated[/] {estimate['tok_per_sec']:.1f} tok/s "
            f"[dim]({estimate['confidence']} confidence)[/]"
        )
    if measured["vram_used_mb"] is not None:
        console.print(f"[bold]GPU memory[/] {measured['vram_used_mb']} MB")
    console.print(f"\n[dim]Saved to {escape(str(path))}[/]")


def bench_command(
    *,
    model_name: str | None,
    backend: str,
    url: str | None,
    engine: str | None,
    served_model: str | None,
    quant: str | None,
    context_length: int,
    runs: int,
    tokens: int,
    yes: bool,
    refresh: bool,
    cpu_only: bool,
    lm_studio_path: list[Path] | None = None,
) -> None:
    """Measure a model, compare it with the estimate and save the result."""
    _validate_lmstudio_path_flags(lm_studio_path)
    _validate_options(backend, url, served_model, runs, tokens)

    hardware = _detected_hardware(cpu_only)
    model = variant = picked = None
    try:
        name = auto_backend() if backend == "auto" else backend
        if model_name or name != "openai":
            models = _load_models(refresh)
            model, variant, picked = _resolve_model(
                models,
                model_name,
                hardware,
                context_length=context_length,
                quant=quant,
                cpu_only=cpu_only,
            )
        runtime = _make_backend(
            name,
            model,
            variant,
            picked,
            url=url,
            engine=engine,
            served_model=served_model,
            context_length=context_length,
            cpu_only=cpu_only,
            lm_studio_path=lm_studio_path,
            yes=yes,
        )
        label = served_model or model.id
        console.print(f"\n[bold green]Measuring {escape(label)}[/] [dim]on {name}[/]")
        samples, vram_used_mb = _run_benchmark(runtime, runs, tokens)
    except BenchError as error:
        console.print(f"[red]Error:[/] {escape(str(error))}")
        raise typer.Exit(code=1) from error

    report = build_report(
        version=_current_version(),
        hardware=hardware,
        backend=runtime,
        model=model,
        variant=variant,
        served_model=served_model,
        context_length=context_length,
        picked=picked,
        samples=samples,
        vram_used_mb=vram_used_mb,
    )
    _print_result(report, save_report(report))
