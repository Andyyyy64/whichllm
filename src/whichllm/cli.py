"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import typer
from rich.console import Console

from whichllm.hardware.types import HardwareInfo

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


def _current_version() -> str:
    """Return installed package version."""
    try:
        return version("whichllm")
    except PackageNotFoundError:
        return "unknown"


def _print_version(value: bool) -> None:
    """Print version and exit when --version is requested."""
    if value:
        console.print(_current_version())
        raise typer.Exit()


def _validate_gpu_flags(
    cpu_only: bool, gpu: str | None, vram: float | None,
) -> None:
    """Validate mutual exclusivity of GPU-related flags."""
    if cpu_only and gpu:
        console.print("[red]Error:[/] --cpu-only and --gpu are mutually exclusive.")
        raise typer.Exit(code=1)
    if vram is not None and not gpu:
        console.print("[red]Error:[/] --vram requires --gpu.")
        raise typer.Exit(code=1)


def _validate_profile(profile: str) -> str:
    """Validate ranking profile option."""
    valid = {"general", "coding", "vision", "math", "any"}
    p = profile.lower()
    if p not in valid:
        console.print(
            "[red]Error:[/] --profile must be one of: general, coding, vision, math, any."
        )
        raise typer.Exit(code=1)
    return p


def _apply_gpu_overrides(
    hardware: HardwareInfo, cpu_only: bool, gpu: str | None, vram: float | None,
) -> HardwareInfo:
    """Replace hardware.gpus based on CLI flags."""
    if cpu_only:
        hardware.gpus = []
    elif gpu:
        from whichllm.hardware.gpu_simulator import create_synthetic_gpu

        try:
            hardware.gpus = [create_synthetic_gpu(gpu, vram)]
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
    return hardware


def _auto_min_params_for_profile(hardware: HardwareInfo, profile: str) -> float | None:
    """Pick automatic min-params threshold for strongest general ranking."""
    if profile != "general":
        return None
    if not hardware.gpus:
        return 3.0
    best_vram_gb = max(g.vram_bytes for g in hardware.gpus) / (1024**3)
    if best_vram_gb >= 30:
        return 12.0
    if best_vram_gb >= 20:
        return 10.0
    if best_vram_gb >= 12:
        return 8.0
    return 7.0


def _include_vision_candidates(profile: str) -> bool:
    """候補取得時にVLMを含めるべきプロファイルか判定する。"""
    return profile.lower() in {"vision", "any"}


def _fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    """上位表示で欠けている公開日時を補完し、更新有無を返す。"""
    missing_ids = [r.model.id for r in results if not r.model.published_at]
    if not missing_ids:
        return False
    published_map = _run_async(fetch_model_published_at(missing_ids))
    if not published_map:
        return False

    updated = False
    for model in all_models:
        published_at = published_map.get(model.id)
        if published_at and not model.published_at:
            model.published_at = published_at
            updated = True
    return updated


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
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache and re-fetch models"),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(4096, "--context-length", "-c", help="Context length for KV cache estimation"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed", help="Minimum tok/s filter"),
    status: bool = typer.Option(
        False,
        "--status",
        help="Show runtime status columns (Speed/Fit) in ranking table",
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
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Override VRAM in GB (requires --gpu)"),
):
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram)
    profile = _validate_profile(profile)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import (
        dicts_to_models,
        fetch_model_published_at,
        fetch_models,
        models_to_dicts,
    )
    from whichllm.models.grouper import group_models
    from whichllm.output.display import display_hardware, display_json, display_ranking

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Detect hardware
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram)
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
                models = _run_async(fetch_models(include_vision=_include_vision_candidates(profile)))
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"Fetched {len(models)} models")
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                sys.exit(1)

        # Step 3: Fetch benchmark scores
        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                progress.update(task, description="Fetching benchmark scores...")
                bench_scores = fetch_benchmark_scores()
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        # Step 4: Group and rank
        progress.update(task, description="Ranking models...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        # general用途はGPUクラスに応じた自動しきい値で小さすぎるモデルを抑制する
        auto_min_params = (
            _auto_min_params_for_profile(hardware, profile)
            if min_params is None
            else min_params
        )

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=min_speed,
            benchmark_scores=bench_scores,
            task_profile=profile,
            require_direct_top=True,
            min_params_b=auto_min_params,
        )

        # 自動しきい値で候補ゼロなら緩和して表示を維持する
        if not results and auto_min_params is not None and min_params is None:
            results = rank_models(
                all_models,
                hardware,
                context_length=context_length,
                top_n=top,
                quant_filter=quant,
                min_speed=min_speed,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=None,
            )

        # 上位候補の公開日時が欠けている場合のみ補完して表示品質を上げる
        if results:
            try:
                if _fill_missing_published_at(all_models, results, fetch_model_published_at):
                    save_cache(models_to_dicts(models))
            except Exception as e:
                progress.update(task, description=f"Published date backfill skipped: {e}")

    # Display results
    if json_output:
        display_json(results, hardware)
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(results, has_gpu=bool(hardware.gpus), show_status=status)
        console.print()


@app.command()
def hardware(
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Override VRAM in GB (requires --gpu)"),
):
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.hardware.detector import detect_hardware
    from whichllm.output.display import display_hardware

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


if __name__ == "__main__":
    app()
