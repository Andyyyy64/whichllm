"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations


import typer

from whichllm.cli_shared import _format_fetch_error, _run_async, console
from whichllm.cli_validation import (
    _auto_min_params_for_profile,
    _validate_profile,
    _validate_ranking_flags,
)


def upgrade_command(
    *,
    target_gpus: list[str],
    context_length: int,
    top: int,
    profile: str,
    cpu_only: bool,
    json_output: bool,
    refresh: bool,
) -> None:
    """Compare the current machine against potential GPU upgrades.

    For each GPU passed on the command line, simulate a system with the same
    CPU/RAM but that GPU, run the ranker, and show the best-N models you'd
    be able to run. Useful for answering "is upgrading from a 3090 to a 4090
    worth it?" — the table shows the quality jump and the speed jump for
    each option.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.hardware.gpu_simulator import create_synthetic_gpu
    from whichllm.hardware.types import HardwareInfo
    from whichllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichllm.models.grouper import group_models
    from whichllm.output.display import display_upgrade, display_upgrade_json

    profile = _validate_profile(profile)
    _validate_ranking_flags(top, None, None)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        current_hw = detect_hardware()
        if cpu_only:
            current_hw.gpus = []

        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models(include_vision=False))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                raise typer.Exit(code=1)

        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception:
                bench_scores = {}

        all_models: list = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        def _rank_for(hw: HardwareInfo):
            min_p = _auto_min_params_for_profile(hw, profile)
            results = rank_models(
                all_models,
                hw,
                context_length=context_length,
                top_n=top,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=min_p,
            )
            if not results and min_p is not None:
                results = rank_models(
                    all_models,
                    hw,
                    context_length=context_length,
                    top_n=top,
                    benchmark_scores=bench_scores,
                    task_profile=profile,
                    require_direct_top=True,
                    min_params_b=None,
                )
            return results

        progress.update(task, description="Ranking current hardware...")
        current_results = _rank_for(current_hw)

        target_results: list[tuple[str, HardwareInfo, list]] = []
        for raw_name in target_gpus:
            progress.update(task, description=f"Ranking {raw_name}...")
            try:
                synthetic = create_synthetic_gpu(raw_name)
            except ValueError as e:
                console.print(f"[yellow]Skipping {raw_name}:[/] {e}")
                continue
            sim_hw = HardwareInfo(
                gpus=[synthetic],
                cpu_name=current_hw.cpu_name,
                cpu_cores=current_hw.cpu_cores,
                has_avx2=current_hw.has_avx2,
                has_avx512=current_hw.has_avx512,
                ram_bytes=current_hw.ram_bytes,
                disk_free_bytes=current_hw.disk_free_bytes,
                os=current_hw.os,
            )
            sim_results = _rank_for(sim_hw)
            target_results.append((raw_name, sim_hw, sim_results))

    if json_output:
        display_upgrade_json(current_hw, current_results, target_results)
    else:
        console.print()
        display_upgrade(current_hw, current_results, target_results)
        console.print()
