"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations


import typer

from whichllm.cli_shared import _format_fetch_error, _run_async, console
from whichllm.cli_validation import (
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

    from dataclasses import replace

    from whichllm.commands.ranking import apply_default_budgets, rank_for_hardware
    from whichllm.hardware.detector import detect_hardware
    from whichllm.hardware.gpu_simulator import create_synthetic_gpu
    from whichllm.hardware.types import HardwareInfo
    from whichllm.models.benchmark import (
        fetch_benchmark_data,
        load_benchmark_cache,
        load_task_cache,
        save_benchmark_cache,
    )
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import (
        ModelFetchReport,
        dicts_to_models,
        fetch_models,
        models_to_dicts,
    )
    from whichllm.models.grouper import group_models
    from whichllm.models.index_client import load_index
    from whichllm.models.shape_fetch import attach_cached_shapes
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
        apply_default_budgets(current_hw)

        progress.update(task, description="Loading models...")
        catalog = load_index(refresh=refresh)
        cached_data = None if refresh or catalog is not None else load_cache()
        if catalog is not None:
            models = catalog.models
        elif cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                report = ModelFetchReport()
                models = _run_async(fetch_models(include_vision=False, report=report))
                save_cache(models_to_dicts(models), complete=report.complete)
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                raise typer.Exit(code=1)

        progress.update(task, description="Loading benchmark data...")
        if catalog is not None:
            bench_scores = catalog.benchmark_scores
            task_scores = catalog.task_scores
        else:
            bench_scores = None if refresh else load_benchmark_cache()
            task_scores = load_task_cache()
        if bench_scores is None:
            try:
                data = _run_async(fetch_benchmark_data(m.id for m in models))
                bench_scores, task_scores = data.general, data.tasks
                save_benchmark_cache(bench_scores, task_scores)
            except Exception:
                bench_scores, task_scores = {}, {}

        all_models: list = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)
        attach_cached_shapes(all_models)

        def _rank_for(hw: HardwareInfo):
            return rank_for_hardware(
                all_models,
                hw,
                context_length=context_length,
                top_n=top,
                benchmark_scores=bench_scores,
                task_scores=task_scores.get(profile),
                profile=profile,
            )

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
            sim_hw = apply_default_budgets(
                replace(current_hw, gpus=[synthetic], budget_notes=[])
            )
            sim_results = _rank_for(sim_hw)
            target_results.append((raw_name, sim_hw, sim_results))

    if json_output:
        display_upgrade_json(current_hw, current_results, target_results)
    else:
        console.print()
        display_upgrade(current_hw, current_results, target_results)
        console.print()
