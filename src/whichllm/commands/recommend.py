"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

import sys

import typer

from whichllm.cli_shared import _format_fetch_error, _run_async, console
from whichllm.cli_validation import (
    _apply_gpu_overrides,
    _apply_memory_budgets,
    _auto_min_params_for_profile,
    _fill_missing_published_at,
    _include_vision_candidates,
    _resolve_evidence_mode,
    _resolve_fit_filter,
    _resolve_speed_filter,
    _validate_gpu_flags,
    _validate_output_flags,
    _validate_profile,
    _validate_ranking_flags,
)


from pathlib import Path
from whichllm.cli_validation import _validate_lmstudio_path_flags
from whichllm.models.artifacts import (
    attach_resolved_artifacts,
)

# Extra ranked rows whose families get their config.json read, so a model
# just below the cut can move up once its exact KV size is known.
_SHAPE_LOOKAHEAD = 10


def _load_live_models(progress, task, refresh: bool, profile: str) -> list:
    """Models from the model cache or a direct Hugging Face fetch."""
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import (
        ModelFetchReport,
        dicts_to_models,
        fetch_models,
        models_to_dicts,
    )

    include_vision = _include_vision_candidates(profile)
    cached_data = None if refresh else load_cache(include_vision=include_vision)
    if cached_data is not None:
        models = dicts_to_models(cached_data)
        progress.update(task, description=f"Loaded {len(models)} models from cache")
        return models
    progress.update(task, description="Fetching models from HuggingFace...")
    try:
        fetch_report = ModelFetchReport()
        models = _run_async(
            fetch_models(include_vision=include_vision, report=fetch_report)
        )
        save_cache(
            models_to_dicts(models),
            include_vision=include_vision,
            complete=fetch_report.complete,
        )
        progress.update(task, description=f"Fetched {len(models)} models")
        return models
    except Exception as e:
        console.print(f"[red]Error fetching models:[/] {_format_fetch_error(e)}")
        sys.exit(1)


def _load_live_benchmarks(progress, task, refresh: bool) -> dict[str, float]:
    """Benchmark scores from the benchmark cache or the live sources."""
    from whichllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )

    progress.update(task, description="Loading benchmark data...")
    bench_scores = None if refresh else load_benchmark_cache()
    if bench_scores is not None:
        return bench_scores
    try:
        progress.update(task, description="Fetching benchmark scores...")
        bench_scores = _run_async(fetch_benchmark_scores())
        save_benchmark_cache(bench_scores)
        return bench_scores
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
        return {}


def _print_data_source(catalog) -> None:
    """Say which data the ranking used, and warn when it is old."""
    if catalog is None:
        console.print(
            "  [dim]Data: fetched live from Hugging Face and benchmark sites.[/dim]"
        )
        return
    built = catalog.built_at.replace("T", " ").replace("Z", " UTC")
    console.print(f"  [dim]Data: whichllm index built {built}.[/dim]")
    age = catalog.age_seconds
    if catalog.offline and age is not None and age > 3 * 24 * 3600:
        console.print(
            f"  [yellow]Note:[/] offline, using an index {age / 86400:.0f} days "
            "old. Recent releases may be missing."
        )


def main_command(
    ctx: typer.Context,
    *,
    refresh: bool,
    top: int,
    context_length: int,
    quant: str | None,
    min_speed: float | None,
    speed: str,
    fit: str,
    gpu_only: bool,
    evidence: str,
    direct: bool,
    status: bool,
    details: bool,
    min_params: float | None,
    profile: str,
    json_output: bool,
    markdown_output: bool,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None,
    gpu_index: int | None,
    vram_headroom: str,
    ram_budget: str | None,
    lm_studio_path: list[Path] | None,
    live: bool = False,
) -> None:
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram, bandwidth, gpu_index)
    _validate_output_flags(json_output, markdown_output)
    _validate_lmstudio_path_flags(lm_studio_path)
    _validate_ranking_flags(top, min_speed, min_params)
    profile = _validate_profile(profile)
    evidence_mode = _resolve_evidence_mode(evidence, direct)
    fit_filter = _resolve_fit_filter(fit, gpu_only)
    speed_filter = _resolve_speed_filter(speed, min_speed)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.cache import update_cached_models
    from whichllm.models.fetcher import fetch_model_published_at, models_to_dicts
    from whichllm.models.grouper import group_models
    from whichllm.models.index_client import load_index
    from whichllm.models.shape_fetch import attach_cached_shapes, fetch_family_shapes
    from whichllm.output.display import (
        display_hardware,
        display_json,
        display_markdown,
        display_ranking,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Detect hardware
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram, bandwidth, gpu_index)
        _apply_memory_budgets(
            hardware, vram_headroom=vram_headroom, ram_budget=ram_budget
        )
        progress.update(task, description="Hardware detected")

        # Step 2: Load models and benchmark scores, from the prebuilt index
        # unless --live asks for a direct fetch or the index is unreachable.
        progress.update(task, description="Loading the whichllm index...")
        catalog = None if live else load_index(refresh=refresh)
        if catalog is not None:
            models = catalog.models
            bench_scores = catalog.benchmark_scores
            progress.update(task, description=f"Loaded {len(models)} models")
        else:
            models = _load_live_models(progress, task, refresh, profile)
            bench_scores = _load_live_benchmarks(progress, task, refresh)

        # Step 4: Group and rank
        progress.update(task, description="Ranking models...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        # NOTE: We no longer merge uploader-reported hf_eval values into the
        # leaderboard scores dict — the ranker now treats them as a separate
        # "self_reported" evidence tier with much lower trust. See
        # ranker.lookup_benchmark_evidence + _SOURCE_WEIGHTS.

        # general用途はGPUクラスに応じた自動しきい値で小さすぎるモデルを抑制する
        auto_min_params = (
            _auto_min_params_for_profile(hardware, profile)
            if min_params is None
            else min_params
        )

        def _rank(top_n: int) -> list:
            rank_kwargs = dict(
                context_length=context_length,
                top_n=top_n,
                quant_filter=quant,
                min_speed=speed_filter,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                evidence_filter=evidence_mode,
                fit_filter=fit_filter,
            )
            ranked = rank_models(
                all_models, hardware, min_params_b=auto_min_params, **rank_kwargs
            )
            # 自動しきい値で候補ゼロなら緩和して表示を維持する
            if not ranked and auto_min_params is not None and min_params is None:
                ranked = rank_models(
                    all_models, hardware, min_params_b=None, **rank_kwargs
                )
            return ranked

        # Exact KV sizes need each model's config.json. The index carries the
        # shapes it could read; a live fetch looks them up only for the
        # families near the top, then ranks again with them.
        attach_cached_shapes(all_models)
        if catalog is None:
            results = _rank(top + _SHAPE_LOOKAHEAD)
            shapeless = [r.model.family_id for r in results if r.model.shape is None]
            if shapeless:
                progress.update(task, description="Reading model configs...")
                try:
                    _run_async(fetch_family_shapes(all_models, shapeless))
                except Exception as e:
                    progress.update(task, description=f"Model configs skipped: {e}")
        results = _rank(top)

        # 上位候補の公開日時が欠けている場合のみ補完して表示品質を上げる
        if results:
            attach_resolved_artifacts(results, all_models, quant_filter=quant)
            from whichllm.models.lmstudio import (
                attach_local_matches,
                discover_lmstudio_ggufs,
                LMStudioPathError,
            )

            try:
                local_models = discover_lmstudio_ggufs(lm_studio_path or ())
            except LMStudioPathError as error:
                console.print(f"[red]Error:[/] {error}")
                raise typer.Exit(code=1) from error
            attach_local_matches(results, local_models)
            try:
                if catalog is None and _fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    update_cached_models(models_to_dicts(models))
            except Exception as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

    # Display results
    empty_message = None
    if fit_filter == "full_gpu":
        empty_message = (
            "No full-GPU models found for this hardware. "
            "Remove --gpu-only or use --fit any to include partial offload "
            "and CPU-only candidates."
        )
    if json_output:
        display_json(results, hardware)
    elif markdown_output:
        display_markdown(
            results,
            hardware,
            show_status=status or not details,
            empty_message=empty_message,
        )
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(
            results,
            has_gpu=bool(hardware.gpus),
            show_status=status or not details,
            empty_message=empty_message,
        )
        _print_data_source(catalog)
        console.print()
