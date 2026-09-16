"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import typer

from whichllm.cli_models import (
    _generate_chat_script,
    _load_models,
    _pick_gguf_variant,
    _resolve_model_deps,
    _search_model,
)
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
) -> None:
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram, bandwidth, gpu_index)
    _validate_output_flags(json_output, markdown_output)
    _validate_ranking_flags(top, min_speed, min_params)
    profile = _validate_profile(profile)
    evidence_mode = _resolve_evidence_mode(evidence, direct)
    fit_filter = _resolve_fit_filter(fit, gpu_only)
    speed_filter = _resolve_speed_filter(speed, min_speed)

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
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram, bandwidth, gpu_index)
        _apply_memory_budgets(
            hardware, vram_headroom=vram_headroom, ram_budget=ram_budget
        )
        progress.update(task, description="Hardware detected")

        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
            progress.update(task, description=f"Loaded {len(models)} models from cache")
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(
                    fetch_models(include_vision=_include_vision_candidates(profile))
                )
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"Fetched {len(models)} models")
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                sys.exit(1)

        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                progress.update(task, description="Fetching benchmark scores...")
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        progress.update(task, description="Ranking models...")
        families = group_models(models)
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

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
            min_speed=speed_filter,
            benchmark_scores=bench_scores,
            task_profile=profile,
            require_direct_top=True,
            min_params_b=auto_min_params,
            evidence_filter=evidence_mode,
            fit_filter=fit_filter,
        )

        if not results and auto_min_params is not None and min_params is None:
            results = rank_models(
                all_models,
                hardware,
                context_length=context_length,
                top_n=top,
                quant_filter=quant,
                min_speed=speed_filter,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=None,
                evidence_filter=evidence_mode,
                fit_filter=fit_filter,
            )

        if results:
            from whichllm.models.artifacts import attach_resolved_artifacts

            attach_resolved_artifacts(results, all_models, quant_filter=quant)
            try:
                if _fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    save_cache(models_to_dicts(models))
            except Exception as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

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
        console.print()


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

    from whichllm.output.display import display_plan, display_plan_json

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    model = _search_model(models, model_name)
    target_quant = quant.upper() if quant else "Q4_K_M"

    if json_output:
        display_plan_json(model, context_length, target_quant)
    else:
        console.print()
        display_plan(model, context_length, target_quant)
        console.print()


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
    """Compare the current machine against potential GPU upgrades."""
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
    if model_name:
        model = _search_model(models, model_name)
    else:
        from whichllm.engine.ranker import rank_models
        from whichllm.hardware.detector import detect_hardware
        from whichllm.models.artifacts import resolve_ranked_gguf_artifact
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
    deps, script_type = _resolve_model_deps(model, variant)
    script = _generate_chat_script(model, variant, context_length, cpu_only)

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


def snippet_command(
    *,
    model_name: str | None,
    quant: str | None,
    refresh: bool,
) -> None:
    """Print a ready-to-run Python script for a model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    if model_name:
        model = _search_model(models, model_name)
    else:
        gguf_models = [m for m in models if m.gguf_variants]
        if not gguf_models:
            console.print("[red]No GGUF models found.[/]")
            raise typer.Exit(code=1)
        gguf_models.sort(key=lambda m: m.downloads, reverse=True)
        model = gguf_models[0]

    variant = _pick_gguf_variant(model, quant)
    deps, _ = _resolve_model_deps(model, variant)

    if variant:
        code = f"""\
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id={model.id!r},
    filename={variant.filename!r},
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = all layers on GPU, 0 = CPU only
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(output["choices"][0]["message"]["content"])
"""
    else:
        code = f"""\
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = {model.id!r}
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
)

inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichllm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))


def hardware_command(
    *,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None,
    gpu_index: int | None,
) -> None:
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram, bandwidth, gpu_index)

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
        _apply_gpu_overrides(hw, cpu_only, gpu, vram, bandwidth, gpu_index)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


__all__ = [
    "hardware_command",
    "main_command",
    "plan_command",
    "run_command",
    "snippet_command",
    "upgrade_command",
]
