"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.utils import _current_version, CONTEXT_LENGTH

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


def _format_fetch_error(error: Exception) -> str:
    """Return a useful one-line fetch error even when str(error) is empty."""
    detail = str(error).strip()
    if detail:
        return detail

    response = getattr(error, "response", None)
    request = getattr(error, "request", None) or getattr(response, "request", None)
    status_code = getattr(response, "status_code", None)
    url = getattr(request, "url", None)
    if status_code and url:
        return f"{type(error).__name__}: HTTP {status_code} for {url}"
    if url:
        return f"{type(error).__name__} while requesting {url}"
    return f"{type(error).__name__} with no detail from the network layer"


def _print_version(value: bool) -> None:
    """Print version and exit when --version is requested."""
    if value:
        console.print(_current_version())
        raise typer.Exit()


def _validate_gpu_flags(
    cpu_only: bool,
    gpu: str | None,
    vram: float | None,
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


def _validate_evidence(evidence: str) -> str:
    """Validate evidence mode option."""
    valid = {"strict", "base", "any"}
    mode = evidence.lower()
    if mode not in valid:
        console.print("[red]Error:[/] --evidence must be one of: strict, base, any.")
        raise typer.Exit(code=1)
    return mode


def _resolve_evidence_mode(evidence: str, direct: bool) -> str:
    """Resolve final evidence mode, keeping --direct as strict alias."""
    mode = _validate_evidence(evidence)
    if direct:
        # 互換性維持のため --direct は strict と同義に固定する。
        return "strict"
    return mode


def _apply_gpu_overrides(
    hardware: HardwareInfo,
    cpu_only: bool,
    gpu: str | None,
    vram: float | None,
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
    """Pick automatic min-params threshold for strongest general ranking.

    The threshold rises with VRAM so a 24GB GPU is steered away from 3-4B
    toys, but tiny GPUs (4-8GB) still see full-GPU options instead of being
    forced into 7B+ partial-offload-only results.
    """
    if profile != "general":
        return None
    if not hardware.gpus:
        return 2.0  # CPU-only: tiny is the only practical choice
    usable_ram = int(hardware.ram_bytes * 0.80)
    best_vram_gb = max(
        (usable_ram if g.shared_memory and g.vram_bytes == 0 else g.vram_bytes)
        for g in hardware.gpus
    ) / (1024**3)
    if best_vram_gb >= 30:
        return 12.0
    if best_vram_gb >= 20:
        return 10.0
    if best_vram_gb >= 12:
        return 8.0
    if best_vram_gb >= 8:
        return 5.0
    if best_vram_gb >= 5:
        return 3.0
    return 2.0


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


def _merge_model_eval_benchmarks(
    models: list,
    benchmark_scores: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Deprecated no-op kept for backward API compatibility.

    Previously this injected each model's uploader-reported ``hf_eval``
    value into the leaderboard scores dict under the model's id, which
    caused those values to be treated as ``direct`` benchmark evidence
    by the ranker. That elevated any account that wrote a high number
    in their model card to the top of the rankings.

    The hf_eval value is now consumed inside ``rank_models`` via
    ``BenchmarkEvidence.source == "self_reported"`` with a much lower
    weight and a dedicated display tag, so we no longer need to mutate
    the leaderboard dict here. Returning the input unchanged keeps any
    external callers working.
    """
    return benchmark_scores, 0


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
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"
    ),
    min_speed: Optional[float] = typer.Option(
        None, "--min-speed", help="Minimum tok/s filter"
    ),
    evidence: str = typer.Option(
        "any",
        "--evidence",
        help="Benchmark evidence filter: strict | base | any",
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Alias of --evidence strict",
    ),
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
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override VRAM in GB (requires --gpu)"
    ),
):
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram)
    profile = _validate_profile(profile)
    evidence_mode = _resolve_evidence_mode(evidence, direct)

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

        # Step 3: Fetch benchmark scores
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
            evidence_filter=evidence_mode,
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
                evidence_filter=evidence_mode,
            )

        # 上位候補の公開日時が欠けている場合のみ補完して表示品質を上げる
        if results:
            try:
                if _fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    save_cache(models_to_dicts(models))
            except Exception as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

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
def plan(
    model_name: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Target quantization (default: Q4_K_M)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
):
    """Show what GPU you need to run a specific model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichllm.output.display import display_plan, display_plan_json

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models(include_vision=True))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                sys.exit(1)

    model = _search_model(models, model_name)

    target_quant = quant.upper() if quant else "Q4_K_M"

    if json_output:
        display_plan_json(model, context_length, target_quant)
    else:
        console.print()
        display_plan(model, context_length, target_quant)
        console.print()


@app.command()
def upgrade(
    target_gpus: list[str] = typer.Argument(
        ...,
        help="GPUs to compare against (e.g. 'RTX 4090' 'RTX 5090' 'H100')",
    ),
    context_length: int = typer.Option(
        8192,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for ranking (e.g. 8192, 64k, 128k)",
    ),
    top: int = typer.Option(3, "--top", "-n", help="Best-N models to compare per GPU"),
    profile: str = typer.Option("general", "--profile", help="Ranking profile"),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Compare against a CPU-only baseline"
    ),
    json_output: bool = typer.Option(False, "--json"),
    refresh: bool = typer.Option(False, "--refresh"),
):
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


def _load_models(refresh: bool, include_vision: bool = True):
    """Load models from cache or fetch from HuggingFace."""
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts

    cached_data = None if refresh else load_cache()
    if cached_data is not None:
        return dicts_to_models(cached_data)
    try:
        models = _run_async(fetch_models(include_vision=include_vision))
        save_cache(models_to_dicts(models))
        return models
    except Exception as e:
        console.print(f"[red]Error fetching models:[/] {_format_fetch_error(e)}")
        sys.exit(1)


def _search_model(models: list, model_name: str):
    """Search for a model by name/ID. Returns single model or exits."""
    query_lower = model_name.lower()
    terms = query_lower.split()

    matches = [m for m in models if m.id.lower() == query_lower]
    if not matches:
        matches = [m for m in models if m.id.lower().endswith("/" + query_lower)]
    if not matches:
        matches = [m for m in models if all(t in m.id.lower() for t in terms)]

    if not matches:
        console.print(f"[red]No model found matching '{model_name}'.[/]")
        suggestions = [m for m in models if any(t in m.id.lower() for t in terms)]
        if suggestions:
            suggestions.sort(key=lambda m: m.downloads, reverse=True)
            console.print("\n[yellow]Did you mean:[/]")
            for m in suggestions[:5]:
                p = (
                    f"{m.parameter_count / 1e9:.1f}B"
                    if m.parameter_count >= 1e9
                    else f"{m.parameter_count / 1e6:.0f}M"
                )
                console.print(f"  • {m.id} ({p})")
        raise typer.Exit(code=1)

    matches.sort(key=lambda m: m.downloads, reverse=True)
    model = matches[0]
    if len(matches) > 1:
        console.print(f"[dim]Found {len(matches)} matches, using: {model.id}[/]")
    return model


def _pick_gguf_variant(model, quant_filter: str | None = None):
    """Pick the best GGUF variant for a model."""
    from whichllm.constants import QUANT_PREFERENCE_ORDER

    if not model.gguf_variants:
        return None

    if quant_filter:
        for v in model.gguf_variants:
            if v.quant_type.upper() == quant_filter.upper():
                return v
        console.print(
            f"[yellow]Warning:[/] {quant_filter} not available, using best match."
        )

    # Pick by preference order
    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def _find_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def _is_same_model_family(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.id == selected.id:
        return True
    if candidate.family_id and selected.family_id:
        if candidate.family_id == selected.family_id:
            return True
    if candidate.base_model and candidate.base_model == selected.id:
        return True
    if selected.base_model and selected.base_model == candidate.id:
        return True
    if candidate.base_model and selected.base_model:
        return candidate.base_model == selected.base_model
    return False


def _has_compatible_parameter_count(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def _resolve_ranked_gguf_for_run(
    selected_model: ModelInfo,
    selected_variant: GGUFVariant,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    """Resolve a ranked GGUF candidate to a real GGUF repo/file for `run`.

    The ranker may synthesize GGUF variants for official safetensors-only repos
    so they can be scored realistically. `run` cannot execute those synthetic
    files directly, so it must find a real GGUF sibling before launching.
    """
    desired_quant = quant_filter or selected_variant.quant_type

    if selected_model.gguf_variants:
        variant = _find_gguf_variant(selected_model, desired_quant)
        return (selected_model, variant) if variant else None

    candidates: list[tuple[bool, int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not _is_same_model_family(model, selected_model):
            continue
        if not _has_compatible_parameter_count(model, selected_model):
            continue
        variant = _find_gguf_variant(model, desired_quant)
        if not variant:
            continue
        explicit_base = model.base_model == selected_model.id
        candidates.append(
            (
                explicit_base,
                model.downloads,
                model.likes,
                model,
                variant,
            )
        )

    if not candidates:
        return None

    _, _, _, model, variant = max(candidates, key=lambda item: item[:3])
    return model, variant


def _resolve_model_deps(model, variant) -> tuple[list[str], str]:
    """Determine pip dependencies and script type for a model.

    Returns (deps, script_type) where script_type is 'gguf' or 'transformers'.
    """
    if variant:
        return ["llama-cpp-python", "huggingface-hub"], "gguf"

    from whichllm.engine.quantization import infer_non_gguf_quant_type

    qt = infer_non_gguf_quant_type(model.id)
    base = ["transformers", "torch", "accelerate"]
    if qt == "AWQ":
        return [*base, "autoawq"], "transformers"
    if qt == "GPTQ":
        return [*base, "auto-gptq"], "transformers"
    return base, "transformers"


def _generate_chat_script(model, variant, context_length: int, cpu_only: bool) -> str:
    """Generate a self-contained Python chat script for any model type."""
    if variant:
        n_gpu = 0 if cpu_only else -1
        return f'''\
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

print("Downloading {model.id} ({variant.quant_type})...")
model_path = hf_hub_download(repo_id="{model.id}", filename="{variant.filename}")
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print("Ready! Type 'exit' to quit.\\n")
messages = []
while True:
    try:
        user_input = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if user_input.strip().lower() in ("exit", "quit", "q"):
        break
    if not user_input.strip():
        continue
    messages.append({{"role": "user", "content": user_input}})
    response = llm.create_chat_completion(messages=messages, stream=True)
    full = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
            full += content
    print()
    messages.append({{"role": "assistant", "content": full}})
print("\\nBye!")
'''

    device_map = '"cpu"' if cpu_only else '"auto"'
    dtype = "torch.float32" if cpu_only else '"auto"'
    return f'''\
import shutil
import tempfile
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "{model.id}"
offload_folder = tempfile.mkdtemp(prefix="whichllm_transformers_offload_")
try:
    print(f"Loading {{model_id}}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={device_map},
        torch_dtype={dtype},
        trust_remote_code=True,
        offload_folder=offload_folder,
    )
    print("Ready! Type 'exit' to quit.\\n")
    messages = []
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in ("exit", "quit", "q"):
            break
        if not user_input.strip():
            continue
        messages.append({{"role": "user", "content": user_input}})
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=model.generate,
            kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer),
        )
        thread.start()
        full = ""
        for text in streamer:
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        messages.append({{"role": "assistant", "content": full}})
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


@app.command()
def run(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to run (default: auto-pick best)"
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode"),
):
    """Download and chat with a model. Picks the best one if none specified."""
    import os
    import shutil
    import subprocess
    import tempfile

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
                resolved = _resolve_ranked_gguf_for_run(
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


@app.command()
def snippet(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to show snippet for (default: auto-pick best)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
):
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
        code = f'''\
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="{model.id}",
    filename="{variant.filename}",
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = all layers on GPU, 0 = CPU only
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(output["choices"][0]["message"]["content"])
'''
    else:
        code = f'''\
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{model.id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
)

inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichllm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))


@app.command()
def hardware(
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override VRAM in GB (requires --gpu)"
    ),
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
