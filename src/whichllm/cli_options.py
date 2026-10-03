"""CLI validation, filtering, and hardware override helpers."""

from __future__ import annotations


import typer

from whichllm.cli_shared import console


from pathlib import Path


def _validate_gpu_flags(
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None = None,
    gpu_index: int | None = None,
) -> None:
    """Validate mutual exclusivity of GPU-related flags."""
    if cpu_only and gpu:
        console.print("[red]Error:[/] --cpu-only and --gpu are mutually exclusive.")
        raise typer.Exit(code=1)
    if cpu_only and (vram is not None or bandwidth is not None):
        console.print("[red]Error:[/] --cpu-only cannot be used with GPU overrides.")
        raise typer.Exit(code=1)
    if vram is not None and vram <= 0:
        console.print("[red]Error:[/] --vram must be greater than 0.")
        raise typer.Exit(code=1)
    if bandwidth is not None and bandwidth <= 0:
        console.print("[red]Error:[/] --bandwidth must be greater than 0.")
        raise typer.Exit(code=1)
    if gpu_index is not None and gpu_index < 0:
        console.print("[red]Error:[/] --gpu-index must be 0 or greater.")
        raise typer.Exit(code=1)
    if gpu_index is not None and gpu:
        console.print("[red]Error:[/] --gpu-index only applies to detected GPUs.")
        raise typer.Exit(code=1)
    if gpu_index is not None and vram is None and bandwidth is None:
        console.print("[red]Error:[/] --gpu-index requires --vram or --bandwidth.")
        raise typer.Exit(code=1)


def _validate_output_flags(json_output: bool, markdown_output: bool) -> None:
    """Validate mutually exclusive output formats."""
    if json_output and markdown_output:
        console.print("[red]Error:[/] --json and --markdown are mutually exclusive.")
        raise typer.Exit(code=1)


def _validate_ranking_flags(
    top: int,
    min_speed: float | None,
    min_params: float | None,
) -> None:
    """Validate ranking/filter flags that otherwise silently distort output.

    Without these guards a non-positive ``--top`` reaches ``results[:top_n]`` in
    :func:`whichllm.engine.ranker.rank_models`: ``--top 0`` returns no
    recommendations at all, and a negative value slices from the end
    (``results[:-5]``), silently returning a truncated, unrequested subset
    instead of the count the user asked for. Negative ``--min-speed`` /
    ``--min-params`` thresholds are likewise meaningless. Fail fast with a clear
    message instead of producing misleading results.
    """
    if top < 1:
        console.print("[red]Error:[/] --top must be 1 or greater.")
        raise typer.Exit(code=1)
    if min_speed is not None and min_speed < 0:
        console.print("[red]Error:[/] --min-speed must be 0 or greater.")
        raise typer.Exit(code=1)
    if min_params is not None and min_params < 0:
        console.print("[red]Error:[/] --min-params must be 0 or greater.")
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


def _resolve_fit_filter(fit: str, gpu_only: bool) -> str:
    """Resolve runtime fit filtering, keeping --gpu-only as a short alias."""
    mode = fit.lower().replace("_", "-").replace(" ", "-")
    if mode not in {"any", "gpu", "full-gpu", "fullgpu"}:
        console.print("[red]Error:[/] --fit must be one of: any, gpu, full-gpu.")
        raise typer.Exit(code=1)
    if gpu_only:
        return "full_gpu"
    return "full_gpu" if mode in {"gpu", "full-gpu", "fullgpu"} else "any"


def _resolve_speed_filter(speed: str, min_speed: float | None) -> float | None:
    """Resolve named speed presets while preserving --min-speed as exact input."""
    if min_speed is not None:
        return min_speed
    mode = speed.lower().replace("_", "-")
    presets = {
        "any": None,
        "usable": 10.0,
        "fast": 30.0,
    }
    if mode not in presets:
        console.print("[red]Error:[/] --speed must be one of: any, usable, fast.")
        raise typer.Exit(code=1)
    return presets[mode]


def _validate_lmstudio_path_flags(paths: list[Path] | None) -> None:
    """Validate explicit LM Studio libraries before doing network work."""
    if not paths:
        return

    from whichllm.models.lmstudio import LMStudioPathError, validate_lmstudio_paths

    try:
        validate_lmstudio_paths(paths)
    except LMStudioPathError as error:
        console.print(f"[red]Error:[/] {error}")
        raise typer.Exit(code=1) from error
