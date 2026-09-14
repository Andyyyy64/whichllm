"""CLI validation, filtering, and hardware override helpers."""

from __future__ import annotations

import re

import typer

from whichllm.cli_shared import _run_async, console
from whichllm.constants import _GiB
from whichllm.hardware.types import HardwareInfo


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
    """Validate ranking/filter flags that otherwise silently distort output."""
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


_MEMORY_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gib|gb|g|mib|mb|m)?$",
    re.IGNORECASE,
)


def _parse_memory_amount(
    value: str, *, option_name: str, total_bytes: int | None = None
) -> int:
    """Parse memory CLI values. Bare numbers are treated as GiB."""
    raw = value.strip()
    if not raw:
        console.print(f"[red]Error:[/] {option_name} cannot be empty.")
        raise typer.Exit(code=1)

    if raw.endswith("%"):
        if total_bytes is None:
            console.print(f"[red]Error:[/] {option_name} percentage needs a base size.")
            raise typer.Exit(code=1)
        try:
            pct = float(raw[:-1])
        except ValueError:
            console.print(f"[red]Error:[/] Invalid {option_name}: {value!r}.")
            raise typer.Exit(code=1)
        if pct < 0:
            console.print(f"[red]Error:[/] {option_name} must be non-negative.")
            raise typer.Exit(code=1)
        return int(total_bytes * pct / 100.0)

    match = _MEMORY_RE.match(raw)
    if not match:
        console.print(
            f"[red]Error:[/] Invalid {option_name}: {value!r}. "
            "Use values like 1.5GB, 512MB, 10%, or 8."
        )
        raise typer.Exit(code=1)

    number = float(match.group("number"))
    unit = (match.group("unit") or "gb").lower()
    if number < 0:
        console.print(f"[red]Error:[/] {option_name} must be non-negative.")
        raise typer.Exit(code=1)

    if unit in {"gib", "gb", "g"}:
        return int(number * _GiB)
    return int(number * 1024**2)


def _auto_vram_headroom(vram_bytes: int) -> int:
    """Default runtime headroom so near-edge fits do not over-promise."""
    if vram_bytes <= 0:
        return 0
    return int(max(512 * 1024**2, min(vram_bytes * 0.05, 2 * _GiB)))


def _parse_vram_headroom(value: str, vram_bytes: int) -> int:
    mode = value.strip().lower()
    if mode == "auto":
        return _auto_vram_headroom(vram_bytes)
    if mode in {"none", "off", "0"}:
        return 0
    return _parse_memory_amount(
        value,
        option_name="--vram-headroom",
        total_bytes=vram_bytes,
    )


def _apply_memory_budgets(
    hardware: HardwareInfo,
    *,
    vram_headroom: str,
    ram_budget: str | None,
) -> HardwareInfo:
    """Apply user-facing memory budgets without mutating detected raw sizes."""
    headroom_mode = vram_headroom.strip().lower()
    if not hardware.gpus and headroom_mode not in {"auto", "none", "off", "0"}:
        _parse_memory_amount(
            vram_headroom,
            option_name="--vram-headroom",
            total_bytes=_GiB,
        )

    reserved_values: list[int] = []
    for gpu in hardware.gpus:
        reserved = _parse_vram_headroom(vram_headroom, gpu.vram_bytes)
        gpu.usable_vram_bytes = max(0, gpu.vram_bytes - reserved)
        if reserved > 0:
            reserved_values.append(reserved)

    if reserved_values:
        unique_reserved = sorted(set(reserved_values))
        if len(unique_reserved) == 1:
            note = f"VRAM headroom: {_format_budget_bytes(unique_reserved[0])} reserved per GPU"
        else:
            note = "VRAM headroom: auto reserve applied per GPU"
        hardware.budget_notes.append(note)

    if ram_budget:
        mode = ram_budget.strip().lower()
        if mode == "available":
            from whichllm.hardware.memory import detect_available_ram_bytes

            hardware.ram_budget_bytes = detect_available_ram_bytes()
            hardware.budget_notes.append(
                f"RAM budget: current available {_format_budget_bytes(hardware.ram_budget_bytes)}"
            )
        elif mode not in {"auto", "none", "off"}:
            hardware.ram_budget_bytes = _parse_memory_amount(
                ram_budget, option_name="--ram-budget", total_bytes=hardware.ram_bytes
            )
            hardware.budget_notes.append(
                f"RAM budget: {_format_budget_bytes(hardware.ram_budget_bytes)}"
            )
    return hardware


def _format_budget_bytes(value: int) -> str:
    if value >= _GiB:
        return f"{value / _GiB:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value / 1024:.0f} KB"


def _apply_gpu_overrides(
    hardware: HardwareInfo,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
    bandwidth: float | None = None,
    gpu_index: int | None = None,
) -> HardwareInfo:
    """Replace hardware.gpus based on CLI flags."""
    if cpu_only:
        hardware.gpus = []
    elif gpu:
        from whichllm.hardware.gpu_simulator import create_synthetic_gpus

        try:
            hardware.gpus = create_synthetic_gpus(gpu, vram)
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
        if bandwidth is not None:
            if len(hardware.gpus) != 1:
                console.print(
                    "[red]Error:[/] --bandwidth currently supports exactly one "
                    "simulated GPU."
                )
                raise typer.Exit(code=1)
            hardware.gpus[0].memory_bandwidth_gbps = bandwidth
    elif vram is not None or bandwidth is not None:
        if not hardware.gpus:
            console.print(
                "[red]Error:[/] --vram/--bandwidth requires a detected GPU or --gpu."
            )
            raise typer.Exit(code=1)
        if gpu_index is None:
            if len(hardware.gpus) > 1:
                console.print(
                    "[red]Error:[/] --gpu-index is required when overriding "
                    "detected hardware with multiple GPUs."
                )
                raise typer.Exit(code=1)
            target_gpu = hardware.gpus[0]
        else:
            if gpu_index >= len(hardware.gpus):
                console.print(
                    f"[red]Error:[/] --gpu-index {gpu_index} is out of range "
                    f"for {len(hardware.gpus)} detected GPU(s)."
                )
                raise typer.Exit(code=1)
            target_gpu = hardware.gpus[gpu_index]

        if vram is not None:
            target_gpu.vram_bytes = int(vram * _GiB)
            target_gpu.usable_vram_bytes = None
            target_gpu.vram_overridden = True
        if bandwidth is not None:
            target_gpu.memory_bandwidth_gbps = bandwidth
    return hardware


def _auto_min_params_for_profile(hardware: HardwareInfo, profile: str) -> float | None:
    """Pick automatic min-params threshold for strongest general ranking."""
    if profile != "general":
        return None
    if not hardware.gpus:
        return 2.0
    from whichllm.hardware.memory import effective_usable_ram

    usable_ram = effective_usable_ram(hardware.ram_bytes, hardware.ram_budget_bytes)
    best_vram_gb = max(
        (
            usable_ram
            if g.shared_memory
            and (g.vram_bytes == 0 or hardware.ram_budget_bytes is not None)
            else (
                g.usable_vram_bytes if g.usable_vram_bytes is not None else g.vram_bytes
            )
        )
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
    """Return whether the fetcher should include vision-language candidates."""
    return profile.lower() in {"vision", "any"}


def _fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    """Backfill missing publish dates for top results."""
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
    """Deprecated no-op kept for backward API compatibility."""
    return benchmark_scores, 0


__all__ = [
    "_apply_gpu_overrides",
    "_apply_memory_budgets",
    "_auto_min_params_for_profile",
    "_auto_vram_headroom",
    "_fill_missing_published_at",
    "_format_budget_bytes",
    "_include_vision_candidates",
    "_merge_model_eval_benchmarks",
    "_parse_memory_amount",
    "_parse_vram_headroom",
    "_resolve_evidence_mode",
    "_resolve_fit_filter",
    "_resolve_speed_filter",
    "_validate_evidence",
    "_validate_gpu_flags",
    "_validate_output_flags",
    "_validate_profile",
    "_validate_ranking_flags",
]
