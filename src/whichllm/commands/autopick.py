"""Pick the best-ranked model that has a runnable artifact."""

from __future__ import annotations

import typer

from whichllm.cli_shared import console
from whichllm.models.artifacts import resolve_ranked_gguf_artifact
from whichllm.models.types import GGUFVariant, ModelInfo


def pick_top_runnable(
    models: list[ModelInfo],
    *,
    context_length: int,
    quant: str | None,
    cpu_only: bool = False,
    top_n: int = 5,
) -> tuple[ModelInfo, GGUFVariant | None]:
    """Rank models for this machine and return the first runnable pick.

    GGUF-ranked candidates are resolved to a real GGUF repository and file.
    A candidate without a matching GGUF build is skipped with a warning.
    The returned variant is None for a non-GGUF pick.
    """
    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.benchmark import load_benchmark_cache
    from whichllm.models.grouper import group_models

    hardware = detect_hardware()
    if cpu_only:
        hardware.gpus = []
    bench_scores = load_benchmark_cache() or {}
    all_models = []
    for family in group_models(models):
        all_models.append(family.base_model)
        all_models.extend(family.variants)

    results = rank_models(
        all_models,
        hardware,
        context_length=context_length,
        top_n=top_n,
        quant_filter=quant,
        benchmark_scores=bench_scores,
    )
    if not results:
        console.print("[red]No runnable model found for your hardware.[/]")
        raise typer.Exit(code=1)

    skipped_gguf: list[str] = []
    picked: tuple[ModelInfo, GGUFVariant | None] | None = None
    for ranked in results:
        if not ranked.gguf_variant:
            picked = (ranked.model, None)
            break
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
            picked = (resolved_model, variant)
            break
        skipped_gguf.append(ranked.model.id)

    if skipped_gguf:
        skipped = ", ".join(skipped_gguf[:3])
        suffix = "..." if len(skipped_gguf) > 3 else ""
        console.print(
            "[yellow]Warning:[/] Skipped GGUF-ranked candidate(s) without "
            f"a matching runnable GGUF repo: {skipped}{suffix}"
        )
    if picked is None:
        console.print(
            "[red]Error:[/] Top recommendations require GGUF builds, "
            "but no matching GGUF repos were found."
        )
        raise typer.Exit(code=1)
    return picked
