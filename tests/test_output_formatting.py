"""Tests for output formatting helpers."""

from whichllm.engine.types import CompatibilityResult
from whichllm.models.types import ModelInfo
from whichllm.output.formatting import _format_speed


def _result(speed: float, confidence: str) -> CompatibilityResult:
    return CompatibilityResult(
        model=ModelInfo(
            id="org/model",
            family_id="org/model",
            name="model",
            parameter_count=1_000_000_000,
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        estimated_tok_per_sec=speed,
        speed_confidence=confidence,
    )


def test_format_speed_colors_by_runtime_speed_not_confidence():
    assert _format_speed(_result(2.5, "medium")) == "[red]2.5 tok/s ~[/red]"
    assert _format_speed(_result(6.0, "low")) == "[yellow]6.0 tok/s ?[/yellow]"
    assert _format_speed(_result(12.0, "medium")) == "[green]12.0 tok/s ~[/green]"
    assert (
        _format_speed(_result(30.0, "low"))
        == "[bright_green]30.0 tok/s ?[/bright_green]"
    )


def test_display_ranking_includes_disk_column_before_fit_vram():
    from io import StringIO
    from rich.console import Console
    import whichllm.output._console as console_mod
    from whichllm.models.types import GGUFVariant
    from whichllm.output.ranking import display_ranking

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=120)
    try:
        res = CompatibilityResult(
            model=ModelInfo(
                id="google/gemma-4-26B-A4B-it",
                family_id="google/gemma-4-26B-A4B-it",
                name="gemma-4-26B-A4B-it",
                parameter_count=25_800_000_000,
            ),
            gguf_variant=GGUFVariant(
                filename="gemma-4-26B-A4B-it-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=15 * 1024**3,
            ),
            can_run=True,
            vram_required_bytes=15 * 1024**3,
            vram_available_bytes=16 * 1024**3,
            estimated_tok_per_sec=25.3,
            speed_confidence="low",
            quality_score=80.8,
            fit_type="full_gpu",
            benchmark_status="direct",
        )
        display_ranking([res], has_gpu=True, show_status=True)
        output = buf.getvalue()
        assert "Disk" in output
        assert "Fit" in output
        assert "VRAM" in output
        assert "15.0 GB" in output
    finally:
        console_mod.console = orig_console
