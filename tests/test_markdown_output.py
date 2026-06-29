"""Tests for pasteable Markdown ranking output."""

from io import StringIO

from rich.console import Console

from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output.markdown import display_markdown


def _capture_markdown(
    results: list[CompatibilityResult],
    hardware: HardwareInfo,
    *,
    show_status: bool,
    empty_message: str | None = None,
) -> str:
    import whichllm.output._console as console_mod

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_markdown(
            results,
            hardware,
            show_status=show_status,
            empty_message=empty_message,
        )
    finally:
        console_mod.console = orig_console
    return buf.getvalue().strip()


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=1008.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


def _result(
    index: int,
    *,
    benchmark_status: str = "direct",
    speed_confidence: str = "medium",
) -> CompatibilityResult:
    model = ModelInfo(
        id=f"org/Test-{index}|Model",
        family_id=f"test-{index}",
        name=f"Test-{index}",
        parameter_count=7_000_000_000 + index,
        downloads=1_500 * index,
        likes=index,
        license="apache-2.0",
        published_at=f"2026-01-0{index}T00:00:00Z",
    )
    return CompatibilityResult(
        model=model,
        gguf_variant=GGUFVariant(
            filename=f"test-{index}.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=4 * 1024**3,
        ),
        can_run=True,
        vram_required_bytes=(4 + index) * 1024**3,
        vram_available_bytes=24 * 1024**3,
        estimated_tok_per_sec=10.0 * index,
        speed_confidence=speed_confidence,
        quality_score=80.0 - index,
        fit_type="full_gpu",
        benchmark_status=benchmark_status,
        benchmark_source=benchmark_status,
        benchmark_confidence=1.0,
    )


def test_display_markdown_runtime_table_top_three():
    output = _capture_markdown(
        [
            _result(1, speed_confidence="medium"),
            _result(2, benchmark_status="estimated", speed_confidence="low"),
            _result(3, benchmark_status="none", speed_confidence="high"),
        ],
        _hardware(),
        show_status=True,
    )

    assert output.startswith("## Recommended Models")
    assert (
        "| # | Model | Params | Quant | Fit | VRAM | Speed | Published | Score | License |"
        in output
    )
    assert (
        "| 1 | org/Test-1\\|Model | 7.0B | Q4_K_M | Full GPU | 5.0 GB | 10.0 tok/s ~ | 2026-01-01 | 79.0 | apache-2.0 |"
        in output
    )
    assert "20.0 tok/s ?" in output
    assert "78.0 ~" in output
    assert "77.0 ?" in output


def test_display_markdown_details_table_uses_metadata_columns():
    output = _capture_markdown([_result(1)], _hardware(), show_status=False)

    assert (
        "| # | Model | Params | Quant | Published | Downloads | Score | License |"
        in output
    )
    assert "Fit | VRAM | Speed" not in output
    assert (
        "| 1 | org/Test-1\\|Model | 7.0B | Q4_K_M | 2026-01-01 | 1.5K | 79.0 | apache-2.0 |"
        in output
    )


def test_display_markdown_links_to_resolved_artifact_repo():
    result = _result(1)
    result.model.id = "Qwen/Qwen3-4B-Thinking-2507"
    result.gguf_variant = GGUFVariant(
        filename="Qwen3-4B-Thinking-2507.Q3_K_M.gguf",
        quant_type="Q3_K_M",
        file_size_bytes=2 * 1024**3,
    )
    result.artifact_model = ModelInfo(
        id="MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF",
        family_id=result.model.family_id,
        name="Qwen3-4B-Thinking-2507-GGUF",
        parameter_count=result.model.parameter_count,
    )
    result.artifact_variant = GGUFVariant(
        filename="Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
        quant_type="Q3_K_M",
        file_size_bytes=2 * 1024**3,
    )

    output = _capture_markdown([result], _hardware(), show_status=False)

    assert (
        "[Qwen/Qwen3-4B-Thinking-2507]"
        "(https://huggingface.co/MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF)" in output
    )
    assert "Q3_K_M" in output


def test_display_markdown_empty_results():
    output = _capture_markdown(
        [],
        _hardware(),
        show_status=True,
        empty_message="Nothing matched.",
    )

    assert output == "## Recommended Models\n\nNothing matched."
