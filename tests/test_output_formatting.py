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


def test_display_ranking_includes_weights_column():
    from io import StringIO
    from rich.console import Console
    import whichllm.output._console as console_mod
    from whichllm.models.types import GGUFVariant
    from whichllm.output.ranking import display_ranking

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=120)
    try:
        res_actual = CompatibilityResult(
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
        res_estimated = CompatibilityResult(
            model=ModelInfo(
                id="casperhansen/deepseek-r1-distill-qwen-7b-awq",
                family_id="qwen",
                name="deepseek-r1-distill-qwen-7b-awq",
                parameter_count=7_000_000_000,
            ),
            gguf_variant=None,
            can_run=True,
            vram_required_bytes=5 * 1024**3,
            vram_available_bytes=16 * 1024**3,
            estimated_tok_per_sec=45.0,
            speed_confidence="low",
            quality_score=75.0,
            fit_type="full_gpu",
            benchmark_status="direct",
        )
        display_ranking([res_actual, res_estimated], has_gpu=True, show_status=True)
        output = buf.getvalue()
        assert "Weights" in output
        assert "Fit" in output
        assert "VRAM" in output
        assert "15.0 GB" in output
        assert "~3.3 GB" in output
        assert "Weights:  ~ = estimated size" in output
    finally:
        console_mod.console = orig_console


# Verify weight formatting and estimation tracking for exact, synthetic, missing-metadata, and non-GGUF models.
def test_format_weights():
    from whichllm.models.types import GGUFVariant
    from whichllm.output.formatting import _format_weights

    res_exact = CompatibilityResult(
        model=ModelInfo(
            id="google/gemma-4-26B-it",
            family_id="google/gemma-4-26B-it",
            name="gemma-4-26B-it",
            parameter_count=26_000_000_000,
        ),
        gguf_variant=GGUFVariant(
            filename="gemma-4-26B-it-Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=15 * 1024**3,
            is_estimated=False,
        ),
        can_run=True,
        vram_required_bytes=15 * 1024**3,
        vram_available_bytes=16 * 1024**3,
    )
    text, is_est = _format_weights(res_exact)
    assert text == "15.0 GB"
    assert is_est is False

    res_synthetic = CompatibilityResult(
        model=ModelInfo(
            id="Qwen/Qwen3.6-27B",
            family_id="qwen",
            name="Qwen3.6-27B",
            parameter_count=27_000_000_000,
        ),
        gguf_variant=GGUFVariant(
            filename="Qwen3.6-27B.Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(27_000_000_000 * 0.5625),
            is_estimated=True,
        ),
        can_run=True,
        vram_required_bytes=16 * 1024**3,
        vram_available_bytes=24 * 1024**3,
    )
    text, is_est = _format_weights(res_synthetic)
    assert text.startswith("~")
    assert is_est is True

    res_missing_meta = CompatibilityResult(
        model=ModelInfo(
            id="some/model",
            family_id="model",
            name="model",
            parameter_count=10_000_000_000,
        ),
        gguf_variant=GGUFVariant(
            filename="model-Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(10_000_000_000 * 0.5625),
            is_estimated=True,
        ),
        can_run=True,
        vram_required_bytes=6 * 1024**3,
        vram_available_bytes=16 * 1024**3,
    )
    text, is_est = _format_weights(res_missing_meta)
    assert text.startswith("~")
    assert is_est is True

    res_non_gguf = CompatibilityResult(
        model=ModelInfo(
            id="casperhansen/deepseek-r1-distill-qwen-7b-awq",
            family_id="qwen",
            name="deepseek-r1-distill-qwen-7b-awq",
            parameter_count=7_000_000_000,
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=5 * 1024**3,
        vram_available_bytes=16 * 1024**3,
    )
    text, is_est = _format_weights(res_non_gguf)
    assert text == "~3.3 GB"
    assert is_est is True


# Verify that ranking table displays estimated markers and footnote for synthetic variants and missing metadata.
def test_display_ranking_weights_synthetic_and_missing_metadata():
    from io import StringIO
    from rich.console import Console
    import whichllm.output._console as console_mod
    from whichllm.models.types import GGUFVariant
    from whichllm.output.ranking import display_ranking

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=120)
    try:
        res_synthetic = CompatibilityResult(
            model=ModelInfo(
                id="Qwen/Qwen3.6-27B",
                family_id="qwen",
                name="Qwen3.6-27B",
                parameter_count=27_000_000_000,
            ),
            gguf_variant=GGUFVariant(
                filename="Qwen3.6-27B.Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=15 * 1024**3,
                is_estimated=True,
            ),
            can_run=True,
            vram_required_bytes=15 * 1024**3,
            vram_available_bytes=16 * 1024**3,
            estimated_tok_per_sec=25.0,
            speed_confidence="medium",
            quality_score=85.0,
            fit_type="full_gpu",
            benchmark_status="direct",
        )
        res_missing_meta = CompatibilityResult(
            model=ModelInfo(
                id="meta-llama/Llama-3.1-8B-Instruct",
                family_id="llama",
                name="Llama-3.1-8B-Instruct",
                parameter_count=8_000_000_000,
            ),
            gguf_variant=GGUFVariant(
                filename="Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=5 * 1024**3,
                is_estimated=True,
            ),
            can_run=True,
            vram_required_bytes=6 * 1024**3,
            vram_available_bytes=16 * 1024**3,
            estimated_tok_per_sec=40.0,
            speed_confidence="medium",
            quality_score=80.0,
            fit_type="full_gpu",
            benchmark_status="direct",
        )
        display_ranking(
            [res_synthetic, res_missing_meta], has_gpu=True, show_status=True
        )
        output = buf.getvalue()
        assert "~15.0 GB" in output
        assert "~5.0 GB" in output
        assert "Weights:  ~ = estimated size" in output
    finally:
        console_mod.console = orig_console


# Verify that ranking table omits estimated markers and footnote when only exact file metadata is present.
def test_display_ranking_weights_exact_metadata_no_footnote():
    from io import StringIO
    from rich.console import Console
    import whichllm.output._console as console_mod
    from whichllm.models.types import GGUFVariant
    from whichllm.output.ranking import display_ranking

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=120)
    try:
        res_actual = CompatibilityResult(
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
                is_estimated=False,
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
        display_ranking([res_actual], has_gpu=True, show_status=True)
        output = buf.getvalue()
        assert "15.0 GB" in output
        assert "~15.0 GB" not in output
        assert "Weights:  ~ = estimated size" not in output
    finally:
        console_mod.console = orig_console


# Verify that GGUF synthesis and sibling extraction correctly set the is_estimated flag.
def test_gguf_variants_preserve_estimated_flag():
    from whichllm.engine.ranking_variants import _synthesize_variants_for_official_repo
    from whichllm.models.gguf import _extract_gguf_variants

    official_model = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    synth = _synthesize_variants_for_official_repo(official_model, None)
    assert len(synth) > 0
    assert all(v.is_estimated for v in synth)

    data_missing = {"siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 0}]}
    extracted_missing = _extract_gguf_variants(data_missing, 10_000_000_000)
    assert len(extracted_missing) == 1
    assert extracted_missing[0].is_estimated is True
    assert extracted_missing[0].file_size_bytes > 0

    data_real = {
        "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 5_000_000_000}]
    }
    extracted_real = _extract_gguf_variants(data_real, 10_000_000_000)
    assert len(extracted_real) == 1
    assert extracted_real[0].is_estimated is False
    assert extracted_real[0].file_size_bytes == 5_000_000_000
