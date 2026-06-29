"""Tests for CLI helper logic."""

import httpx
import pytest
from typer import Exit

import whichllm.cli as cli_mod
import whichllm.__main__ as main_mod
from whichllm.cli import (
    _apply_memory_budgets,
    _apply_gpu_overrides,
    _auto_min_params_for_profile,
    _extract_id_size_b,
    _fill_missing_published_at,
    _format_fetch_error,
    _generate_chat_script,
    _include_vision_candidates,
    _merge_model_eval_benchmarks,
    _parse_memory_amount,
    _pick_gguf_variant,
    _resolve_ranked_gguf_for_run,
    _resolve_evidence_mode,
    _resolve_fit_filter,
    _resolve_speed_filter,
    _search_model,
    _validate_evidence,
    _validate_gpu_flags,
    app,
)
from whichllm.utils import _current_version
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from typer.testing import CliRunner


def _hw_with_gpu(vram_gb: int) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                memory_bandwidth_gbps=1.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=1,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )


def test_auto_min_params_general_by_vram():
    # Updated thresholds: tiny GPUs (4-8GB) get a lower floor so they can
    # surface full-GPU 3-4B models instead of being forced into 7B+
    # partial-offload-only candidates.
    assert _auto_min_params_for_profile(_hw_with_gpu(4), "general") == 2.0
    assert _auto_min_params_for_profile(_hw_with_gpu(6), "general") == 3.0
    assert _auto_min_params_for_profile(_hw_with_gpu(8), "general") == 5.0
    assert _auto_min_params_for_profile(_hw_with_gpu(12), "general") == 8.0
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "general") == 10.0
    assert _auto_min_params_for_profile(_hw_with_gpu(32), "general") == 12.0


def test_auto_min_params_non_general_disabled():
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "coding") is None


def test_auto_min_params_uses_usable_vram_budget():
    hw = _hw_with_gpu(20)
    hw.gpus[0].usable_vram_bytes = int(19.0 * 1024**3)

    assert _auto_min_params_for_profile(hw, "general") == 8.0


def test_auto_min_params_uses_ram_budget_for_shared_memory_gpu():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M2",
                vendor="apple",
                vram_bytes=16 * 1024**3,
                usable_vram_bytes=15 * 1024**3,
                shared_memory=True,
            )
        ],
        ram_bytes=16 * 1024**3,
        ram_budget_bytes=4 * 1024**3,
    )

    assert _auto_min_params_for_profile(hw, "general") == 2.0


def test_apply_gpu_overrides_accepts_multiple_simulated_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=64 * 1024**3, os="linux")

    _apply_gpu_overrides(hw, cpu_only=False, gpu=["2x RTX 4090"], vram=None)

    assert len(hw.gpus) == 2
    assert all(gpu.vendor == "nvidia" for gpu in hw.gpus)
    assert all(gpu.vram_bytes == 24 * 1024**3 for gpu in hw.gpus)


def test_validate_gpu_flags_allows_detected_vram_override():
    _validate_gpu_flags(cpu_only=False, gpu=None, vram=8.0, bandwidth=None)


def test_validate_gpu_flags_rejects_non_positive_overrides():
    with pytest.raises(Exit):
        _validate_gpu_flags(cpu_only=False, gpu=None, vram=0, bandwidth=None)
    with pytest.raises(Exit):
        _validate_gpu_flags(cpu_only=False, gpu=None, vram=None, bandwidth=-1)


def test_validate_gpu_flags_rejects_gpu_index_with_simulated_gpu():
    with pytest.raises(Exit):
        _validate_gpu_flags(
            cpu_only=False,
            gpu=["RTX 4090"],
            vram=24.0,
            bandwidth=None,
            gpu_index=0,
        )


def test_apply_gpu_overrides_updates_detected_shared_memory_gpu():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Intel UHD Graphics",
                vendor="intel",
                vram_bytes=0,
                shared_memory=True,
                memory_bandwidth_gbps=None,
            )
        ],
        cpu_name="CPU",
        cpu_cores=8,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )

    _apply_gpu_overrides(hw, cpu_only=False, gpu=None, vram=6.0, bandwidth=88.5)

    assert hw.gpus[0].vram_bytes == 6 * 1024**3
    assert hw.gpus[0].usable_vram_bytes is None
    assert hw.gpus[0].memory_bandwidth_gbps == 88.5
    assert hw.gpus[0].shared_memory is True
    assert hw.gpus[0].vram_overridden is True


def test_apply_gpu_overrides_updates_selected_detected_gpu_only():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA RTX 4060",
                vendor="nvidia",
                vram_bytes=8 * 1024**3,
                memory_bandwidth_gbps=272.0,
            ),
            GPUInfo(
                name="Intel UHD Graphics",
                vendor="intel",
                vram_bytes=0,
                shared_memory=True,
            ),
        ],
        cpu_name="CPU",
        cpu_cores=8,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )

    _apply_gpu_overrides(
        hw, cpu_only=False, gpu=None, vram=4.0, bandwidth=60.0, gpu_index=1
    )

    assert hw.gpus[0].vram_bytes == 8 * 1024**3
    assert hw.gpus[0].memory_bandwidth_gbps == 272.0
    assert hw.gpus[1].vram_bytes == 4 * 1024**3
    assert hw.gpus[1].memory_bandwidth_gbps == 60.0
    assert hw.gpus[1].vram_overridden is True


def test_apply_gpu_overrides_requires_gpu_index_for_multiple_detected_gpus():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(name="GPU 0", vendor="nvidia", vram_bytes=8 * 1024**3),
            GPUInfo(name="GPU 1", vendor="intel", vram_bytes=0, shared_memory=True),
        ],
        cpu_name="CPU",
        cpu_cores=8,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )

    with pytest.raises(Exit):
        _apply_gpu_overrides(hw, cpu_only=False, gpu=None, vram=4.0)


def test_apply_gpu_overrides_updates_simulated_gpu_bandwidth():
    hw = HardwareInfo(gpus=[], ram_bytes=32 * 1024**3, os="linux")

    _apply_gpu_overrides(
        hw, cpu_only=False, gpu=["Unknown GPU"], vram=4.0, bandwidth=72.0
    )

    assert len(hw.gpus) == 1
    assert hw.gpus[0].vram_bytes == 4 * 1024**3
    assert hw.gpus[0].memory_bandwidth_gbps == 72.0


def test_apply_gpu_overrides_rejects_override_without_gpu():
    hw = HardwareInfo(gpus=[], ram_bytes=32 * 1024**3, os="linux")

    with pytest.raises(Exit):
        _apply_gpu_overrides(hw, cpu_only=False, gpu=None, vram=4.0)


def test_include_vision_candidates_by_profile():
    assert _include_vision_candidates("vision") is True
    assert _include_vision_candidates("any") is True
    assert _include_vision_candidates("general") is False
    assert _include_vision_candidates("coding") is False


def test_fill_missing_published_at_updates_models():
    model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
    )

    async def _fake_fetch(ids: list[str]) -> dict[str, str]:
        assert ids == ["Qwen/Qwen3-8B-AWQ"]
        return {"Qwen/Qwen3-8B-AWQ": "2026-03-05T08:00:00.000Z"}

    updated = _fill_missing_published_at([model], [result], _fake_fetch)
    assert updated is True
    assert model.published_at == "2026-03-05T08:00:00.000Z"


def test_version_option_prints_version_and_exits():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert _current_version() in result.stdout


def test_module_entrypoint_uses_cli_app():
    assert main_mod.app is app


def test_format_fetch_error_uses_exception_class_when_message_is_empty():
    class EmptyNetworkError(Exception):
        def __str__(self) -> str:
            return ""

    assert _format_fetch_error(EmptyNetworkError()) == (
        "EmptyNetworkError with no detail from the network layer"
    )


def test_format_fetch_error_includes_status_and_url_for_empty_http_error():
    request = httpx.Request("GET", "https://huggingface.co/api/models")
    response = httpx.Response(429, request=request)
    error = httpx.HTTPStatusError("", request=request, response=response)

    assert _format_fetch_error(error) == (
        "HTTPStatusError: HTTP 429 for https://huggingface.co/api/models"
    )


def test_merge_model_eval_benchmarks_is_now_a_noop():
    """As of the self_reported evidence tier, _merge_model_eval_benchmarks
    must NOT mutate the leaderboard scores. Uploader-reported hf_eval values
    are consumed directly by the ranker as a separate, low-trust source.
    """
    model_direct_missing = ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        family_id="llama-3.1-8b",
        name="Llama-3.1-8B-Instruct",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 66.4},
    )
    model_already_present = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 70.0},
    )
    original = {"Qwen/Qwen2.5-7B-Instruct": 71.2}
    merged, injected = _merge_model_eval_benchmarks(
        [model_direct_missing, model_already_present],
        original,
    )
    # Function is a deprecation no-op now.
    assert injected == 0
    assert merged is original or merged == original
    # Critically, the uploader-reported value MUST NOT have been injected
    # under the model id, because doing so would make it appear as a
    # direct leaderboard hit.
    assert "meta-llama/Llama-3.1-8B-Instruct" not in merged


def test_validate_evidence_accepts_all_modes():
    assert _validate_evidence("strict") == "strict"
    assert _validate_evidence("base") == "base"
    assert _validate_evidence("any") == "any"


def test_validate_evidence_rejects_unknown_mode():
    with pytest.raises(Exit):
        _validate_evidence("foo")


def test_resolve_evidence_mode_direct_alias_wins():
    assert _resolve_evidence_mode("base", direct=True) == "strict"


def test_resolve_fit_filter_accepts_gpu_only_alias():
    assert _resolve_fit_filter("any", gpu_only=False) == "any"
    assert _resolve_fit_filter("gpu", gpu_only=False) == "full_gpu"
    assert _resolve_fit_filter("full-gpu", gpu_only=False) == "full_gpu"
    assert _resolve_fit_filter("full_gpu", gpu_only=False) == "full_gpu"
    assert _resolve_fit_filter("any", gpu_only=True) == "full_gpu"


def test_resolve_fit_filter_rejects_unknown_mode():
    with pytest.raises(Exit):
        _resolve_fit_filter("partial", gpu_only=False)


def test_resolve_speed_filter_presets_and_min_speed_override():
    assert _resolve_speed_filter("any", min_speed=None) is None
    assert _resolve_speed_filter("usable", min_speed=None) == 10.0
    assert _resolve_speed_filter("fast", min_speed=None) == 30.0
    assert _resolve_speed_filter("fast", min_speed=2.5) == 2.5


def test_resolve_speed_filter_rejects_unknown_mode():
    with pytest.raises(Exit):
        _resolve_speed_filter("slowish", min_speed=None)


def test_parse_memory_amount_supports_gb_mb_and_percent():
    assert _parse_memory_amount("1.5GB", option_name="--x") == int(1.5 * 1024**3)
    assert _parse_memory_amount("512MB", option_name="--x") == 512 * 1024**2
    assert _parse_memory_amount("8", option_name="--x") == 8 * 1024**3
    assert (
        _parse_memory_amount("10%", option_name="--x", total_bytes=20 * 1024**3)
        == 2 * 1024**3
    )


def test_apply_memory_budgets_sets_vram_headroom_and_ram_budget():
    hw = _hw_with_gpu(16)

    _apply_memory_budgets(hw, vram_headroom="1GB", ram_budget="8GB")

    assert hw.gpus[0].vram_bytes == 16 * 1024**3
    assert hw.gpus[0].usable_vram_bytes == 15 * 1024**3
    assert hw.ram_budget_bytes == 8 * 1024**3
    assert any("VRAM headroom" in note for note in hw.budget_notes)
    assert any("RAM budget" in note for note in hw.budget_notes)


def test_apply_memory_budgets_validates_vram_headroom_without_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=16 * 1024**3)

    with pytest.raises(Exit):
        _apply_memory_budgets(hw, vram_headroom="nope", ram_budget=None)


def test_apply_memory_budgets_accepts_valid_noop_vram_headroom_without_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=16 * 1024**3)

    _apply_memory_budgets(hw, vram_headroom="10%", ram_budget=None)

    assert hw.gpus == []
    assert hw.ram_budget_bytes is None


def test_main_passes_gpu_only_fit_filter(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["fit_filter"] = kwargs.get("fit_filter")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                quality_score=80.0,
            )
        ]

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking",
        lambda results, **kwargs: None,
    )

    result = CliRunner().invoke(app, ["--gpu-only"])

    assert result.exit_code == 0
    assert captured["fit_filter"] == "full_gpu"


def test_main_passes_speed_preset_and_default_runtime_columns(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["min_speed"] = kwargs.get("min_speed")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                estimated_tok_per_sec=8.0,
                quality_score=80.0,
            )
        ]

    def fake_display_ranking(results, **kwargs):
        captured["show_status"] = kwargs.get("show_status")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichllm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--speed", "usable"])

    assert result.exit_code == 0
    assert captured["min_speed"] == 10.0
    assert captured["show_status"] is True


def test_main_details_flag_restores_metadata_columns(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["show_status"] = kwargs.get("show_status")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichllm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--details", "--min-params", "1"])

    assert result.exit_code == 0
    assert captured["show_status"] is False


def test_main_markdown_alias_dispatches_markdown_output(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                estimated_tok_per_sec=12.0,
                quality_score=80.0,
            )
        ]

    def fake_display_markdown(results, hardware, **kwargs):
        captured["called"] = True
        captured["show_status"] = kwargs.get("show_status")

    def fail_display_hardware(hardware):
        raise AssertionError("markdown output should not render Rich hardware panel")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_markdown", fake_display_markdown
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", fail_display_hardware
    )

    result = CliRunner().invoke(app, ["-m"])

    assert result.exit_code == 0
    assert captured["called"] is True
    assert captured["show_status"] is True


def test_main_json_and_markdown_are_mutually_exclusive():
    result = CliRunner().invoke(app, ["--json", "--markdown"])

    assert result.exit_code == 1
    assert "--json and --markdown are mutually exclusive" in result.stdout


def test_main_empty_gpu_only_result_shows_fit_message(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["empty_message"] = kwargs.get("empty_message")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichllm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--fit", "full-gpu", "--min-params", "1"])

    assert result.exit_code == 0
    assert "No full-GPU models found" in captured["empty_message"]


# --------------- plan command tests ---------------


def test_plan_no_model_found_shows_error(monkeypatch):
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda: [])
    runner = CliRunner()
    result = runner.invoke(app, ["plan", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_plan_display_plan_renders_tables():
    """display_plan should render model info, VRAM table, and GPU table."""
    from whichllm.output.display import display_plan

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )
    # Should not raise
    display_plan(model, context_length=4096, target_quant="Q4_K_M")


def test_plan_display_plan_json_outputs_valid_json():
    """display_plan_json should output valid JSON."""
    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichllm.output.display import display_plan_json

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )
    # Capture output
    buf = StringIO()
    import whichllm.output._console as console_mod

    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_plan_json(model, context_length=4096, target_quant="Q4_K_M")
    finally:
        console_mod.console = orig_console
    raw = buf.getvalue().strip()
    data = json_mod.loads(raw)
    assert data["model"]["id"] == "test-org/Test-Model-7B-GGUF"
    assert "vram_by_quant" in data
    assert "gpu_compatibility" in data
    assert data["target_quant"] == "Q4_K_M"


# --------------- helper tests ---------------


def _make_model(
    model_id="org/Test-7B-GGUF",
    downloads=100,
    gguf_variants=None,
    parameter_count=7_000_000_000,
):
    return ModelInfo(
        id=model_id,
        family_id="test-7b",
        name="Test-7B",
        parameter_count=parameter_count,
        downloads=downloads,
        likes=10,
        gguf_variants=gguf_variants or [],
    )


def test_search_model_exact_match():
    models = [_make_model("org/Llama-8B"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "org/Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_endswith_match():
    models = [_make_model("org/Llama-8B"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_term_match():
    models = [_make_model("org/Llama-3.1-8B-GGUF"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "llama 8b")
    assert result.id == "org/Llama-3.1-8B-GGUF"


def test_search_model_not_found():
    models = [_make_model("org/Llama-8B")]
    with pytest.raises(Exit):
        _search_model(models, "nonexistent_xyz")


# --- regression tests for size-token substring matching (#107) ---


def test_search_model_7b_does_not_match_1_7b():
    """'qwen 7b' should NOT match Qwen3-1.7B (issue #107)."""
    models = [
        _make_model(
            "org/Qwen3-1.7B-GGUF", downloads=9999, parameter_count=1_700_000_000
        ),
        _make_model("org/Qwen3-7B-GGUF", downloads=100, parameter_count=7_000_000_000),
    ]
    result = _search_model(models, "qwen 7b")
    assert result.id == "org/Qwen3-7B-GGUF"


def test_search_model_2b_does_not_match_12b():
    """'gemma 2b' should NOT match gemma-3-12b-it (issue #107)."""
    models = [
        _make_model(
            "google/gemma-3-12b-it", downloads=5000, parameter_count=12_000_000_000
        ),
        _make_model("google/gemma-2b", downloads=100, parameter_count=2_000_000_000),
    ]
    result = _search_model(models, "gemma 2b")
    assert result.id == "google/gemma-2b"


def test_search_model_3b_does_not_match_30b_a3b():
    """'qwen 3b' should NOT match Qwen3-30B-A3B (issue #107)."""
    models = [
        _make_model(
            "org/Qwen3-30B-A3B-GGUF", downloads=8000, parameter_count=30_000_000_000
        ),
        _make_model("org/Qwen3-3B-GGUF", downloads=50, parameter_count=3_000_000_000),
    ]
    result = _search_model(models, "qwen 3b")
    assert result.id == "org/Qwen3-3B-GGUF"


def test_search_model_no_size_token_still_works():
    """Queries without a size token should still use plain substring matching."""
    models = [
        _make_model("org/Llama-3.1-8B-GGUF", parameter_count=8_000_000_000),
        _make_model("org/Qwen-7B", parameter_count=7_000_000_000),
    ]
    result = _search_model(models, "llama gguf")
    assert result.id == "org/Llama-3.1-8B-GGUF"


def test_search_model_millions_size_token():
    """'500m' size token should match a ~500M parameter model."""
    models = [
        _make_model("org/SmolLM-500M", downloads=200, parameter_count=500_000_000),
        _make_model("org/SmolLM-1.7B", downloads=300, parameter_count=1_700_000_000),
    ]
    result = _search_model(models, "smollm 500m")
    assert result.id == "org/SmolLM-500M"


def test_search_model_decimal_size_token():
    """'0.5b' size token should match a ~500M parameter model."""
    models = [
        _make_model("org/TinyModel-0.5B", downloads=100, parameter_count=500_000_000),
        _make_model("org/TinyModel-3B", downloads=500, parameter_count=3_000_000_000),
    ]
    result = _search_model(models, "tinymodel 0.5b")
    assert result.id == "org/TinyModel-0.5B"


def test_search_model_zero_param_count_passes_through():
    """Models with parameter_count=0 (missing metadata) should not be excluded."""
    models = [
        _make_model("org/Mystery-7B-GGUF", downloads=500, parameter_count=0),
    ]
    result = _search_model(models, "mystery 7b")
    assert result.id == "org/Mystery-7B-GGUF"


def test_search_model_first_size_token_wins():
    """When multiple size tokens appear, only the first is used as a filter."""
    models = [
        _make_model(
            "org/Qwen3-30B-A3B-GGUF", downloads=8000, parameter_count=30_000_000_000
        ),
        _make_model(
            "org/Qwen3-7B-3B-GGUF", downloads=100, parameter_count=7_000_000_000
        ),
    ]
    # "7b" is the first size token and filters by ~7B; "3b" becomes a text term
    result = _search_model(models, "qwen 7b 3b")
    assert result.id == "org/Qwen3-7B-3B-GGUF"


def test_search_model_7b_prefers_closest_size_over_downloads():
    """'qwen 7b' should pick 8B (closest to 7B) over 4B even if 4B has more downloads."""
    models = [
        _make_model("org/Qwen3-4B-GGUF", downloads=9000, parameter_count=4_000_000_000),
        _make_model("org/Qwen3-8B-GGUF", downloads=100, parameter_count=8_000_000_000),
    ]
    result = _search_model(models, "qwen 7b")
    assert result.id == "org/Qwen3-8B-GGUF"


def test_search_model_3b_prefers_closest_size_over_downloads():
    """'qwen 3b' should pick 3B (exact) over 4B even if 4B has more downloads."""
    models = [
        _make_model("org/Qwen3-4B-GGUF", downloads=9000, parameter_count=4_000_000_000),
        _make_model("org/Qwen3-3B-GGUF", downloads=50, parameter_count=3_000_000_000),
    ]
    result = _search_model(models, "qwen 3b")
    assert result.id == "org/Qwen3-3B-GGUF"


def test_search_model_size_tiebreak_falls_back_to_downloads():
    """When two models are equally close in size, prefer the one with more downloads."""
    models = [
        _make_model(
            "org/Qwen3-8B-A-GGUF", downloads=100, parameter_count=8_000_000_000
        ),
        _make_model(
            "org/Qwen3-8B-B-GGUF", downloads=500, parameter_count=8_000_000_000
        ),
    ]
    result = _search_model(models, "qwen 7b")
    assert result.id == "org/Qwen3-8B-B-GGUF"


def test_search_model_unknown_param_count_ranks_after_known():
    """Models with parameter_count=0 should rank after models with known sizes."""
    models = [
        _make_model("org/Qwen3-Unknown-GGUF", downloads=9999, parameter_count=0),
        _make_model("org/Qwen3-7B-GGUF", downloads=10, parameter_count=7_000_000_000),
    ]
    result = _search_model(models, "qwen 7b")
    assert result.id == "org/Qwen3-7B-GGUF"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("org/Qwen3-8B-GGUF", 8.0),
        ("org/Qwen3.5-9B-NVFP4", 9.0),
        ("org/Qwen3-30B-A3B-GGUF", 30.0),
        ("org/SmolLM-500M", 0.5),
        ("org/TinyModel-1.7B-Chat", 1.7),
        ("org/Llama-3.1-8B-Instruct", 8.0),
        ("org/NoSizeLabel-GGUF", None),
    ],
)
def test_extract_id_size_b(model_id, expected):
    assert _extract_id_size_b(model_id) == expected


def test_search_model_7b_prefers_id_label_over_param_count():
    """'qwen 7b' should not pick a 9B-labeled repo even if its param count is closer."""
    models = [
        _make_model(
            "org/Qwen3.5-9B-NVFP4", downloads=500, parameter_count=6_600_000_000
        ),
        _make_model("org/Qwen3-8B-GGUF", downloads=100, parameter_count=8_000_000_000),
    ]
    result = _search_model(models, "qwen 7b")
    assert result.id == "org/Qwen3-8B-GGUF"


def test_pick_gguf_variant_by_preference():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = _make_model(gguf_variants=variants)
    result = _pick_gguf_variant(model)
    assert result.quant_type == "Q4_K_M"


def test_pick_gguf_variant_with_filter():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = _make_model(gguf_variants=variants)
    result = _pick_gguf_variant(model, quant_filter="Q2_K")
    assert result.quant_type == "Q2_K"


def test_pick_gguf_variant_no_variants():
    model = _make_model(gguf_variants=[])
    result = _pick_gguf_variant(model)
    assert result is None


def test_resolve_ranked_synthetic_gguf_to_real_repo():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3.6-27B-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, real_gguf])

    assert resolved is not None
    model, variant = resolved
    assert model.id == "unsloth/Qwen3.6-27B-GGUF"
    assert variant.filename == "Qwen3.6-27B-Q4_K_M.gguf"


def test_resolve_ranked_synthetic_gguf_prefers_exact_quant():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    q4_match = ModelInfo(
        id="smaller/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=10,
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, q5_only, q4_match],
    )

    assert resolved is not None
    model, variant = resolved
    assert model.id == "smaller/Qwen3.6-27B-GGUF"
    assert variant.quant_type == "Q4_K_M"


def test_resolve_ranked_synthetic_gguf_rejects_quant_mismatch():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, q5_only])

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_without_real_repo_returns_none():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    unrelated = ModelInfo(
        id="other/Model-7B-GGUF",
        family_id="model-7b",
        name="Model-7B-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="other.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    assert (
        _resolve_ranked_gguf_for_run(selected, synthetic, [selected, unrelated]) is None
    )


def test_resolve_ranked_synthetic_gguf_rejects_size_mismatch():
    selected = ModelInfo(
        id="deepseek-ai/DeepSeek-V4-Flash",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash",
        parameter_count=158_000_000_000,
    )
    mtp_head = ModelInfo(
        id="converter/deepseek-v4-flash-mtp-gguf",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash-MTP-GGUF",
        parameter_count=6_600_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="mtp.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="DeepSeek-V4-Flash.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=90_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, mtp_head])

    assert resolved is None


# --------------- run/snippet command tests ---------------


def test_run_exits_gracefully():
    """run should fail gracefully (uv missing, or no model found)."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "some-model"])
    if result.exit_code != 0:
        assert any(
            msg in result.stdout
            for msg in ("uv is required", "No model found", "llama-cpp-python")
        )


def test_transformers_chat_script_passes_tokenizer_mapping_to_generate():
    model = _make_model(model_id="org/Test-7B")

    script = _generate_chat_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert "return_dict=True" in script
    assert "kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer)" in script
    assert "kwargs=dict(input_ids=inputs" not in script


def test_transformers_chat_script_provides_disk_offload_folder():
    model = _make_model(model_id="org/Test-7B")

    script = _generate_chat_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert 'tempfile.mkdtemp(prefix="whichllm_transformers_offload_")' in script
    assert "offload_folder=offload_folder" in script
    assert "shutil.rmtree(offload_folder, ignore_errors=True)" in script


def test_run_auto_pick_resolves_ranked_gguf_before_launch(monkeypatch):
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["quant_filter"] = kwargs.get("quant_filter")
        return [
            CompatibilityResult(
                model=selected,
                gguf_variant=synthetic,
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=90.0,
            )
        ]

    def fake_generate_chat_script(model, variant, context_length, cpu_only):
        captured["model_id"] = model.id
        captured["variant"] = variant
        return "print('ok')"

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [selected, real_gguf])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(cli_mod, "_generate_chat_script", fake_generate_chat_script)
    monkeypatch.setattr("subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", "--quant", "Q4_K_M"])

    assert result.exit_code == 0
    assert captured["quant_filter"] == "Q4_K_M"
    assert captured["model_id"] == "unsloth/Qwen3.6-27B-GGUF"
    assert captured["variant"].filename == "q4.gguf"
    assert "llama-cpp-python" in captured["cmd"]
    assert "transformers" not in captured["cmd"]


def test_snippet_no_model_found(monkeypatch):
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [])
    runner = CliRunner()
    result = runner.invoke(app, ["snippet", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_json_output_includes_benchmark_source_and_confidence():
    """display_json should include benchmark_source and benchmark_confidence."""
    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichllm.output.display import display_json

    model = ModelInfo(
        id="test-org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=100,
        likes=10,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=8_000_000_000,
        vram_available_bytes=24_000_000_000,
        quality_score=55.0,
        benchmark_status="estimated",
        benchmark_source="line_interp",
        benchmark_confidence=0.34,
    )
    hw = HardwareInfo(
        gpus=[],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_budget_bytes=32 * 1024**3,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
        budget_notes=["RAM budget: 32.0 GB"],
    )

    buf = StringIO()
    import whichllm.output._console as console_mod

    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_json([result], hw)
    finally:
        console_mod.console = orig_console

    data = json_mod.loads(buf.getvalue().strip())
    entry = data["models"][0]
    assert data["hardware"]["ram_budget_bytes"] == 32 * 1024**3
    assert data["hardware"]["budget_notes"] == ["RAM budget: 32.0 GB"]
    assert entry["artifact_repo_id"] is None
    assert entry["artifact_filename"] is None
    assert entry["benchmark_status"] == "estimated"
    assert entry["benchmark_source"] == "line_interp"
    assert entry["benchmark_confidence"] == 0.34


def test_json_output_includes_resolved_artifact_fields():
    """display_json should expose the actual GGUF repo/file when resolved."""
    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichllm.output.display import display_json

    model = ModelInfo(
        id="Qwen/Qwen3-4B-Thinking-2507",
        family_id="qwen3-4b-thinking",
        name="Qwen3-4B-Thinking-2507",
        parameter_count=4_000_000_000,
    )
    artifact = ModelInfo(
        id="MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF",
        family_id="qwen3-4b-thinking",
        name="Qwen3-4B-Thinking-2507-GGUF",
        parameter_count=4_000_000_000,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=GGUFVariant(
            filename="Qwen3-4B-Thinking-2507.Q3_K_M.gguf",
            quant_type="Q3_K_M",
            file_size_bytes=2_000_000_000,
        ),
        artifact_model=artifact,
        artifact_variant=GGUFVariant(
            filename="Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
            quant_type="Q3_K_M",
            file_size_bytes=2_000_000_000,
        ),
        can_run=True,
        vram_required_bytes=3_000_000_000,
        vram_available_bytes=8_000_000_000,
    )
    hw = HardwareInfo(
        gpus=[],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )

    buf = StringIO()
    import whichllm.output._console as console_mod

    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_json([result], hw)
    finally:
        console_mod.console = orig_console

    entry = json_mod.loads(buf.getvalue().strip())["models"][0]
    assert entry["model_id"] == "Qwen/Qwen3-4B-Thinking-2507"
    assert entry["artifact_repo_id"] == "MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF"
    assert entry["artifact_filename"] == "Qwen3-4B-Thinking-2507-Q3_K_M.gguf"
