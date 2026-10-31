"""Tests for CLI helper logic."""

import ast
import re
from pathlib import Path

import httpx
import pytest
from typer import Exit

import whichllm.commands.run as cli_mod
import whichllm.commands.snippet as snippet_mod
import whichllm.__main__ as main_mod
from whichllm.cli_validation import (
    _validate_ranking_flags,
    _apply_memory_budgets,
    _apply_gpu_overrides,
    _auto_min_params_for_profile,
    _fill_missing_published_at,
    _include_vision_candidates,
    _merge_model_eval_benchmarks,
    _parse_memory_amount,
    _resolve_evidence_mode,
    _resolve_fit_filter,
    _resolve_speed_filter,
    _validate_evidence,
    _validate_gpu_flags,
)
from whichllm.cli_models import (
    _extract_id_size_b,
    _generate_chat_script,
    _pick_gguf_variant,
    _search_model,
)
from whichllm.cli_shared import _format_fetch_error
from whichllm.models import lmstudio
from whichllm.models.artifacts import (
    resolve_ranked_gguf_artifact as _resolve_ranked_gguf_for_run,
)
from whichllm.cli import app
from whichllm.utils import _current_version
from whichllm.engine.types import CompatibilityResult, Placement
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
    assert _resolve_speed_filter("usable", min_speed=None) == 8.0
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
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
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
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichllm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--speed", "usable"])

    assert result.exit_code == 0
    assert captured["min_speed"] == 8.0
    assert captured["show_status"] is True


def test_main_passes_repeatable_lm_studio_paths(monkeypatch, tmp_path):
    model = ModelInfo(
        id="org/Test-7B-GGUF",
        family_id="test-7b",
        name="Test-7B-GGUF",
        parameter_count=7_000_000_000,
        published_at="2026-01-01T00:00:00.000Z",
    )
    variant = GGUFVariant(
        filename="Test-7B-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=4 * 1024**3,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=5 * 1024**3,
        vram_available_bytes=8 * 1024**3,
    )
    captured: dict[str, object] = {}

    def fake_discover(paths):
        captured["paths"] = [str(path) for path in paths]
        return []

    def fake_attach(results, local_models):
        captured["attached"] = results == [result] and local_models == []

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models", lambda *args, **kwargs: [result]
    )
    monkeypatch.setattr(
        "whichllm.models.lmstudio.discover_lmstudio_ggufs", fake_discover
    )
    monkeypatch.setattr("whichllm.models.lmstudio.attach_local_matches", fake_attach)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    cli_result = CliRunner().invoke(
        app,
        [
            "--lm-studio-path",
            str(first),
            "--lm-studio-path",
            str(second),
        ],
    )

    assert cli_result.exit_code == 0
    assert captured["paths"] == [str(first), str(second)]
    assert captured["attached"] is True


def test_main_reports_missing_explicit_lm_studio_path(tmp_path):
    missing = tmp_path / "missing"

    result = CliRunner().invoke(app, ["--lm-studio-path", str(missing)])

    assert result.exit_code == 1
    assert "LM Studio path does not exist" in result.output
    assert str(missing) in result.output.replace("\n", "")


def test_main_details_flag_restores_metadata_columns(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["show_status"] = kwargs.get("show_status")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
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
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
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


def test_main_top_zero_rejected():
    result = CliRunner().invoke(app, ["--top", "0"])

    assert result.exit_code == 1
    assert "--top must be 1 or greater" in result.stdout


def test_main_top_negative_rejected():
    # ``results[:-5]`` would silently drop the best-ranked models.
    result = CliRunner().invoke(app, ["--top=-5"])

    assert result.exit_code == 1
    assert "--top must be 1 or greater" in result.stdout


def test_main_negative_min_speed_rejected():
    result = CliRunner().invoke(app, ["--min-speed=-1"])

    assert result.exit_code == 1
    assert "--min-speed must be 0 or greater" in result.stdout


def test_main_negative_min_params_rejected():
    result = CliRunner().invoke(app, ["--min-params=-2"])

    assert result.exit_code == 1
    assert "--min-params must be 0 or greater" in result.stdout


def test_validate_ranking_flags_accepts_valid_values():
    # A valid combination must not raise (no recommendations are dropped).
    _validate_ranking_flags(top=10, min_speed=0.0, min_params=None)
    _validate_ranking_flags(top=1, min_speed=None, min_params=7.0)


def test_upgrade_top_zero_rejected():
    # The upgrade command shares the same --top -> results[:top_n] hazard.
    result = CliRunner().invoke(app, ["upgrade", "RTX 4090", "--top", "0"])

    assert result.exit_code == 1
    assert "--top must be 1 or greater" in result.stdout


def test_main_empty_gpu_only_result_shows_fit_message(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["empty_message"] = kwargs.get("empty_message")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
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
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    runner = CliRunner()
    result = runner.invoke(app, ["plan", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_plan_fetches_exact_repo_id_when_missing_from_cache(monkeypatch):
    import whichllm.models.fetcher as fetcher
    import whichllm.models.hf as hf

    requested: list[str] = []
    displayed: list[str] = []

    async def fake_fetch_model_by_id(model_id):
        requested.append(model_id)
        return _make_model(model_id, parameter_count=8_202_227_712)

    async def unexpected_fetch_models(*args, **kwargs):
        raise AssertionError("Exact Repo ID lookup must not fetch the bulk catalog")

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: None)
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)
    monkeypatch.setattr(fetcher, "fetch_models", unexpected_fetch_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_plan",
        lambda model, context_length, target_quant: displayed.append(model.id),
    )

    result = CliRunner().invoke(app, ["plan", "ilsp/Llama-Krikri-8B-Instruct"])

    assert result.exit_code == 0
    assert requested == ["ilsp/Llama-Krikri-8B-Instruct"]
    assert displayed == ["ilsp/Llama-Krikri-8B-Instruct"]


def test_plan_uses_cached_exact_repo_without_network(monkeypatch):
    import whichllm.models.hf as hf
    from whichllm.models.fetcher import models_to_dicts

    cached_model = _make_model("org/Cached-8B")
    displayed: list[str] = []

    async def unexpected_fetch(model_id):
        raise AssertionError(f"Unexpected direct fetch for {model_id}")

    monkeypatch.setattr(
        "whichllm.models.cache.load_cache",
        lambda **_kwargs: models_to_dicts([cached_model]),
    )
    monkeypatch.setattr(hf, "fetch_model_by_id", unexpected_fetch)
    monkeypatch.setattr(
        "whichllm.output.display.display_plan",
        lambda model, context_length, target_quant: displayed.append(model.id),
    )

    result = CliRunner().invoke(app, ["plan", "org/Cached-8B"])

    assert result.exit_code == 0
    assert displayed == ["org/Cached-8B"]


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (404, "Repository not found on Hugging Face"),
        (401, "private or gated"),
        (403, "private or gated"),
    ],
)
def test_plan_reports_repo_fetch_http_errors(monkeypatch, status_code, expected):
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{model_id}")
        response = httpx.Response(status_code, request=request)
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Missing-8B"])

    assert result.exit_code != 0
    assert expected in " ".join(result.stdout.split())


def test_plan_reports_missing_repo_from_hf_401_body(monkeypatch):
    """Hugging Face answers 401 for repositories that do not exist."""
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{model_id}")
        response = httpx.Response(
            401, json={"error": "Invalid username or password."}, request=request
        )
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Typo-8B"])
    output = " ".join(result.stdout.split())

    assert result.exit_code != 0
    assert "Repository not found on Hugging Face" in output


def test_plan_reports_missing_repo_from_hf_plain_text_401(monkeypatch):
    """The error body is not always JSON."""
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{model_id}")
        response = httpx.Response(401, text="Invalid credentials", request=request)
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Typo-8B"])
    output = " ".join(result.stdout.split())

    assert result.exit_code != 0
    assert "Repository not found on Hugging Face" in output


def test_plan_reports_restricted_repo_401_body_as_access_error(monkeypatch):
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{model_id}")
        response = httpx.Response(
            401,
            json={
                "error": "Access to model org/Gated-8B is restricted and you are "
                "not in the authorized list."
            },
            request=request,
        )
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Gated-8B"])
    output = " ".join(result.stdout.split())

    assert result.exit_code != 0
    assert "Cannot access Hugging Face repository" in output
    assert "private or gated" in output
    assert "Repository not found" not in output


def test_plan_reports_repo_with_insufficient_metadata(monkeypatch):
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        return None

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Metadata-Free-Model"])

    assert result.exit_code != 0
    assert "does not expose enough model metadata" in " ".join(result.stdout.split())


def test_plan_reports_repo_network_error(monkeypatch):
    import whichllm.models.hf as hf

    async def fake_fetch_model_by_id(model_id):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{model_id}")
        raise httpx.ConnectError("network unavailable", request=request)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/Remote-8B"])
    output = " ".join(result.stdout.split())

    assert result.exit_code != 0
    assert "Error fetching Hugging Face repository" in output
    assert "network unavailable" in output


def test_plan_does_not_fetch_malformed_repo_id(monkeypatch):
    import whichllm.models.hf as hf

    requested: list[str] = []

    async def fake_fetch_model_by_id(model_id):
        requested.append(model_id)
        return _make_model(model_id)

    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr(hf, "fetch_model_by_id", fake_fetch_model_by_id)

    result = CliRunner().invoke(app, ["plan", "org/nested/Model-8B"])

    assert result.exit_code != 0
    assert requested == []
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
    **kwargs,
):
    return ModelInfo(
        id=model_id,
        family_id="test-7b",
        name="Test-7B",
        parameter_count=parameter_count,
        downloads=downloads,
        likes=10,
        gguf_variants=gguf_variants or [],
        **kwargs,
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


def test_search_model_skips_draft_head_sized_from_its_name():
    """A draft head whose size was parsed from its name must not beat the model."""
    models = [
        _make_model(
            "AtomicChat/Qwen3-8B-DFlash-GGUF",
            downloads=0,
            parameter_count=8_000_000_000,
        ),
        _make_model(
            "Qwen/Qwen3-8B", downloads=8_000_000, parameter_count=8_190_000_000
        ),
    ]
    assert _search_model(models, "qwen3 8b").id == "Qwen/Qwen3-8B"
    assert (
        _search_model(models, "qwen3 8b gguf").id == "AtomicChat/Qwen3-8B-DFlash-GGUF"
    )
    assert (
        _search_model(models, "qwen3 8b dflash").id == "AtomicChat/Qwen3-8B-DFlash-GGUF"
    )


def test_search_model_same_size_label_ranks_by_downloads_not_param_count():
    """The parameter count is often parsed from the label, so it cannot break a tie."""
    models = [
        _make_model(
            "someone/Qwen3-8B-GGUF", downloads=0, parameter_count=8_000_000_000
        ),
        _make_model(
            "Qwen/Qwen3-8B-GGUF", downloads=670_000, parameter_count=8_190_000_000
        ),
    ]
    assert _search_model(models, "qwen3 8b").id == "Qwen/Qwen3-8B-GGUF"


def test_search_model_prefers_original_over_quantized_upload():
    models = [
        _make_model(
            "unsloth/Llama-3.3-70B-Instruct",
            downloads=650_000,
            parameter_count=70_000_000_000,
            base_model_relation="quantized",
        ),
        _make_model(
            "meta-llama/Llama-3.3-70B-Instruct",
            downloads=390_000,
            parameter_count=70_000_000_000,
            base_model_relation="finetune",
        ),
        _make_model(
            "unsloth/Llama-3.3-70B-Instruct-GGUF",
            downloads=300_000,
            parameter_count=70_000_000_000,
            base_model_relation="quantized",
        ),
        # No base_model tag, so it would pass for an original.
        _make_model(
            "someone/Llama-3.3-70B-Instruct-GGUF",
            downloads=50_000,
            parameter_count=70_000_000_000,
        ),
    ]
    assert (
        _search_model(models, "llama 3 70b").id == "meta-llama/Llama-3.3-70B-Instruct"
    )
    # A format word asks for an upload, so downloads decide among them.
    assert (
        _search_model(models, "llama 70b gguf").id
        == "unsloth/Llama-3.3-70B-Instruct-GGUF"
    )


def test_search_model_prefers_closest_name():
    """Tokens the query does not mention (VL, MedGemma, FP8) count against a repo."""
    models = [
        _make_model(
            "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            downloads=1_200_000,
            parameter_count=8_000_000_000,
        ),
        _make_model(
            "Qwen/Qwen3-8B-GGUF", downloads=670_000, parameter_count=8_000_000_000
        ),
        _make_model(
            "google/medgemma-4b-it", downloads=900_000, parameter_count=4_000_000_000
        ),
        _make_model(
            "google/gemma-4-31B-it", downloads=500_000, parameter_count=31_000_000_000
        ),
        _make_model(
            "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            downloads=1_100_000,
            parameter_count=30_000_000_000,
        ),
        _make_model(
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            downloads=870_000,
            parameter_count=30_000_000_000,
        ),
    ]
    assert _search_model(models, "qwen3 8b gguf").id == "Qwen/Qwen3-8B-GGUF"
    assert _search_model(models, "gemma 4").id == "google/gemma-4-31B-it"
    assert (
        _search_model(models, "qwen3 coder").id == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    )


def test_search_model_ignores_versions_and_org_prefix_in_names():
    models = [
        _make_model(
            "someone/Mistral-Small-24B-Instruct-Jbliterated", downloads=900_000
        ),
        _make_model("mistralai/Mistral-Small-3.1-24B-Instruct-2503", downloads=800_000),
        _make_model("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", downloads=1_700_000),
        _make_model("unsloth/Nemotron-3-Nano-30B-A3B-GGUF", downloads=400_000),
    ]
    assert (
        _search_model(models, "mistral small").id
        == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    assert (
        _search_model(models, "nemotron nano").id
        == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    )


def test_search_model_skips_base_checkpoint():
    models = [
        _make_model("Qwen/Qwen2.5-7B", downloads=900_000),
        _make_model("Qwen/Qwen2.5-7B-Instruct", downloads=800_000),
    ]
    assert _search_model(models, "qwen2.5 7b").id == "Qwen/Qwen2.5-7B-Instruct"


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
        base_model_relation="quantized",
        tags=("base_model:quantized:Qwen/Qwen3.6-27B",),
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


def test_resolve_ranked_synthetic_gguf_rejects_finetuned_repo():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    finetuned_gguf = ModelInfo(
        id="converter/Qwen3.6-27B-MTP-pi-tune-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-MTP-pi-tune-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        base_model="Qwen/Qwen3.6-27B",
        base_model_relation="finetune",
        tags=("base_model:finetune:Qwen/Qwen3.6-27B",),
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3.6-27B-MTP-Q4_K_M.gguf",
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
        [selected, finetuned_gguf],
    )

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_rejects_unproven_base_relation():
    selected = ModelInfo(
        id="google/gemma-4-31B-it",
        family_id="gemma-4-31b-it",
        name="gemma-4-31B-it",
        parameter_count=31_000_000_000,
    )
    unproven_gguf = ModelInfo(
        id="converter/Gemma-4-31B-custom-GGUF",
        family_id="gemma-4-31b-it",
        name="Gemma-4-31B-custom-GGUF",
        parameter_count=31_000_000_000,
        downloads=1_000_000,
        base_model="google/gemma-4-31b-it",
        gguf_variants=[
            GGUFVariant(
                filename="Gemma-4-31B-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=19_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="gemma-4-31B-it.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=19_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, unproven_gguf],
    )

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_rejects_other_quantized_checkpoint():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    other_checkpoint = ModelInfo(
        id="converter/Qwen3.6-27B-tuned-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-tuned-GGUF",
        parameter_count=27_000_000_000,
        base_model="tuner/Qwen3.6-27B-tuned",
        base_model_relation="quantized",
        tags=("base_model:quantized:tuner/Qwen3.6-27B-tuned",),
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3.6-27B-tuned-Q4_K_M.gguf",
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
        [selected, other_checkpoint],
    )

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_rejects_renamed_merge_claiming_quantized():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    merged_gguf = ModelInfo(
        id="converter/Qwopus3.6-27B-Fusion-GGUF",
        family_id="qwen3-27b",
        name="Qwopus3.6-27B-Fusion-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        base_model="Qwen/Qwen3.6-27B",
        base_model_relation="quantized",
        tags=(
            "gguf",
            "merge",
            "task-vector",
            "base_model:quantized:Qwen/Qwen3.6-27B",
        ),
        gguf_variants=[
            GGUFVariant(
                filename="Qwopus3.6-27B-Fusion-Q5_K_M.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=19_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q5_K_M.gguf",
        quant_type="Q5_K_M",
        file_size_bytes=19_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, merged_gguf],
    )

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_rejects_conflicting_base_relations():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    merged_gguf = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        base_model="Qwen/Qwen3.6-27B",
        base_model_relation="quantized",
        tags=(
            "gguf",
            "base_model:quantized:Qwen/Qwen3.6-27B",
            "base_model:finetune:Qwen/Qwen3.6-27B",
        ),
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

    assert (
        _resolve_ranked_gguf_for_run(
            selected,
            synthetic,
            [selected, merged_gguf],
        )
        is None
    )


def test_resolve_ranked_existing_gguf_repo_does_not_require_base_relation():
    direct_gguf = ModelInfo(
        id="author/custom-model-GGUF",
        family_id="custom-model",
        name="custom-model-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="custom-model-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )

    resolved = _resolve_ranked_gguf_for_run(
        direct_gguf,
        direct_gguf.gguf_variants[0],
        [direct_gguf],
    )

    assert resolved == (direct_gguf, direct_gguf.gguf_variants[0])


def test_resolve_ranked_synthetic_gguf_accepts_owner_prefixed_conversion_name():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    direct_gguf = ModelInfo(
        id="converter/Qwen_Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen_Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        base_model="Qwen/Qwen3.6-27B",
        base_model_relation="quantized",
        tags=("base_model:quantized:Qwen/Qwen3.6-27B",),
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

    resolved = _resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, direct_gguf],
    )

    assert resolved == (direct_gguf, direct_gguf.gguf_variants[0])


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
        base_model="Qwen/Qwen3.6-27B",
        base_model_relation="quantized",
        tags=("base_model:quantized:Qwen/Qwen3.6-27B",),
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


def test_run_requires_uv(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "some-model"])

    assert result.exit_code == 1
    assert "uv is required" in result.stdout


def _uv_only(name):
    return "/usr/bin/uv" if name == "uv" else None


def test_run_no_model_found_exits_gracefully(monkeypatch):
    monkeypatch.setattr("shutil.which", _uv_only)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [])

    runner = CliRunner()
    result = runner.invoke(app, ["run", "some-model"])

    assert result.exit_code == 1
    assert "No model found" in result.stdout


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


def test_gguf_chat_script_treats_hf_metadata_as_literals():
    model = _make_model(model_id="org/Test-7B")
    filename = 'weights-Q4_K_M.gguf"); print("injected"); #.gguf'
    variant = GGUFVariant(
        filename=filename,
        quant_type="Q4_K_M",
        file_size_bytes=1,
    )

    tree = ast.parse(
        _generate_chat_script(model, variant, context_length=4096, cpu_only=False)
    )
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"model_id", "filename", "quant_type"}
    }

    assert assignments == {
        "model_id": model.id,
        "filename": filename,
        "quant_type": variant.quant_type,
    }


def test_snippet_treats_hf_metadata_as_literals(monkeypatch):
    filename = 'weights-Q4_K_M.gguf"); print("injected"); #.gguf'
    model = _make_model(model_id="org/Test-7B")
    model.gguf_variants = [
        GGUFVariant(
            filename=filename,
            quant_type="Q4_K_M",
            file_size_bytes=1,
        )
    ]
    monkeypatch.setattr(snippet_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["snippet", "Test-7B"])

    assert result.exit_code == 0
    assert f"repo_id={model.id!r}" in result.stdout
    assert f"filename={filename!r}" in result.stdout


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
        base_model_relation="quantized",
        tags=("base_model:quantized:Qwen/Qwen3.6-27B",),
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

    def fake_generate_chat_script(model, variant, context_length, cpu_only, **kw):
        captured["model_id"] = model.id
        captured["variant"] = variant
        return "print('ok')"

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("shutil.which", _uv_only)
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
    # Without llama-cli the chat is a CPU-only source build; say so first.
    assert "runs on the CPU only" in " ".join(result.stdout.split())


def test_run_auto_pick_ranks_with_the_index_benchmarks(monkeypatch):
    """`run` without a name used to rank with the --live benchmark cache only,
    which is empty on the index path, so it picked without any benchmarks."""
    from whichllm.models.index_client import IndexCatalog

    model = ModelInfo(
        id="org/Model-8B-GGUF",
        family_id="org/model-8b",
        name="Model-8B-GGUF",
        parameter_count=8_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="m.gguf", quant_type="Q4_K_M", file_size_bytes=5 * 10**9
            )
        ],
    )
    index_scores = {"org/Model-8B": 61.0}
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["benchmark_scores"] = kwargs.get("benchmark_scores")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=model.gguf_variants[0],
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=60.0,
            )
        ]

    class Completed:
        returncode = 0

    monkeypatch.setattr("shutil.which", _uv_only)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr(
        "whichllm.models.index_client.load_index",
        lambda refresh=False: IndexCatalog(
            models=[model],
            benchmark_scores=index_scores,
            built_at="2026-10-29T00:00:00Z",
        ),
    )
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        cli_mod, "_generate_chat_script", lambda *a, **kw: "print('ok')"
    )
    monkeypatch.setattr("subprocess.run", lambda cmd: Completed())

    result = CliRunner().invoke(app, ["run"])

    assert result.exit_code == 0
    assert captured["benchmark_scores"] == index_scores


def _run_with_llama_cli(monkeypatch, models, rank_results, argv):
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/opt/llama/llama-cli" if name == "llama-cli" else None,
    )
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: models)
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models", lambda *a, **kw: rank_results
    )
    monkeypatch.setattr("subprocess.run", fake_run)
    result = CliRunner().invoke(app, argv)
    return result, captured


def test_run_uses_llama_cli_with_the_recipe_flags(monkeypatch):
    model = ModelInfo(
        id="org/MoE-35B-A3B-GGUF",
        family_id="org/moe-35b-a3b",
        name="MoE-35B-A3B-GGUF",
        parameter_count=35_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="m.gguf", quant_type="Q4_K_M", file_size_bytes=20 * 10**9
            )
        ],
    )
    picked = CompatibilityResult(
        model=model,
        gguf_variant=model.gguf_variants[0],
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=60.0,
        placement=Placement("expert_offload", 7, 13, n_cpu_moe=31, num_layers=40),
    )

    result, captured = _run_with_llama_cli(monkeypatch, [model], [picked], ["run"])

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == [
        "/opt/llama/llama-cli",
        "-hf",
        "org/MoE-35B-A3B-GGUF:Q4_K_M",
        "-c",
        "4096",
        "-ngl",
        "99",
        "--n-cpu-moe",
        "31",
        "-fa",
        "on",
    ]
    assert "uv" not in " ".join(captured["cmd"])


def test_run_named_model_lets_llama_cpp_fit_it(monkeypatch):
    model = ModelInfo(
        id="org/Small-2B-GGUF",
        family_id="org/small-2b",
        name="Small-2B-GGUF",
        parameter_count=2_000_000_000,
        gguf_variants=[
            GGUFVariant(filename="s.gguf", quant_type="Q4_K_M", file_size_bytes=10**9)
        ],
    )

    result, captured = _run_with_llama_cli(
        monkeypatch, [model], [], ["run", "org/Small-2B-GGUF", "-q", "Q4_K_M"]
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == [
        "/opt/llama/llama-cli",
        "-hf",
        "org/Small-2B-GGUF:Q4_K_M",
        "-c",
        "4096",
    ]


def test_run_ranks_like_the_main_command(monkeypatch):
    from whichllm.commands.autopick import pick_top_runnable
    from whichllm.models.index_client import IndexCatalog

    model = ModelInfo(
        id="org/Model-8B-GGUF",
        family_id="org/model-8b",
        name="Model-8B-GGUF",
        parameter_count=8_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="m.gguf", quant_type="Q4_K_M", file_size_bytes=5 * 10**9
            )
        ],
    )
    calls: list[tuple] = []

    def fake_rank_models(models, hardware, **kwargs):
        calls.append(
            (
                hardware.gpus[0].usable_vram_bytes,
                {k: kwargs.get(k) for k in sorted(kwargs) if k != "top_n"},
            )
        )
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=model.gguf_variants[0],
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=60.0,
            )
        ]

    catalog = IndexCatalog(
        models=[model],
        benchmark_scores={"org/Model-8B": 61.0},
        built_at="2026-10-29T00:00:00Z",
    )
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(12)
    )
    monkeypatch.setattr(
        "whichllm.models.index_client.load_index", lambda refresh=False: catalog
    )
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )

    assert CliRunner().invoke(app, []).exit_code == 0
    main_call = calls[-1]
    pick_top_runnable([model], context_length=4096, quant=None)

    # The same VRAM headroom, size floor and evidence rules as `whichllm`.
    assert calls[-1] == main_call
    assert main_call[0] < _hw_with_gpu(12).gpus[0].vram_bytes


def _local_gguf_model(
    model_id: str = "org/Test-7B-GGUF",
    filename: str = "q4.gguf",
) -> ModelInfo:
    """Build a GGUF model whose artifact can be matched in a local library."""
    return _make_model(
        model_id=model_id,
        gguf_variants=[
            GGUFVariant(
                filename=filename,
                quant_type="Q4_K_M",
                file_size_bytes=1,
            )
        ],
    )


def _write_library_file(
    root: Path,
    filename: str,
    *,
    repo_id: str = "org/Test-7B-GGUF",
) -> Path:
    """Create one GGUF file inside an LM Studio library layout."""
    owner, repo = repo_id.split("/", 1)
    path = root / owner / repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gguf")
    return path


def _capture_run(monkeypatch, captured: dict[str, object]) -> None:
    """Stub uv and capture the generated script instead of launching it."""

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        captured["script"] = Path(cmd[-1]).read_text()
        return Completed()

    monkeypatch.setattr("shutil.which", _uv_only)
    monkeypatch.setattr("subprocess.run", fake_run)


def _flat(text: str) -> str:
    """Collapse console line wrapping so long paths can be asserted."""
    return re.sub(r"\s+", "", text)


def test_run_uses_verified_local_artifact_without_hf_download(monkeypatch, tmp_path):
    model = _local_gguf_model()
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "q4.gguf")
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0
    assert _flat(f"Using local LM Studio file: {artifact}") in _flat(result.stdout)
    assert f"local_path = {str(artifact)!r}" in captured["script"]
    assert "all(os.path.isfile(part) for part in local_parts)" in captured["script"]
    assert "llama-cpp-python" in captured["cmd"]
    # The isolated build runs before the script starts, so the download stays as
    # the fallback for a file that goes away in between.
    assert "huggingface-hub" in captured["cmd"]


def test_run_uses_entry_part_of_complete_split_artifact(monkeypatch, tmp_path):
    parts = [f"q4-{part:05d}-of-00003.gguf" for part in (1, 2, 3)]
    model = _local_gguf_model(filename=parts[0])
    root = tmp_path / "lmstudio"
    for part in parts:
        _write_library_file(root, part)
    entry = root / "org" / "Test-7B-GGUF" / parts[0]
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0
    assert _flat(f"Using local LM Studio file: {entry}") in _flat(result.stdout)
    assert f"local_path = {str(entry)!r}" in captured["script"]
    assert str(entry.with_name(parts[1])) in captured["script"]


def test_run_warns_about_incomplete_explicit_split_set(monkeypatch, tmp_path):
    parts = [f"q4-{part:05d}-of-00003.gguf" for part in (1, 2, 3)]
    model = _local_gguf_model(filename=parts[0])
    root = tmp_path / "lmstudio"
    _write_library_file(root, parts[0])
    _write_library_file(root, parts[1])
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    flat_stdout = _flat(result.stdout)
    assert result.exit_code == 0
    assert "Using local LM Studio file" not in result.stdout
    assert _flat("is missing part 3 of 3; downloading instead") in flat_stdout
    assert "hf_hub_download" in captured["script"]
    assert "huggingface-hub" in captured["cmd"]


def test_run_stays_quiet_when_no_local_artifact_exists(monkeypatch, tmp_path):
    model = _local_gguf_model()
    root = tmp_path / "lmstudio"
    root.mkdir()
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0
    assert "Warning" not in result.stdout
    assert "Using local LM Studio file" not in result.stdout
    assert "hf_hub_download" in captured["script"]


def test_run_ignores_split_artifact_parts_from_another_library(monkeypatch, tmp_path):
    parts = [f"q4-{part:05d}-of-00002.gguf" for part in (1, 2)]
    model = _local_gguf_model(filename=parts[0])
    first, second = tmp_path / "lib-a", tmp_path / "lib-b"
    _write_library_file(first, parts[0])
    _write_library_file(second, parts[1])
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(
        app,
        [
            "run",
            "Test-7B",
            "--lm-studio-path",
            str(first),
            "--lm-studio-path",
            str(second),
        ],
    )

    assert result.exit_code == 0
    assert "Using local LM Studio file" not in result.stdout
    assert _flat("is missing part 2 of 2; downloading instead") in _flat(result.stdout)
    assert "hf_hub_download" in captured["script"]
    assert "huggingface-hub" in captured["cmd"]


def test_run_uses_complete_split_artifact_from_a_later_library(monkeypatch, tmp_path):
    parts = [f"q4-{part:05d}-of-00002.gguf" for part in (1, 2)]
    model = _local_gguf_model(filename=parts[0])
    first, second = tmp_path / "lib-a", tmp_path / "lib-b"
    _write_library_file(first, parts[0])
    for part in parts:
        _write_library_file(second, part)
    entry = second / "org" / "Test-7B-GGUF" / parts[0]
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(
        app,
        [
            "run",
            "Test-7B",
            "--lm-studio-path",
            str(first),
            "--lm-studio-path",
            str(second),
        ],
    )

    assert result.exit_code == 0
    assert _flat(f"Using local LM Studio file: {entry}") in _flat(result.stdout)
    assert "Warning" not in result.stdout
    assert f"local_path = {str(entry)!r}" in captured["script"]


def test_run_falls_back_when_local_file_disappears_before_launch(monkeypatch, tmp_path):
    model = _local_gguf_model()
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "q4.gguf")
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    real_find = lmstudio.find_local_artifact

    def find_then_remove(repo_id, artifact_path, local_models):
        match = real_find(repo_id, artifact_path, local_models)
        if match is not None:
            match.path.unlink()
        return match

    monkeypatch.setattr(cli_mod, "find_local_artifact", find_then_remove)

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0
    assert not artifact.exists()
    assert _flat("missing or incomplete; downloading instead") in _flat(result.stdout)
    assert "hf_hub_download" in captured["script"]
    assert "huggingface-hub" in captured["cmd"]


def test_run_reports_missing_explicit_lm_studio_path(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", _uv_only)
    missing = tmp_path / "absent"

    result = CliRunner().invoke(
        app, ["run", "Test-7B", "--lm-studio-path", str(missing)]
    )

    assert result.exit_code == 1
    assert "LM Studio path does not exist" in result.stdout


def test_run_llama_cli_opens_a_local_artifact(monkeypatch, tmp_path):
    model = _local_gguf_model("org/Small-2B-GGUF", "s.gguf")
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "s.gguf", repo_id="org/Small-2B-GGUF")

    result, captured = _run_with_llama_cli(
        monkeypatch,
        [model],
        [],
        ["run", "org/Small-2B-GGUF", "-q", "Q4_K_M", "--lm-studio-path", str(root)],
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == [
        "/opt/llama/llama-cli",
        "-m",
        str(artifact),
        "-c",
        "4096",
    ]
    assert "-hf" not in captured["cmd"]
    assert _flat(f"Using local LM Studio file: {artifact}") in _flat(result.stdout)


def test_run_llama_cli_keeps_the_recipe_flags_for_a_local_artifact(
    monkeypatch, tmp_path
):
    model = _local_gguf_model("org/MoE-35B-A3B-GGUF", "m.gguf")
    picked = CompatibilityResult(
        model=model,
        gguf_variant=model.gguf_variants[0],
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=60.0,
        placement=Placement("expert_offload", 7, 13, n_cpu_moe=31, num_layers=40),
    )
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "m.gguf", repo_id="org/MoE-35B-A3B-GGUF")

    result, captured = _run_with_llama_cli(
        monkeypatch, [model], [picked], ["run", "--lm-studio-path", str(root)]
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == [
        "/opt/llama/llama-cli",
        "-m",
        str(artifact),
        "-c",
        "4096",
        "-ngl",
        "99",
        "--n-cpu-moe",
        "31",
        "-fa",
        "on",
    ]


def test_run_llama_cli_warns_about_incomplete_explicit_split_set(monkeypatch, tmp_path):
    parts = [f"s-{part:05d}-of-00002.gguf" for part in (1, 2)]
    model = _local_gguf_model("org/Small-2B-GGUF", parts[0])
    root = tmp_path / "lmstudio"
    _write_library_file(root, parts[0], repo_id="org/Small-2B-GGUF")

    result, captured = _run_with_llama_cli(
        monkeypatch,
        [model],
        [],
        ["run", "org/Small-2B-GGUF", "-q", "Q4_K_M", "--lm-studio-path", str(root)],
    )

    assert result.exit_code == 0, result.output
    assert _flat("is missing part 2 of 2; downloading instead") in _flat(result.stdout)
    assert captured["cmd"] == [
        "/opt/llama/llama-cli",
        "-hf",
        "org/Small-2B-GGUF:Q4_K_M",
        "-c",
        "4096",
    ]


def test_gguf_chat_script_uses_local_path_without_download():
    model = _local_gguf_model()
    variant = model.gguf_variants[0]
    local_path = '/tmp/lib/org/Test-7B-GGUF/q4.gguf"); print("injected")'

    script = _generate_chat_script(
        model,
        variant,
        context_length=4096,
        cpu_only=False,
        n_gpu_layers=7,
        local_path=local_path,
    )
    tree = ast.parse(script)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"model_id", "filename", "quant_type", "local_path", "local_parts"}
    }

    assert "huggingface_hub import hf_hub_download" in script
    assert "n_gpu_layers=7" in script
    assert "all(os.path.isfile(part) for part in local_parts)" in script
    assert assignments == {
        "model_id": model.id,
        "filename": variant.filename,
        "quant_type": variant.quant_type,
        "local_path": local_path,
        "local_parts": (local_path,),
    }


def test_run_passes_the_ranked_layer_split_with_a_local_artifact(monkeypatch, tmp_path):
    """The v0.6 layer split and the local reuse have to reach the script together."""
    model = _local_gguf_model("org/MoE-35B-A3B-GGUF", "m.gguf")
    picked = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=60.0,
        placement=Placement("expert_offload", 7, 13, n_cpu_moe=31, num_layers=40),
    )
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "m.gguf", repo_id="org/MoE-35B-A3B-GGUF")
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_generate(model, variant, context_length, cpu_only, **kwargs):
        captured.update(kwargs)
        return "print('ok')"

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("shutil.which", _uv_only)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.index_client.load_index", lambda **kw: None)
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", lambda *a, **kw: [picked])
    monkeypatch.setattr(cli_mod, "_generate_chat_script", fake_generate)
    monkeypatch.setattr("subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", "--lm-studio-path", str(root)])

    assert result.exit_code == 0, result.output
    assert captured["local_path"] == str(artifact)
    assert captured["n_gpu_layers"] == 14
    assert "huggingface-hub" in captured["cmd"]


def test_run_reuses_only_when_the_query_resolves_to_the_gguf_repo(
    monkeypatch, tmp_path
):
    """v0.6 resolves a plain name to the original repo, so only a gguf query reuses."""
    original = _make_model("org/Test-7B", downloads=900, base_model_relation="finetune")
    gguf_repo = _make_model(
        "org/Test-7B-GGUF",
        downloads=100,
        base_model="org/Test-7B",
        base_model_relation="quantized",
        gguf_variants=[
            GGUFVariant(filename="q4.gguf", quant_type="Q4_K_M", file_size_bytes=1)
        ],
    )
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "q4.gguf", repo_id="org/Test-7B-GGUF")
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [original, gguf_repo])

    plain = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert plain.exit_code == 0, plain.output
    assert "Using local LM Studio file" not in plain.stdout
    assert "Warning" not in plain.stdout
    assert "llama-cpp-python" not in captured["cmd"]

    with_format = CliRunner().invoke(
        app, ["run", "Test-7B gguf", "--lm-studio-path", str(root)]
    )

    assert with_format.exit_code == 0, with_format.output
    assert _flat(f"Using local LM Studio file: {artifact}") in _flat(with_format.stdout)
    assert f"local_path = {str(artifact)!r}" in captured["script"]


def test_run_warns_when_the_entry_part_is_missing(monkeypatch, tmp_path):
    """A library that kept only the later parts cannot load either."""
    parts = [f"q4-{part:05d}-of-00002.gguf" for part in (1, 2)]
    model = _local_gguf_model(filename=parts[0])
    root = tmp_path / "lmstudio"
    _write_library_file(root, parts[1])
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0
    assert _flat("is missing part 1 of 2; downloading instead") in _flat(result.stdout)
    assert "Using local LM Studio file" not in result.stdout
    assert f"local_path = {None!r}" not in captured["script"]


def test_run_survives_a_library_path_with_markup_characters(monkeypatch, tmp_path):
    model = _local_gguf_model()
    root = tmp_path / "li[b]rary"
    artifact = _write_library_file(root, "q4.gguf")
    captured: dict[str, object] = {}
    _capture_run(monkeypatch, captured)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "Test-7B", "--lm-studio-path", str(root)])

    assert result.exit_code == 0, result.output
    assert _flat(f"Using local LM Studio file: {artifact}") in _flat(result.stdout)


def test_run_prints_the_local_file_in_the_offload_hint(monkeypatch, tmp_path):
    model = _local_gguf_model("org/MoE-35B-A3B-GGUF", "m.gguf")
    picked = CompatibilityResult(
        model=model,
        gguf_variant=model.gguf_variants[0],
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=60.0,
        placement=Placement("expert_offload", 7, 13, n_cpu_moe=31, num_layers=40),
    )
    root = tmp_path / "lmstudio"
    artifact = _write_library_file(root, "m.gguf", repo_id="org/MoE-35B-A3B-GGUF")

    uv_result, _uv_captured = _run_with_llama_cli_uv_only(
        monkeypatch,
        [model],
        [picked],
        ["run", "--lm-studio-path", str(root)],
    )

    flat_stdout = _flat(uv_result.stdout)
    assert uv_result.exit_code == 0, uv_result.output
    assert _flat("serve it with llama.cpp") in flat_stdout
    assert f"llama-server-m{artifact}" in flat_stdout
    assert "-hf" not in flat_stdout


def _run_with_llama_cli_uv_only(monkeypatch, models, rank_results, argv):
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("shutil.which", _uv_only)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: models)
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models", lambda *a, **kw: rank_results
    )
    monkeypatch.setattr(
        cli_mod, "_generate_chat_script", lambda *a, **kw: "print('ok')"
    )
    monkeypatch.setattr("subprocess.run", fake_run)
    result = CliRunner().invoke(app, argv)
    return result, captured


def test_generated_script_falls_back_when_a_local_part_disappears(
    tmp_path, monkeypatch
):
    """The isolated build runs first, so the script checks the files again."""
    import sys
    import types

    model = _local_gguf_model()
    variant = model.gguf_variants[0]
    missing = tmp_path / "gone.gguf"
    downloads: list[dict] = []

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.Llama = FakeLlama
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: downloads.append(kwargs) or str(missing)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    def stop_reading(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", stop_reading)

    script = _generate_chat_script(
        model,
        variant,
        context_length=4096,
        cpu_only=False,
        local_path=str(missing),
        local_parts=(str(missing),),
    )

    exec(compile(script, "<generated>", "exec"), {"__name__": "__main__"})

    assert downloads == [{"repo_id": model.id, "filename": variant.filename}]


def test_generated_script_opens_an_existing_local_part_set(tmp_path, monkeypatch):
    import sys
    import types

    model = _local_gguf_model()
    variant = model.gguf_variants[0]
    entry = tmp_path / "q4-00001-of-00002.gguf"
    entry.write_bytes(b"gguf")
    part = tmp_path / "q4-00002-of-00002.gguf"
    part.write_bytes(b"gguf")
    loaded: list[str] = []
    downloads: list[dict] = []

    class FakeLlama:
        def __init__(self, **kwargs):
            loaded.append(kwargs["model_path"])

    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.Llama = FakeLlama
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: downloads.append(kwargs) or ""
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError)
    )

    script = _generate_chat_script(
        model,
        variant,
        context_length=4096,
        cpu_only=False,
        local_path=str(entry),
        local_parts=(str(entry), str(part)),
    )

    exec(compile(script, "<generated>", "exec"), {"__name__": "__main__"})

    assert loaded == [str(entry)]
    assert downloads == []


def test_snippet_no_model_found(monkeypatch):
    monkeypatch.setattr(snippet_mod, "_load_models", lambda refresh: [])
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
    assert entry["local_match"] is False
    assert entry["local_path"] is None
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
        local_path="/models/Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
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
    assert entry["local_match"] is True
    assert entry["local_path"] == "/models/Qwen3-4B-Thinking-2507-Q3_K_M.gguf"


def test_main_vision_profile_requires_vision_cache_and_saves_coverage(monkeypatch):
    import whichllm.models.fetcher as fetcher

    load_calls: list[dict] = []
    saved: list[dict] = []
    fetch_calls: list[dict] = []

    def fake_load_cache(**kwargs):
        load_calls.append(kwargs)
        return None

    def fake_save_cache(models, **kwargs):
        saved.append(kwargs)

    async def fake_fetch_models(*, include_vision, report):
        fetch_calls.append({"include_vision": include_vision})
        report.failed.append("Qwen/Qwen3.8-27B")
        return []

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", fake_load_cache)
    monkeypatch.setattr("whichllm.models.cache.save_cache", fake_save_cache)
    monkeypatch.setattr(fetcher, "fetch_models", fake_fetch_models)
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models", lambda models, hardware, **kw: []
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )

    result = CliRunner().invoke(app, ["--profile", "vision"])

    assert result.exit_code == 0, result.output
    assert load_calls == [{"include_vision": True}]
    assert fetch_calls == [{"include_vision": True}]
    assert saved == [{"include_vision": True, "complete": False}]


def test_main_backfill_does_not_extend_cache_lifetime(monkeypatch):
    import whichllm.models.fetcher as fetcher

    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
    )
    updated: list[list] = []

    async def fake_published_at(model_ids):
        return {"org/Test-7B": "2026-10-01T00:00:00.000Z"}

    def fail_save_cache(*args, **kwargs):
        raise AssertionError("backfill must not save a fresh cache")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr(
        "whichllm.models.cache.load_cache",
        lambda **_kwargs: fetcher.models_to_dicts([model]),
    )
    monkeypatch.setattr("whichllm.models.cache.save_cache", fail_save_cache)
    monkeypatch.setattr(
        "whichllm.models.cache.update_cached_models",
        lambda models: updated.append(models),
    )
    monkeypatch.setattr(fetcher, "fetch_model_published_at", fake_published_at)
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models",
        lambda models, hardware, **kw: [
            CompatibilityResult(
                model=next(m for m in models if m.id == "org/Test-7B"),
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                quality_score=80.0,
            )
        ],
    )
    monkeypatch.setattr(
        "whichllm.models.lmstudio.discover_lmstudio_ggufs", lambda paths: []
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )

    result = CliRunner().invoke(app, ["--min-params", "1"])

    assert result.exit_code == 0, result.output
    assert len(updated) == 1
    assert updated[0][0]["published_at"] == "2026-10-01T00:00:00.000Z"


def test_snippet_auto_pick_uses_ranked_runnable_gguf(monkeypatch):
    popular = ModelInfo(
        id="org/Popular-1B-GGUF",
        family_id="popular-1b",
        name="Popular-1B-GGUF",
        parameter_count=1_000_000_000,
        downloads=9_000_000,
        gguf_variants=[
            GGUFVariant(filename="p.gguf", quant_type="Q4_K_M", file_size_bytes=1)
        ],
    )
    ranked = ModelInfo(
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
        base_model_relation="quantized",
        tags=("base_model:quantized:Qwen/Qwen3.6-27B",),
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

    monkeypatch.setattr(
        snippet_mod, "_load_models", lambda refresh: [popular, ranked, real_gguf]
    )
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(24)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models",
        lambda models, hardware, **kwargs: [
            CompatibilityResult(
                model=ranked,
                gguf_variant=synthetic,
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=90.0,
            )
        ],
    )

    result = CliRunner().invoke(app, ["snippet"])

    assert result.exit_code == 0, result.output
    assert "repo_id='unsloth/Qwen3.6-27B-GGUF'" in result.stdout
    assert "filename='q4.gguf'" in result.stdout
    assert "Popular-1B" not in result.stdout


def test_snippet_auto_pick_exits_when_nothing_runs(monkeypatch):
    monkeypatch.setattr(snippet_mod, "_load_models", lambda refresh: [])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr(
        "whichllm.engine.ranker.rank_models", lambda models, hardware, **kw: []
    )

    result = CliRunner().invoke(app, ["snippet"])

    assert result.exit_code == 1
    assert "No runnable model found" in result.stdout


def test_main_reads_configs_for_top_families_then_reranks(monkeypatch):
    import whichllm.models.shape_fetch as shape_fetch

    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    rank_calls: list[int] = []
    fetched: list[list[str]] = []

    def fake_rank_models(models, hardware, **kwargs):
        rank_calls.append(kwargs["top_n"])
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

    async def fake_fetch_family_shapes(models, family_ids):
        fetched.append(list(family_ids))
        return 0

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(shape_fetch, "fetch_family_shapes", fake_fetch_family_shapes)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )

    result = CliRunner().invoke(app, ["--top", "3", "--min-params", "1"])

    assert result.exit_code == 0, result.output
    assert fetched == [["test-7b"]]
    assert rank_calls == [13, 3]


def _index_catalog_with(model):
    from whichllm.models.index_client import IndexCatalog

    return IndexCatalog(
        models=[model],
        benchmark_scores={model.id: 70.0},
        built_at="2026-10-29T03:17:00Z",
    )


def _main_without_network(monkeypatch, captured):
    import whichllm.models.fetcher as fetcher
    import whichllm.models.shape_fetch as shape_fetch

    def fake_rank_models(models, hardware, **kwargs):
        captured["models"] = [m.id for m in models]
        captured["scores"] = kwargs["benchmark_scores"]
        return []

    async def no_fetch(*args, **kwargs):
        raise AssertionError("the index path must not fetch from the Hub")

    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(shape_fetch, "fetch_family_shapes", no_fetch)
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )
    return fetcher


def test_main_ranks_from_the_index_without_hub_calls(monkeypatch):
    captured: dict = {}
    fetcher = _main_without_network(monkeypatch, captured)
    model = _make_model("org/Indexed-8B")

    async def no_fetch(*args, **kwargs):
        raise AssertionError("the index path must not fetch from the Hub")

    monkeypatch.setattr(fetcher, "fetch_models", no_fetch)
    monkeypatch.setattr("whichllm.models.benchmark.fetch_benchmark_scores", no_fetch)
    monkeypatch.setattr(
        "whichllm.models.index_client.load_index",
        lambda refresh=False: _index_catalog_with(model),
    )

    result = CliRunner().invoke(app, ["--min-params", "1"])

    assert result.exit_code == 0, result.output
    assert captured["models"] == ["org/Indexed-8B"]
    assert captured["scores"] == {"org/Indexed-8B": 70.0}
    assert "whichllm index built 2026-10-29 03:17:00 UTC" in result.output


def test_json_says_which_index_the_scores_came_from(monkeypatch):
    import json as json_mod

    captured: dict = {}
    _main_without_network(monkeypatch, captured)
    catalog = _index_catalog_with(_make_model("org/Indexed-8B"))
    catalog.sources = [{"name": "AA", "version": "v4.3.2", "entries": 3}]
    monkeypatch.setattr(
        "whichllm.models.index_client.load_index", lambda refresh=False: catalog
    )

    result = CliRunner().invoke(app, ["--json", "--min-params", "1"])

    assert result.exit_code == 0, result.output
    data = json_mod.loads(result.stdout)["data"]
    assert data == {
        "source": "index",
        "built_at": "2026-10-29T03:17:00Z",
        "benchmark_sources": catalog.sources,
    }


def test_main_live_flag_skips_the_index(monkeypatch):
    captured: dict = {}
    _main_without_network(monkeypatch, captured)

    def unexpected_index(refresh=False):
        raise AssertionError("--live must not read the index")

    monkeypatch.setattr("whichllm.models.index_client.load_index", unexpected_index)
    monkeypatch.setattr(
        "whichllm.models.cache.load_cache",
        lambda **_kwargs: models_to_dicts_for_cli([_make_model("org/Live-8B")]),
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})

    result = CliRunner().invoke(app, ["--live", "--min-params", "1"])

    assert result.exit_code == 0, result.output
    assert captured["models"] == ["org/Live-8B"]
    assert "fetched live" in result.output


def models_to_dicts_for_cli(models):
    from whichllm.models.fetcher import models_to_dicts

    return models_to_dicts(models)
