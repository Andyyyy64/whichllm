"""Tests for read-only LM Studio GGUF discovery and matching."""

from io import StringIO

from rich.console import Console

from whichllm.engine.types import CompatibilityResult
from whichllm.models.lmstudio import (
    LocalGGUF,
    attach_local_matches,
    default_lmstudio_paths,
    discover_lmstudio_ggufs,
)
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output.ranking import display_ranking


def _result(
    *,
    repo_id: str = "lmstudio-community/Qwen3-8B-GGUF",
    filename: str = "Qwen3-8B-Q4_K_M.gguf",
    quant_type: str = "Q4_K_M",
) -> CompatibilityResult:
    model = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
    )
    artifact = ModelInfo(
        id=repo_id,
        family_id="qwen3-8b",
        name=repo_id.rsplit("/", 1)[-1],
        parameter_count=8_000_000_000,
    )
    variant = GGUFVariant(
        filename=filename,
        quant_type=quant_type,
        file_size_bytes=5_000_000_000,
    )
    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=6_000_000_000,
        vram_available_bytes=8_000_000_000,
        artifact_model=artifact,
        artifact_variant=variant,
    )


def test_default_paths_include_current_and_legacy_locations(tmp_path):
    assert default_lmstudio_paths(tmp_path) == (
        tmp_path / ".lmstudio" / "models",
        tmp_path / ".cache" / "lm-studio" / "models",
    )


def test_discover_ggufs_extracts_repo_and_quant_and_ignores_non_models(tmp_path):
    repo = tmp_path / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3-8B-GGUF"
    repo.mkdir(parents=True)
    model_path = repo / "Qwen3-8B-Q4_K_M.GGUF"
    model_path.write_bytes(b"")
    (repo / "mmproj-Qwen3-8B-BF16.gguf").write_bytes(b"")
    (repo / "notes.txt").write_text("not a model")

    discovered = discover_lmstudio_ggufs(home=tmp_path)

    assert discovered == [
        LocalGGUF(
            path=model_path,
            repo_id="lmstudio-community/Qwen3-8B-GGUF",
            quant_type="Q4_K_M",
        )
    ]


def test_discover_custom_path_deduplicates_a_default_library(tmp_path):
    root = tmp_path / ".lmstudio" / "models"
    repo = root / "bartowski" / "Llama-3-8B-GGUF"
    repo.mkdir(parents=True)
    model_path = repo / "Llama-3-8B-Q8_0.gguf"
    model_path.write_bytes(b"")

    discovered = discover_lmstudio_ggufs([root], home=tmp_path)

    assert len(discovered) == 1
    assert discovered[0].path == model_path


def test_discover_missing_paths_is_empty(tmp_path):
    assert discover_lmstudio_ggufs([tmp_path / "missing"], home=tmp_path) == []


def test_attach_local_match_uses_resolved_artifact_repo_and_quant(tmp_path):
    result = _result()
    local_path = tmp_path / "different-local-name-Q4_K_M.gguf"
    local_models = [
        LocalGGUF(
            path=local_path,
            repo_id="LMSTUDIO-COMMUNITY/qwen3-8b-gguf",
            quant_type="q4_k_m",
        )
    ]

    attach_local_matches([result], local_models)

    assert result.local_path == str(local_path)


def test_attach_local_match_rejects_wrong_repo_or_quant(tmp_path):
    result = _result()
    local_models = [
        LocalGGUF(
            path=tmp_path / "wrong-repo-Q4_K_M.gguf",
            repo_id="other/Qwen3-8B-GGUF",
            quant_type="Q4_K_M",
        ),
        LocalGGUF(
            path=tmp_path / "right-repo-Q8_0.gguf",
            repo_id="lmstudio-community/Qwen3-8B-GGUF",
            quant_type="Q8_0",
        ),
    ]

    attach_local_matches([result], local_models)

    assert result.local_path is None


def test_attach_local_match_uses_exact_filename_for_flat_custom_path(tmp_path):
    result = _result()
    local_path = tmp_path / "Qwen3-8B-Q4_K_M.gguf"

    attach_local_matches(
        [result],
        [LocalGGUF(path=local_path, repo_id=None, quant_type="Q4_K_M")],
    )

    assert result.local_path == str(local_path)


def test_rich_ranking_marks_local_model_as_installed(tmp_path):
    import whichllm.output._console as console_mod

    result = _result()
    result.local_path = str(tmp_path / result.artifact_variant.filename)
    buffer = StringIO()
    original_console = console_mod.console
    console_mod.console = Console(file=buffer, force_terminal=False, width=120)
    try:
        display_ranking([result], show_status=True)
    finally:
        console_mod.console = original_console

    assert "Installed" in buffer.getvalue()
