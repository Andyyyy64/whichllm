"""Tests for read-only LM Studio GGUF discovery and matching."""

from io import StringIO

import pytest
from rich.console import Console

from whichllm.engine.types import CompatibilityResult
from whichllm.models import lmstudio
from whichllm.models.lmstudio import (
    LocalGGUF,
    LMStudioPathError,
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


def _add_model_file(
    home,
    filename: str,
    *,
    repo_id: str = "lmstudio-community/Qwen3-8B-GGUF",
    root=None,
):
    root = root or home / ".lmstudio" / "models"
    owner, repo = repo_id.split("/", 1)
    path = root / owner / repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gguf")
    return path


def test_default_paths_include_only_current_location(tmp_path):
    assert default_lmstudio_paths(tmp_path) == (tmp_path / ".lmstudio" / "models",)


def test_discover_ggufs_records_repo_artifact_and_ignores_non_models(tmp_path):
    model_path = _add_model_file(tmp_path, "Qwen3-8B-Q4_K_M.GGUF")
    repo = model_path.parent
    (repo / "mmproj-Qwen3-8B-BF16.gguf").write_bytes(b"")
    (repo / "notes.txt").write_text("not a model")

    discovered = discover_lmstudio_ggufs(home=tmp_path)

    assert discovered == [
        LocalGGUF(
            path=model_path,
            repo_id="lmstudio-community/Qwen3-8B-GGUF",
            artifact_path="Qwen3-8B-Q4_K_M.GGUF",
        )
    ]


def test_legacy_location_requires_explicit_path(tmp_path):
    legacy = tmp_path / ".cache" / "lm-studio" / "models"
    model_path = _add_model_file(
        tmp_path,
        "Llama-3-8B-Q8_0.gguf",
        repo_id="bartowski/Llama-3-8B-GGUF",
        root=legacy,
    )

    assert discover_lmstudio_ggufs(home=tmp_path) == []
    assert discover_lmstudio_ggufs([legacy], home=tmp_path) == [
        LocalGGUF(
            path=model_path,
            repo_id="bartowski/Llama-3-8B-GGUF",
            artifact_path="Llama-3-8B-Q8_0.gguf",
        )
    ]


def test_discover_custom_path_deduplicates_default_library(tmp_path):
    root = tmp_path / ".lmstudio" / "models"
    model_path = _add_model_file(
        tmp_path,
        "Llama-3-8B-Q8_0.gguf",
        repo_id="bartowski/Llama-3-8B-GGUF",
    )

    discovered = discover_lmstudio_ggufs([root], home=tmp_path)

    assert len(discovered) == 1
    assert discovered[0].path == model_path


def test_discover_missing_explicit_path_raises(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(LMStudioPathError, match="path does not exist"):
        discover_lmstudio_ggufs([missing], home=tmp_path)


def test_discover_explicit_file_path_raises(tmp_path):
    not_a_library = tmp_path / "model.gguf"
    not_a_library.write_bytes(b"gguf")

    with pytest.raises(LMStudioPathError, match="path is not a directory"):
        discover_lmstudio_ggufs([not_a_library], home=tmp_path)


def test_discover_reports_walk_error_for_explicit_path(monkeypatch, tmp_path):
    library = tmp_path / "library"
    library.mkdir()

    def denied_walk(root, *, topdown, onerror, followlinks):
        onerror(PermissionError(13, "Permission denied", str(root)))
        return []

    monkeypatch.setattr(lmstudio.os, "walk", denied_walk)

    with pytest.raises(LMStudioPathError, match="cannot be read"):
        discover_lmstudio_ggufs([library], home=tmp_path)


def test_discover_ignores_broken_file_symlink(tmp_path):
    repo = tmp_path / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3-8B-GGUF"
    repo.mkdir(parents=True)
    (repo / "Qwen3-8B-Q4_K_M.gguf").symlink_to(repo / "missing.gguf")

    assert discover_lmstudio_ggufs(home=tmp_path) == []


def test_discover_accepts_resolvable_file_symlink(tmp_path):
    target = tmp_path / "downloaded.gguf"
    target.write_bytes(b"gguf")
    link = (
        tmp_path
        / ".lmstudio"
        / "models"
        / "lmstudio-community"
        / "Qwen3-8B-GGUF"
        / "Qwen3-8B-Q4_K_M.gguf"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    discovered = discover_lmstudio_ggufs(home=tmp_path)

    assert len(discovered) == 1
    assert discovered[0].path == link


def test_attach_local_match_requires_resolved_repo_and_exact_artifact(tmp_path):
    result = _result()
    local_path = _add_model_file(tmp_path, result.artifact_variant.filename)
    local_models = discover_lmstudio_ggufs(home=tmp_path)

    attach_local_matches([result], local_models)

    assert result.local_path == str(local_path)


def test_attach_local_match_rejects_same_repo_and_quant_with_different_filename(
    tmp_path,
):
    result = _result()
    _add_model_file(tmp_path, "different-name-Q4_K_M.gguf")

    attach_local_matches([result], discover_lmstudio_ggufs(home=tmp_path))

    assert result.local_path is None


def test_attach_local_match_rejects_exact_filename_in_wrong_repo(tmp_path):
    result = _result()
    _add_model_file(
        tmp_path,
        result.artifact_variant.filename,
        repo_id="other/Qwen3-8B-GGUF",
    )

    attach_local_matches([result], discover_lmstudio_ggufs(home=tmp_path))

    assert result.local_path is None


def test_attach_local_match_rejects_filename_without_repo(tmp_path):
    result = _result()
    custom = tmp_path / "flat-library"
    custom.mkdir()
    local_path = custom / result.artifact_variant.filename
    local_path.write_bytes(b"gguf")

    attach_local_matches(
        [result],
        discover_lmstudio_ggufs([custom], home=tmp_path),
    )

    assert result.local_path is None


def test_attach_local_match_requires_resolved_artifact(tmp_path):
    result = _result()
    _add_model_file(tmp_path, result.artifact_variant.filename)
    result.artifact_model = None
    result.artifact_variant = None

    attach_local_matches([result], discover_lmstudio_ggufs(home=tmp_path))

    assert result.local_path is None


def test_attach_local_match_accepts_complete_split_artifact(tmp_path):
    filename = "weights/model-Q4_K_M-00002-of-00002.gguf"
    result = _result(filename=filename)
    _add_model_file(tmp_path, "weights/model-Q4_K_M-00001-of-00002.gguf")
    matched_path = _add_model_file(tmp_path, filename)

    attach_local_matches([result], discover_lmstudio_ggufs(home=tmp_path))

    assert result.local_path == str(matched_path)


def test_attach_local_match_rejects_incomplete_split_artifact(tmp_path):
    filename = "model-Q4_K_M-00002-of-00002.gguf"
    result = _result(filename=filename)
    _add_model_file(tmp_path, filename)

    attach_local_matches([result], discover_lmstudio_ggufs(home=tmp_path))

    assert result.local_path is None


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
