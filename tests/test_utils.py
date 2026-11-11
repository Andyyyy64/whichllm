"""Tests for shared utilities."""

from pathlib import Path

import pytest
import typer

from whichllm.utils import _cache_dir, context_length_option, parse_context_length


def test_cache_dir_defaults_to_dot_cache(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    result = _cache_dir()
    assert result == Path.home() / ".cache" / "whichllm"


def test_cache_dir_respects_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = _cache_dir()
    assert result == tmp_path / "whichllm"


def test_cache_dir_falls_back_on_empty_xdg(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    result = _cache_dir()
    assert result == Path.home() / ".cache" / "whichllm"


def test_cache_dir_ignores_relative_xdg(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/path")
    result = _cache_dir()
    assert result == Path.home() / ".cache" / "whichllm"


# --- parse_context_length tests ---


def test_parse_plain_integer():
    assert parse_context_length("4096") == 4096
    assert parse_context_length("131072") == 131072


def test_parse_k_suffix():
    assert parse_context_length("64k") == 64 * 1024
    assert parse_context_length("128K") == 128 * 1024


def test_parse_m_suffix():
    assert parse_context_length("1m") == 1024 * 1024
    assert parse_context_length("2M") == 2 * 1024 * 1024


def test_parse_fractional_suffix():
    assert parse_context_length("1.5k") == int(1.5 * 1024)
    assert parse_context_length("0.5m") == int(0.5 * 1024 * 1024)


def test_parse_whitespace_is_stripped():
    assert parse_context_length("  64k  ") == 64 * 1024


def test_parse_rejects_invalid_string():
    with pytest.raises(ValueError, match="Invalid context length"):
        parse_context_length("abc")


def test_parse_rejects_zero():
    with pytest.raises(ValueError, match="must be positive"):
        parse_context_length("0")


def test_parse_rejects_negative():
    with pytest.raises(ValueError, match="must be positive"):
        parse_context_length("-1")


def test_the_option_parser_passes_the_int_default_through():
    assert context_length_option(4096) == 4096


def test_the_option_parser_parses_shorthand():
    assert context_length_option("64k") == 64 * 1024


def test_the_option_parser_reports_a_bad_value_as_a_usage_error():
    with pytest.raises(typer.BadParameter, match="Invalid context length 'lots'"):
        context_length_option("lots")


@pytest.mark.parametrize(
    "args",
    [
        ["--context-length", "lots"],
        ["plan", "some/model", "-c", "lots"],
        ["run", "-c", "lots"],
        ["upgrade", "RTX 4090", "-c", "lots"],
    ],
)
def test_every_command_refuses_a_bad_context_length_with_the_reason(args):
    from typer.testing import CliRunner

    from whichllm.cli import app

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert "Invalid context length 'lots'" in " ".join(result.output.split())


def test_no_option_uses_an_argument_typer_removed():
    # typer 0.29 removed click_type from Option() and Argument(), and every
    # fresh install broke at import. uv.lock holds an older typer, so only the
    # CI job with the newest dependencies would notice it coming back.
    from pathlib import Path

    import whichllm

    for path in Path(whichllm.__file__).parent.rglob("*.py"):
        assert "click_type=" not in path.read_text(encoding="utf-8"), path.name
