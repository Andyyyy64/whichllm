"""Tests for shared utilities."""

from pathlib import Path

import pytest

from whichllm.utils import _cache_dir, parse_context_length, CONTEXT_LENGTH


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


def test_click_type_passes_int_through():
    assert CONTEXT_LENGTH.convert(4096, None, None) == 4096


def test_click_type_parses_shorthand():
    assert CONTEXT_LENGTH.convert("64k", None, None) == 64 * 1024
