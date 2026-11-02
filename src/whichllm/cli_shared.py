"""Shared CLI primitives."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

from whichllm.utils import _current_version

console = Console()


def _tolerate_narrow_encodings() -> None:
    """Print what a non-UTF-8 terminal or file cannot encode as "?".

    Redirected output on Windows uses the console code page, such as cp932 or
    cp1252, which has no box-drawing or check-mark characters, so
    `whichllm plan "qwen3 32b" > plan.txt` stopped with a UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


def _format_fetch_error(error: Exception) -> str:
    """Return a useful one-line fetch error even when str(error) is empty."""
    detail = str(error).strip()
    if detail:
        return detail

    response = getattr(error, "response", None)
    request = getattr(error, "request", None) or getattr(response, "request", None)
    status_code = getattr(response, "status_code", None)
    url = getattr(request, "url", None)
    if status_code and url:
        return f"{type(error).__name__}: HTTP {status_code} for {url}"
    if url:
        return f"{type(error).__name__} while requesting {url}"
    return f"{type(error).__name__} with no detail from the network layer"


def _print_version(value: bool) -> None:
    """Print version and exit when --version is requested."""
    if value:
        console.print(_current_version())
        raise typer.Exit()


__all__ = [
    "_run_async",
    "_format_fetch_error",
    "_print_version",
    "_tolerate_narrow_encodings",
    "console",
]
