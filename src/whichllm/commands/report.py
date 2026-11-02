"""`whichllm report`: share a measurement through a GitHub issue you submit."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import typer
from rich.markup import escape

from whichllm.bench.report import (
    MAX_ISSUE_URL_LENGTH,
    ReportError,
    build_manual_report,
    issue_body,
    issue_title,
    issue_url,
    latest_report_path,
    load_report,
    report_problems,
)
from whichllm.cli_shared import console
from whichllm.commands.bench import _detected_hardware
from whichllm.utils import _current_version


def _manual_report(
    engine: str | None,
    served_model: str | None,
    tok_per_sec: float | None,
    quant: str | None,
    context_length: int,
) -> dict:
    if not (engine and served_model and tok_per_sec):
        console.print(
            "[red]Error:[/] --manual needs --engine, --model and --tok-per-sec"
        )
        raise typer.Exit(code=2)
    report = build_manual_report(
        version=_current_version(),
        hardware=_detected_hardware(False),
        engine=engine,
        model=served_model,
        quant=quant,
        context_length=context_length,
        tok_per_sec=tok_per_sec,
    )
    problems = report_problems(report)
    if problems:
        console.print(f"[red]Error:[/] {escape('; '.join(problems))}")
        raise typer.Exit(code=2)
    return report


def _saved_report(file: Path | None) -> dict:
    path = file if file is not None else latest_report_path()
    if path is None:
        console.print(
            "[red]Error:[/] No saved report. Run `whichllm bench` first, or use "
            "--manual."
        )
        raise typer.Exit(code=1)
    try:
        return load_report(path)
    except ReportError as error:
        console.print(f"[red]Error:[/] {escape(str(error))}")
        raise typer.Exit(code=1) from error


def _show(report: dict) -> None:
    console.print(f"\n[bold]{escape(issue_title(report))}[/]\n")
    console.print(issue_body(report), markup=False, highlight=False)


def _open_issue(report: dict) -> None:
    url = issue_url(report)
    if len(url) > MAX_ISSUE_URL_LENGTH:
        short = issue_url(report, include_body=False)
        webbrowser.open(short)
        console.print(
            "\nThis report is too long for a link, so paste the text above into "
            "the form:"
        )
        console.print(short, markup=False, highlight=False, soft_wrap=True)
        return
    if not webbrowser.open(url):
        console.print("\nNo browser opened. Open this link to continue:")
    else:
        console.print("\nThe issue is open in your browser. Submit it there:")
    console.print(url, markup=False, highlight=False, soft_wrap=True)


def report_command(
    *,
    file: Path | None,
    print_only: bool,
    manual: bool,
    engine: str | None,
    served_model: str | None,
    tok_per_sec: float | None,
    quant: str | None,
    context_length: int,
) -> None:
    """Show a report, then open it as a prefilled issue after a yes."""
    if manual:
        report = _manual_report(
            engine, served_model, tok_per_sec, quant, context_length
        )
    else:
        report = _saved_report(file)

    _show(report)
    if print_only:
        return
    console.print(
        "\n[dim]whichllm sends nothing itself. This opens a prefilled GitHub issue, "
        "and it is public once you submit it there.[/]"
    )
    if not typer.confirm("Open it in your browser?", default=False):
        console.print("Nothing was opened.")
        return
    _open_issue(report)
