"""`whichllm report`: nothing leaves without the person reading and confirming it."""

from __future__ import annotations

import getpass
import json
import re
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from tests.test_bench import _hardware, _report
from whichllm.bench.report import (
    ISSUE_URL,
    MAX_ISSUE_URL_LENGTH,
    ReportError,
    build_manual_report,
    issue_body,
    issue_title,
    issue_url,
    latest_report_path,
    load_report,
    report_problems,
    save_report,
)
from whichllm.cli import app
from whichllm.commands import report as report_module

_TEMPLATE = Path(__file__).parent.parent / ".github/ISSUE_TEMPLATE/measurement.yml"


def _edited(**changes) -> dict:
    report = _report()
    report.update(changes)
    return report


def test_a_fresh_report_has_no_problems():
    assert report_problems(_report()) == []


def test_a_manual_report_has_no_problems():
    report = build_manual_report(
        version="0.6.0",
        hardware=_hardware(),
        engine="mlx",
        model="mlx-community/Qwen3.5-9B-4bit",
        quant=None,
        context_length=4096,
        tok_per_sec=42.0,
    )

    assert report_problems(report) == []
    assert report["method"] == "manual"
    assert report["estimate"] is None


@pytest.mark.parametrize(
    "report",
    [
        "not a report",
        _edited(hostname="my-desktop"),
        _edited(schema_version=2),
        _edited(method="guessed"),
        _edited(hardware="a PC"),
        _edited(measurement=None),
    ],
    ids=["text", "extra-field", "schema", "method", "section-type", "no-measurement"],
)
def test_a_broken_report_is_refused(report):
    assert report_problems(report)


@pytest.mark.parametrize("section", ["hardware", "runtime", "model", "measurement"])
def test_a_field_added_to_a_section_is_refused(section):
    report = _report()
    report[section]["username"] = "someone"

    assert any("username" in problem for problem in report_problems(report))


def test_a_field_added_to_a_gpu_is_refused():
    report = _report()
    report["hardware"]["gpus"][0]["serial"] = "0420"

    assert any("serial" in problem for problem in report_problems(report))


@pytest.mark.parametrize("median", [0, -1, 1e9, "fast", True])
def test_an_impossible_speed_is_refused(median):
    report = _report()
    report["measurement"]["median_tok_per_sec"] = median

    assert report_problems(report)


def test_load_report_refuses_what_is_not_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ReportError, match="not JSON"):
        load_report(path)


def test_latest_report_is_the_newest_name(tmp_path):
    (tmp_path / "20261101-100000.json").write_text("{}", encoding="utf-8")
    newest = tmp_path / "20261101-120000.json"
    newest.write_text("{}", encoding="utf-8")
    (tmp_path / "20261101-110000.json").write_text("{}", encoding="utf-8")

    assert latest_report_path(tmp_path) == newest
    assert latest_report_path(tmp_path / "missing") is None


def test_the_issue_link_prefills_the_template():
    report = _report()

    url = urlparse(issue_url(report))
    query = parse_qs(url.query)

    assert f"{url.scheme}://{url.netloc}{url.path}" == ISSUE_URL
    assert query["template"] == ["measurement.yml"]
    assert query["title"] == [issue_title(report)]
    assert json.loads(query["report"][0]) == report


def test_the_issue_title_names_the_device_and_the_speed():
    title = issue_title(_report())

    assert title == (
        "[Measurement] NVIDIA GeForce RTX 5080, Model-GGUF Q4_K_M: 90.0 tok/s"
    )


def test_the_issue_title_falls_back_to_the_cpu():
    report = _report()
    report["hardware"]["gpus"] = []

    assert "Intel Core Ultra 9 285K" in issue_title(report)


def test_the_body_holds_nothing_that_identifies_the_machine():
    report = _report()
    body = issue_title(report) + issue_body(report)

    for secret in (getpass.getuser(), socket.gethostname(), str(Path.home())):
        assert not re.search(rf"\b{re.escape(secret)}\b", body, re.IGNORECASE)


def test_the_form_has_the_field_the_link_fills():
    template = _TEMPLATE.read_text(encoding="utf-8")

    assert "id: report" in template
    assert 'labels: ["measurement"]' in template
    assert "public" in template


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_open(url):
        calls.append(url)
        return True

    monkeypatch.setattr(report_module.webbrowser, "open", fake_open)
    return calls


@pytest.fixture
def saved(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return save_report(_report())


def test_no_report_says_how_to_get_one(tmp_path, monkeypatch, opened):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["report"])

    assert result.exit_code == 1
    assert "whichllm bench" in result.output
    assert opened == []


def test_saying_no_opens_nothing(saved, opened):
    result = CliRunner().invoke(app, ["report"], input="n\n")

    assert result.exit_code == 0
    assert opened == []
    assert "Nothing was opened" in result.output


def test_the_default_answer_opens_nothing(saved, opened):
    result = CliRunner().invoke(app, ["report"], input="\n")

    assert result.exit_code == 0
    assert opened == []


def test_the_whole_report_shows_before_the_question(saved, opened):
    result = CliRunner().invoke(app, ["report"], input="n\n")

    assert result.output.index('"median_tok_per_sec"') < result.output.index(
        "Open it in your browser?"
    )
    assert "sends nothing itself" in result.output


def test_saying_yes_opens_the_issue_and_shows_the_link(saved, opened):
    result = CliRunner().invoke(app, ["report"], input="y\n")

    assert result.exit_code == 0
    assert len(opened) == 1
    assert opened[0].startswith(ISSUE_URL)
    assert "template=measurement.yml" in result.output


def test_print_asks_nothing_and_opens_nothing(saved, opened):
    result = CliRunner().invoke(app, ["report", "--print"])

    assert result.exit_code == 0
    assert opened == []
    assert '"median_tok_per_sec"' in result.output
    assert "Open it" not in result.output


def test_an_edited_report_with_extra_data_is_refused(saved, opened):
    report = json.loads(saved.read_text(encoding="utf-8"))
    report["hostname"] = "my-desktop"
    saved.write_text(json.dumps(report), encoding="utf-8")

    result = CliRunner().invoke(app, ["report", "--file", str(saved)], input="y\n")

    assert result.exit_code == 1
    assert "hostname" in result.output
    assert opened == []


def test_a_report_too_long_for_a_link_is_pasted(saved, opened, monkeypatch):
    monkeypatch.setattr(report_module, "MAX_ISSUE_URL_LENGTH", 100)

    result = CliRunner().invoke(app, ["report"], input="y\n")

    assert "paste the text above" in result.output
    assert "report=" not in opened[0]
    assert len(opened[0]) < MAX_ISSUE_URL_LENGTH


def test_no_browser_still_prints_the_link(saved, monkeypatch):
    monkeypatch.setattr(report_module.webbrowser, "open", lambda url: False)

    result = CliRunner().invoke(app, ["report"], input="y\n")

    assert "No browser opened" in result.output
    assert "template=measurement.yml" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["report", "--manual"],
        ["report", "--manual", "--engine", "mlx"],
        ["report", "--manual", "--engine", "mlx", "--model", "m"],
    ],
)
def test_manual_needs_all_three_values(args, opened):
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert opened == []


def test_manual_refuses_an_impossible_speed(monkeypatch, opened):
    monkeypatch.setattr(report_module, "_detected_hardware", lambda _: _hardware())

    result = CliRunner().invoke(
        app,
        [
            "report",
            "--manual",
            "--engine",
            "mlx",
            "--model",
            "m",
            "--tok-per-sec",
            "-5",
        ],
    )

    assert result.exit_code == 2
    assert opened == []


def test_manual_reports_the_speed_given(monkeypatch, opened):
    monkeypatch.setattr(report_module, "_detected_hardware", lambda _: _hardware())

    result = CliRunner().invoke(
        app,
        [
            "report",
            "--manual",
            "--engine",
            "mlx",
            "--model",
            "mlx-community/Qwen3.5-9B-4bit",
            "--tok-per-sec",
            "42",
        ],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    report = json.loads(parse_qs(urlparse(opened[0]).query)["report"][0])
    assert report["method"] == "manual"
    assert report["runtime"]["engine"] == "mlx"
    assert report["measurement"]["median_tok_per_sec"] == 42
