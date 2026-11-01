"""Verified measurements: from an untrusted issue to a confidence level."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from tests.test_bench import _report
from whichllm.engine import measurements
from whichllm.engine.measurements import counts_as_evidence, gpu_evidence
from whichllm.engine.performance import estimate_speed_uncertainty
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3


def _load_script():
    path = Path(__file__).parent.parent / "scripts" / "collect_measurements.py"
    spec = importlib.util.spec_from_file_location("collect_measurements", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_measurements = _load_script()


def _issue(number: int, report: object) -> dict:
    block = report if isinstance(report, str) else json.dumps(report, indent=2)
    body = f"### Report\n\n```json\n{block}\n```\n\n### Notes\n\n_No response_\n"
    return {"number": number, "body": body}


def _rows(*issues: dict) -> list[dict]:
    rows, skipped = collect_measurements.collect(list(issues))
    assert skipped == []
    return rows


def test_the_shipped_file_is_valid():
    data = json.loads(collect_measurements.SNAPSHOT_PATH.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    for row in data["measurements"]:
        assert isinstance(counts_as_evidence(row), bool)


def test_a_report_becomes_one_row():
    row = _rows(_issue(301, _report()))[0]

    assert row["issue"] == 301
    assert row["gpu"] == "NVIDIA GeForce RTX 5080"
    assert row["gpu_count"] == 1
    assert row["tok_per_sec"] == 90.0
    assert row["estimate_tok_per_sec"] == 80.0
    assert row["placement"] == "gpu"
    assert counts_as_evidence(row)


def test_rows_are_sorted_by_issue():
    rows = _rows(_issue(302, _report()), _issue(301, _report()))

    assert [row["issue"] for row in rows] == [301, 302]


def _runtime_report(engine: str = "llama.cpp", app: str | None = None) -> dict:
    report = _report()
    report["runtime"] = {"engine": engine, "app": app, "version": None}
    return report


@pytest.mark.parametrize(
    ("change", "counts"),
    [
        (lambda r: None, True),
        (lambda r: r["runtime"].update(app="Ollama"), False),
        (lambda r: r["runtime"].update(app="LM Studio"), False),
        (lambda r: r["runtime"].update(engine="vllm"), False),
        (lambda r: r.update(method="manual"), False),
        (lambda r: r["placement"].update(kind="layer_offload"), False),
        (lambda r: r["hardware"]["gpus"].append(r["hardware"]["gpus"][0]), False),
        (lambda r: r.update(estimate=None), False),
    ],
    ids=[
        "plain",
        "ollama",
        "lm-studio",
        "vllm",
        "manual",
        "split",
        "two-gpus",
        "no-estimate",
    ],
)
def test_only_plain_llama_cpp_counts_as_evidence(change, counts):
    report = _report()
    change(report)

    assert counts_as_evidence(_rows(_issue(1, report))[0]) is counts


@pytest.mark.parametrize(
    "body",
    [
        "",
        "no block at all",
        "```json\nnot json\n```",
        "```json\n[1, 2]\n```",
    ],
    ids=["empty", "no-block", "not-json", "list"],
)
def test_a_bad_body_stays_out(body):
    rows, skipped = collect_measurements.collect([{"number": 7, "body": body}])

    assert rows == []
    assert skipped[0].startswith("#7:")


def test_a_report_inside_an_oversized_issue_stays_out():
    issue = _issue(8, _report())
    issue["body"] += "x" * 30_000

    rows, skipped = collect_measurements.collect([issue])

    assert rows == []
    assert "too long" in skipped[0]


def _report_with(section: str, field: str, value: object) -> dict:
    report = _report()
    report[section][field] = value
    return report


@pytest.mark.parametrize(
    "report",
    [
        _report_with("hardware", "hostname", "my-desktop"),
        _report_with("runtime", "engine", "x" * 500),
        _report_with("runtime", "engine", "llama.cpp\x1b[2J"),
        _report_with("runtime", "engine", "$(touch pwned)\n"),
        _report_with("model", "repo", ""),
        _report_with("model", "context_length", -1),
        _report_with("model", "file_size_bytes", "huge"),
        _report_with("measurement", "median_tok_per_sec", 1e9),
        _report_with("measurement", "median_tok_per_sec", 900.0),
        _report_with("placement", "kind", "quantum"),
    ],
    ids=[
        "extra-field",
        "long-text",
        "escape-codes",
        "newline",
        "empty-repo",
        "negative",
        "not-a-number",
        "impossible",
        "ten-times-the-estimate",
        "unknown-placement",
    ],
)
def test_a_hostile_report_stays_out(report):
    rows, skipped = collect_measurements.collect([_issue(9, report)])

    assert rows == []
    assert len(skipped) == 1


def test_one_bad_issue_does_not_stop_the_others():
    rows, skipped = collect_measurements.collect(
        [_issue(1, _report()), _issue(2, "not json"), _issue(3, _report())]
    )

    assert [row["issue"] for row in rows] == [1, 3]
    assert len(skipped) == 1


def test_only_verified_issues_are_asked_for(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["shell"] = kwargs.get("shell", False)
        return subprocess.CompletedProcess(command, 0, "[]", "")

    monkeypatch.setattr(collect_measurements.subprocess, "run", fake_run)

    assert collect_measurements.fetch_issues() == []
    assert "measurement-verified" in captured["command"]
    assert captured["shell"] is False


def test_main_writes_the_snapshot(tmp_path, monkeypatch, capsys):
    source = tmp_path / "issues.json"
    source.write_text(json.dumps([_issue(5, _report())]), encoding="utf-8")
    target = tmp_path / "measurements.json"
    monkeypatch.setattr(collect_measurements, "SNAPSHOT_PATH", target)

    assert collect_measurements.main(["--issues", str(source)]) == 0

    data = json.loads(target.read_text(encoding="utf-8"))
    assert [row["issue"] for row in data["measurements"]] == [5]
    assert "1 measurements from 1 issues" in capsys.readouterr().out


def _row(
    gpu: str = "NVIDIA GeForce RTX 3060 Mobile",
    tok_per_sec: float = 50.0,
    **changes,
) -> dict:
    row = {
        "gpu": gpu,
        "gpu_count": 1,
        "engine": "llama.cpp",
        "app": None,
        "method": "auto",
        "placement": "gpu",
        "tok_per_sec": tok_per_sec,
        "estimate_tok_per_sec": 50.0,
    }
    row.update(changes)
    return row


def test_two_reports_say_nothing_and_three_do(monkeypatch):
    monkeypatch.setattr(measurements, "_MEASUREMENTS", [_row(), _row()])
    assert gpu_evidence("NVIDIA GeForce RTX 3060 Mobile") is None

    monkeypatch.setattr(measurements, "_MEASUREMENTS", [_row(), _row(), _row()])
    assert gpu_evidence("NVIDIA GeForce RTX 3060 Mobile").reports == 3


def test_reports_match_across_spellings(monkeypatch):
    rows = [_row(), _row(), _row()]
    monkeypatch.setattr(measurements, "_MEASUREMENTS", rows)

    for name in (
        "GeForce RTX 3060 Mobile (simulated)",
        "nvidia geforce rtx 3060 mobile",
    ):
        assert gpu_evidence(name).reports == 3
    assert gpu_evidence("NVIDIA GeForce RTX 3060") is None


def test_reference_values_never_count(monkeypatch):
    rows = [_row(app="Ollama") for _ in range(5)] + [
        _row(method="manual") for _ in range(5)
    ]
    monkeypatch.setattr(measurements, "_MEASUREMENTS", rows)

    assert gpu_evidence("NVIDIA GeForce RTX 3060 Mobile") is None


def _confidence(gpu_name: str) -> tuple[str, list[str]]:
    variant = GGUFVariant("m-Q4_K_M.gguf", "Q4_K_M", 4 * _GiB)
    model = ModelInfo(
        id="org/M-GGUF",
        family_id="m",
        name="M",
        parameter_count=int(7e9),
        gguf_variants=[variant],
    )
    gpu = GPUInfo(
        name=gpu_name,
        vendor="nvidia",
        vram_bytes=8 * _GiB,
        compute_capability=(8, 6),
        memory_bandwidth_gbps=336.0,
    )
    confidence, _, notes = estimate_speed_uncertainty(
        model, variant, gpu, "full_gpu", 50.0
    )
    return confidence, notes


def test_verified_reports_lift_a_laptop_gpu_to_high(monkeypatch):
    name = "NVIDIA GeForce RTX 3060 Mobile"
    assert _confidence(name)[0] == "low"

    rows = [_row(gpu=name, tok_per_sec=speed) for speed in (48.0, 50.0, 53.0)]
    monkeypatch.setattr(measurements, "_MEASUREMENTS", rows)

    confidence, notes = _confidence(name)
    assert confidence == "high"
    assert any("3 verified reports" in note for note in notes)


def test_reports_far_from_the_estimate_lift_nothing(monkeypatch):
    name = "NVIDIA GeForce RTX 3060 Mobile"
    rows = [_row(gpu=name, tok_per_sec=20.0) for _ in range(4)]
    monkeypatch.setattr(measurements, "_MEASUREMENTS", rows)

    assert _confidence(name)[0] == "low"
