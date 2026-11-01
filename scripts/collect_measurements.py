"""Turn verified measurement issues into ``src/whichllm/data/measurements.json``.

Reads the issues labeled ``measurement-verified``, checks every report and
rewrites the file. Review the diff and open a PR by hand: nothing here merges.

Only a maintainer can add that label, but the text in an issue is still untrusted
input. It is parsed as JSON and checked field by field. It is never evaluated,
never passed to a shell, and never reaches a workflow.

Usage:
    uv run --locked python scripts/collect_measurements.py
    uv run --locked python scripts/collect_measurements.py --issues issues.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from whichllm.bench.report import report_problems

REPO = "Andyyyy64/whichllm"
LABEL = "measurement-verified"
SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "whichllm"
    / "data"
    / "measurements.json"
)

_REPORT_BLOCK_RE = re.compile(r"^```json\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)
_MAX_BODY_CHARS = 20_000
_MAX_TEXT_CHARS = 200
_PLACEMENTS = {"gpu", "expert_offload", "layer_offload", "cpu", None}
# A run this far from the estimate is a broken report, not a data point.
_MIN_RATIO = 0.1
_MAX_RATIO = 10.0


class Rejected(Exception):
    """Why an issue's report stays out of the file."""


def extract_report(body: str) -> object:
    """The JSON in the form's Report field."""
    if len(body) > _MAX_BODY_CHARS:
        raise Rejected("the issue is too long")
    match = _REPORT_BLOCK_RE.search(body)
    if match is None:
        raise Rejected("no fenced JSON block")
    try:
        return json.loads(match.group(1))
    except ValueError as error:
        raise Rejected(f"the block is not JSON: {error}") from error


def _text(value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_TEXT_CHARS
        or not value.isprintable()
    ):
        raise Rejected(f"bad text field: {value!r}")
    return value


def _positive_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rejected(f"bad number: {value!r}")
    return value


def _speed(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise Rejected(f"bad speed: {value!r}")
    return float(value)


def to_measurement(number: int, report: object) -> dict:
    """One row of measurements.json, copied field by field from a checked report."""
    problems = report_problems(report)
    if problems:
        raise Rejected("; ".join(sorted(problems)))

    gpus = report["hardware"]["gpus"]
    placement = (report["placement"] or {}).get("kind")
    if placement not in _PLACEMENTS:
        raise Rejected(f"bad placement: {placement!r}")
    tok_per_sec = _speed(report["measurement"]["median_tok_per_sec"])
    estimate = _speed((report["estimate"] or {}).get("tok_per_sec"))
    if estimate is not None and not _MIN_RATIO <= tok_per_sec / estimate <= _MAX_RATIO:
        raise Rejected(f"{tok_per_sec} tok/s against an estimate of {estimate}")

    return {
        "issue": number,
        "gpu": _text(gpus[0].get("name")) if gpus else None,
        "gpu_count": len(gpus),
        "cpu": _text(report["hardware"]["cpu"]),
        "repo": _text(report["model"]["repo"]),
        "quant": _text(report["model"]["quant"], optional=True),
        "file_size_bytes": _positive_int(report["model"]["file_size_bytes"]),
        "context_length": _positive_int(report["model"]["context_length"]),
        "placement": placement,
        "engine": _text(report["runtime"]["engine"]),
        "app": _text(report["runtime"]["app"], optional=True),
        "method": report["method"],
        "tok_per_sec": tok_per_sec,
        "estimate_tok_per_sec": estimate,
        "whichllm_version": _text(report["whichllm_version"]),
    }


def collect(issues: list[dict]) -> tuple[list[dict], list[str]]:
    """Rows sorted by issue, and one line for each issue that stayed out."""
    rows: list[dict] = []
    skipped: list[str] = []
    for issue in sorted(issues, key=lambda item: item["number"]):
        number = issue["number"]
        try:
            rows.append(to_measurement(number, extract_report(issue.get("body") or "")))
        except Rejected as error:
            skipped.append(f"#{number}: {error}")
    return rows, skipped


def fetch_issues() -> list[dict]:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            REPO,
            "--label",
            LABEL,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,body",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def write_snapshot(rows: list[dict]) -> None:
    snapshot = {"schema_version": 1, "measurements": rows}
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--issues", type=Path, help="Read issues from this JSON file instead of gh"
    )
    args = parser.parse_args(argv)

    if args.issues is not None:
        issues = json.loads(args.issues.read_text(encoding="utf-8"))
    else:
        issues = fetch_issues()
    rows, skipped = collect(issues)
    write_snapshot(rows)
    print(f"{len(rows)} measurements from {len(issues)} issues")
    for line in skipped:
        print(f"skipped {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
