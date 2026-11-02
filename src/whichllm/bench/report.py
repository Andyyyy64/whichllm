"""The saved result of a benchmark, and how it reaches a GitHub issue.

A report holds what the measurement needs to be useful to someone else, and
nothing that identifies the machine's owner: no username, hostname, file path,
serial number, IP address or generated text. Fields are listed one by one for
that reason, so a new field on a hardware object cannot leak in unnoticed, and a
report with a field outside that list is refused.

whichllm never sends a report. It opens a prefilled issue in the browser, and the
person reads it and submits it there.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from whichllm.bench.backends import Backend, Sample
from whichllm.constants import _GiB
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.utils import _cache_dir

SCHEMA_VERSION = 1

ISSUE_URL = "https://github.com/Andyyyy64/whichllm/issues/new"
ISSUE_TEMPLATE = "measurement.yml"
# GitHub answers 414 somewhere past 8 KB, and browsers differ before that.
MAX_ISSUE_URL_LENGTH = 6000
_MAX_TOK_PER_SEC = 100_000.0

_SECTION_FIELDS = {
    "hardware": {
        "os",
        "cpu",
        "cpu_cores",
        "ram_gb",
        "ram_bandwidth_gbps",
        "ram_bandwidth_source",
        "gpus",
    },
    "runtime": {"engine", "app", "version"},
    "model": {"repo", "filename", "quant", "file_size_bytes", "context_length"},
    "placement": {"kind", "n_cpu_moe", "n_gpu_layers"},
    "measurement": {"tok_per_sec", "median_tok_per_sec", "tokens", "vram_used_mb"},
    "estimate": {"tok_per_sec", "confidence", "required_memory_gb", "fit"},
}
_GPU_FIELDS = {"name", "vendor", "vram_gb", "bandwidth_gbps"}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "whichllm_version",
    "method",
    *_SECTION_FIELDS,
}
_OPTIONAL_SECTIONS = {"placement", "estimate"}


class ReportError(Exception):
    """A report that cannot be used, with a message for the user."""


def reports_dir() -> Path:
    return _cache_dir() / "reports"


def _hardware_section(hardware: HardwareInfo) -> dict:
    return {
        "os": hardware.os,
        "cpu": hardware.cpu_name,
        "cpu_cores": hardware.cpu_cores,
        "ram_gb": round(hardware.ram_bytes / _GiB, 1),
        "ram_bandwidth_gbps": hardware.ram_bandwidth_gbps,
        "ram_bandwidth_source": hardware.ram_bandwidth_source,
        "gpus": [
            {
                "name": gpu.name,
                "vendor": gpu.vendor,
                "vram_gb": round(gpu.vram_bytes / _GiB, 1),
                "bandwidth_gbps": gpu.memory_bandwidth_gbps,
            }
            for gpu in hardware.gpus
        ],
    }


def _model_section(
    model: ModelInfo | None,
    variant: GGUFVariant | None,
    served_model: str | None,
    context_length: int,
) -> dict:
    return {
        "repo": model.id if model is not None else served_model,
        "filename": variant.filename if variant is not None else None,
        "quant": variant.quant_type if variant is not None else None,
        "file_size_bytes": (
            variant.file_size_bytes
            if variant is not None and not variant.is_estimated
            else None
        ),
        "context_length": context_length,
    }


def _placement_section(picked: CompatibilityResult | None) -> dict | None:
    if picked is None or picked.placement is None:
        return None
    placement = picked.placement
    return {
        "kind": placement.kind,
        "n_cpu_moe": placement.n_cpu_moe,
        "n_gpu_layers": placement.n_gpu_layers,
    }


def _estimate_section(picked: CompatibilityResult | None) -> dict | None:
    if picked is None:
        return None
    speed = picked.estimated_tok_per_sec
    return {
        "tok_per_sec": round(speed, 2) if speed else None,
        "confidence": picked.speed_confidence,
        "required_memory_gb": round(picked.vram_required_bytes / _GiB, 2),
        "fit": picked.fit_type,
    }


def build_report(
    *,
    version: str,
    hardware: HardwareInfo,
    backend: Backend,
    model: ModelInfo | None,
    variant: GGUFVariant | None,
    served_model: str | None,
    context_length: int,
    picked: CompatibilityResult | None,
    samples: list[Sample],
    vram_used_mb: int | None,
) -> dict:
    """The report for one benchmark. Placement and estimate need a ranked pick."""
    speeds = [sample.tok_per_sec for sample in samples]
    return {
        "schema_version": SCHEMA_VERSION,
        "whichllm_version": version,
        "method": "auto",
        "hardware": _hardware_section(hardware),
        "runtime": {
            "engine": backend.engine,
            "app": backend.app,
            "version": backend.version,
        },
        "model": _model_section(model, variant, served_model, context_length),
        "placement": _placement_section(picked),
        "measurement": {
            "tok_per_sec": [round(speed, 2) for speed in speeds],
            "median_tok_per_sec": round(statistics.median(speeds), 2),
            "tokens": samples[0].tokens,
            "vram_used_mb": vram_used_mb,
        },
        "estimate": _estimate_section(picked),
    }


def build_manual_report(
    *,
    version: str,
    hardware: HardwareInfo,
    engine: str,
    model: str,
    quant: str | None,
    context_length: int,
    tok_per_sec: float,
) -> dict:
    """A speed the person measured themselves, on a runtime bench cannot drive."""
    return {
        "schema_version": SCHEMA_VERSION,
        "whichllm_version": version,
        "method": "manual",
        "hardware": _hardware_section(hardware),
        "runtime": {"engine": engine, "app": None, "version": None},
        "model": {
            "repo": model,
            "filename": None,
            "quant": quant,
            "file_size_bytes": None,
            "context_length": context_length,
        },
        "placement": None,
        "measurement": {
            "tok_per_sec": [tok_per_sec],
            "median_tok_per_sec": tok_per_sec,
            "tokens": None,
            "vram_used_mb": None,
        },
        "estimate": None,
    }


def save_report(report: dict, directory: Path | None = None) -> Path:
    """Write a report under a timestamped name. It stays on this machine."""
    target = directory if directory is not None else reports_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = target / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def latest_report_path(directory: Path | None = None) -> Path | None:
    """The newest saved report. Names sort by time."""
    target = directory if directory is not None else reports_dir()
    saved = sorted(target.glob("*.json")) if target.is_dir() else []
    return saved[-1] if saved else None


def report_problems(report: object) -> list[str]:
    """What makes this report unusable. An edited file can hold anything."""
    if not isinstance(report, dict):
        return ["a report is a JSON object"]
    problems = [f"unknown field {name!r}" for name in report.keys() - _TOP_LEVEL_FIELDS]
    problems += [
        f"missing field {name!r}" for name in _TOP_LEVEL_FIELDS - report.keys()
    ]
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("method") not in ("auto", "manual"):
        problems.append("method must be auto or manual")
    for section, fields in _SECTION_FIELDS.items():
        value = report.get(section)
        if value is None and section in _OPTIONAL_SECTIONS:
            continue
        if not isinstance(value, dict):
            problems.append(f"{section} must be an object")
            continue
        problems += [
            f"unknown field {section}.{name}" for name in value.keys() - fields
        ]
        problems += [
            f"missing field {section}.{name}" for name in fields - value.keys()
        ]
    if not problems:
        problems += _value_problems(report)
    return problems


def _value_problems(report: dict) -> list[str]:
    problems = []
    gpus = report["hardware"]["gpus"]
    if not isinstance(gpus, list) or not all(isinstance(gpu, dict) for gpu in gpus):
        return ["hardware.gpus must be a list of objects"]
    for gpu in gpus:
        problems += [f"unknown field gpus.{name}" for name in gpu.keys() - _GPU_FIELDS]
    median = report["measurement"]["median_tok_per_sec"]
    if (
        not isinstance(median, (int, float))
        or isinstance(median, bool)
        or not 0 < median <= _MAX_TOK_PER_SEC
    ):
        problems.append("measurement.median_tok_per_sec is out of range")
    return problems


def load_report(path: Path) -> dict:
    """Read a saved report, refusing anything outside the fields above."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReportError(f"Could not read {path}: {error}") from error
    except ValueError as error:
        raise ReportError(f"{path} is not JSON: {error}") from error
    problems = report_problems(report)
    if problems:
        raise ReportError(f"{path} is not a report: " + "; ".join(sorted(problems)))
    return report


def issue_title(report: dict) -> str:
    gpus = report["hardware"]["gpus"]
    device = gpus[0]["name"] if gpus else report["hardware"]["cpu"]
    model = str(report["model"]["repo"]).rsplit("/", 1)[-1]
    quant = report["model"]["quant"]
    label = f"{model} {quant}" if quant else model
    speed = report["measurement"]["median_tok_per_sec"]
    return f"[Measurement] {device}, {label}: {speed:.1f} tok/s"


def issue_body(report: dict) -> str:
    return json.dumps(report, indent=2)


def issue_url(report: dict, *, include_body: bool = True) -> str:
    """A prefilled issue. Opening it sends nothing until the person submits it."""
    query = {"template": ISSUE_TEMPLATE, "title": issue_title(report)}
    if include_body:
        query["report"] = issue_body(report)
    return f"{ISSUE_URL}?{urlencode(query)}"
