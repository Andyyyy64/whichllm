"""The curated benchmark snapshots in whichllm/data: format and provenance."""

from __future__ import annotations

import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

import whichllm.models.benchmark_sources as sources
from whichllm.data import DATA_DIR, read_json

_DATE_RE = re.compile(r"\d{4}-\d{2}(-\d{2})?")
_SNAPSHOTS = ("aa_index.json", "livebench.json", "vision.json")


@pytest.mark.parametrize("name", _SNAPSHOTS)
def test_snapshot_says_when_it_was_read(name):
    snapshot = read_json(name)
    assert _DATE_RE.fullmatch(snapshot["read_at"])
    assert snapshot["note"]
    assert snapshot["models"]
    assert all("/" in model_id for model_id in snapshot["models"])


def test_displayed_snapshot_date_is_the_newest_read_date():
    months = [read_json(name)["read_at"][:7] for name in _SNAPSHOTS]
    assert sources.BENCHMARK_SNAPSHOT == max(months)


def test_every_aa_entry_says_where_it_came_from():
    for model_id, entry in read_json("aa_index.json")["models"].items():
        assert isinstance(entry["raw"], (int, float)), model_id
        if entry["kind"] == "live":
            # The AA model page the value was read from.
            assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*", entry["slug"]), model_id
            assert set(entry) <= {"raw", "kind", "slug", "estimated"}, model_id
        else:
            # A hand estimate: no AA page to point at.
            assert entry["kind"] == "peer", model_id
            assert set(entry) == {"raw", "kind"}, model_id


def test_aa_name_map_entries_are_repo_ids():
    names = read_json("aa_names.json")
    assert all(
        ids and all("/" in repo for repo in ids) for ids in names["names"].values()
    )
    assert names["notes"].keys() <= names["names"].keys()


def test_livebench_carryover_is_dated_and_explained():
    carryover = read_json("livebench.json")["carryover"]
    assert _DATE_RE.fullmatch(carryover["read_at"])
    assert carryover["note"]
    assert carryover["models"]


def _import_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_livebench_csv.py"
    spec = importlib.util.spec_from_file_location("import_livebench_csv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_livebench_import_rewrites_only_the_measured_models(tmp_path, monkeypatch):
    script = _import_script()
    snapshot = tmp_path / "livebench.json"
    snapshot.write_text((DATA_DIR / "livebench.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(script, "SNAPSHOT_PATH", snapshot)
    csv = "model,coding,math\nglm-5.1,70,71.04\ngemini-9,90,90\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(csv))

    assert script.main(["import_livebench_csv.py", "2026-11-05"]) == 0

    written = json.loads(snapshot.read_text(encoding="utf-8"))
    assert written["read_at"] == "2026-11-05"
    assert written["source"] == "https://livebench.ai/table_2026_11_05.csv"
    assert written["models"] == {"zai-org/GLM-5.1": 70.5}
    assert written["carryover"] == read_json("livebench.json")["carryover"]


def test_livebench_import_needs_the_csv_date(monkeypatch):
    script = _import_script()
    monkeypatch.setattr("sys.stdin", io.StringIO("model\n"))
    assert script.main(["import_livebench_csv.py"]) == 2
