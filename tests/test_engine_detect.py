"""Finding installed engines: binaries, installer paths and fingerprinted ports."""

import json
import stat
import sys

import httpx
import pytest
from typer.testing import CliRunner

from whichllm.cli import app
from whichllm.engines import load_cards
from whichllm.engines.detect import Installed, _version, detect_installed
from whichllm.hardware.types import HardwareInfo

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="fake engines are shell scripts"
)


@pytest.mark.parametrize(
    ("printed", "version"),
    [
        ("b9986-91c631b21", "b9986"),
        ("version: 12069 (abc1234)\nbuilt with gcc", "b12069"),
        ("ollama version is 0.40.14", "0.40.14"),
        ("v1.124.2", "1.124.2"),
        ('{"result":"KoboldCpp","version":"1.124.2"}', "1.124.2"),
        ("no version here", None),
    ],
)
def test_versions_read_the_way_each_engine_prints_them(printed, version):
    assert _version(printed) == version


def _fake(directory, name: str, output: str) -> None:
    script = directory / name
    script.write_text(f"#!/bin/sh\necho '{output}'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _no_ports() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )


@posix_only
def test_a_mainline_llama_server_is_llama_cpp(monkeypatch, tmp_path):
    _fake(tmp_path, "llama-server", "usage: llama-server [options] --fit on|off")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    found = detect_installed(load_cards(), "linux", client=_no_ports())

    assert set(found) == {"llama.cpp"}
    assert found["llama.cpp"].path == str(tmp_path / "llama-server")


@posix_only
def test_an_ik_llama_server_is_told_apart_by_its_help(monkeypatch, tmp_path):
    # Both ship a binary named llama-server; only ik's --help has this flag.
    _fake(tmp_path, "llama-server", "usage: llama-server [options] --run-time-repack")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    found = detect_installed(load_cards(), "linux", client=_no_ports())

    assert set(found) == {"ik_llama.cpp"}


@posix_only
def test_the_llama_installer_path_counts_with_its_build(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    _fake(home / ".local" / "bin", "llama", "b9986-91c631b21")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(home))

    found = detect_installed(load_cards(), "linux", client=_no_ports())

    assert found["llama.cpp"] == Installed(
        path=str(home / ".local" / "bin" / "llama"), version="b9986", running=False
    )


def test_a_server_counts_only_with_its_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.port == 11434 and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.14"})
        # Something else on LM Studio's port, without LM Studio's greeting.
        if request.url.port == 1234:
            return httpx.Response(200, json={"hello": "world"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(answer))
    found = detect_installed(load_cards(), "linux", client=client)

    assert found == {"ollama": Installed(path=None, version="0.40.14", running=True)}


def test_without_probing_no_port_is_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"asked {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(refuse))

    assert detect_installed(load_cards(), "linux", probe=False, client=client) == {}


def _machine() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="CPU", cpu_cores=8, has_avx2=True, ram_bytes=32 << 30, os="linux"
    )


def test_engines_shows_what_is_installed(monkeypatch):
    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", _machine)
    monkeypatch.setattr(
        "whichllm.engines.detect.detect_installed",
        lambda cards, os_name: {
            "ollama": Installed(path="/usr/bin/ollama", version="0.40.14", running=True)
        },
    )

    result = CliRunner().invoke(app, ["engines"], terminal_width=160)
    shown = " ".join(result.output.replace("│", " ").split())

    assert result.exit_code == 0, result.output
    assert "Installed" in shown
    assert "yes, 0.40.14, running" in shown


def test_engines_json_carries_the_install(monkeypatch):
    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", _machine)
    monkeypatch.setattr(
        "whichllm.engines.detect.detect_installed",
        lambda cards, os_name: {
            "llama.cpp": Installed(path="/opt/llama", version="b9986", running=False)
        },
    )

    result = CliRunner().invoke(app, ["engines", "--json"])

    rows = {row["id"]: row for row in json.loads(result.output)["engines"]}
    assert rows["llama.cpp"]["installed"] is True
    assert rows["llama.cpp"]["installed_version"] == "b9986"
    assert rows["llama.cpp"]["installed_path"] == "/opt/llama"
    assert rows["ollama"]["installed"] is False


def test_a_simulated_gpu_does_not_report_this_machines_installs(monkeypatch):
    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", _machine)

    def must_not_run(cards, os_name):
        raise AssertionError("installs belong to the real machine")

    monkeypatch.setattr("whichllm.engines.detect.detect_installed", must_not_run)

    result = CliRunner().invoke(
        app, ["engines", "--gpu", "RTX 4090"], terminal_width=160
    )

    assert result.exit_code == 0, result.output
    assert "Installed" not in result.output
