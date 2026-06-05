"""Tests for Windows CPU name fallback detection."""

from __future__ import annotations

import subprocess

from whichllm.hardware import cpu


def test_windows_cpu_name_uses_cim_when_wmic_is_empty(monkeypatch):
    monkeypatch.setattr(cpu.platform, "system", lambda: "Windows")

    def fake_run(args, **kwargs):
        if args == ["wmic", "cpu", "get", "name"]:
            return subprocess.CompletedProcess(args, 0, stdout="Name\n\n", stderr="")
        if args[0] == "powershell":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="Intel(R) Xeon(R) W-11955M CPU @ 2.60GHz\n",
                stderr="",
            )
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)

    assert cpu.detect_cpu_name() == "Intel(R) Xeon(R) W-11955M CPU @ 2.60GHz"


def test_windows_cpu_name_uses_wmic_when_available(monkeypatch):
    monkeypatch.setattr(cpu.platform, "system", lambda: "Windows")

    def fake_run(args, **kwargs):
        if args == ["wmic", "cpu", "get", "name"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="Name\nIntel(R) Core(TM) i9-11900H @ 2.50GHz\n",
                stderr="",
            )
        raise AssertionError("CIM fallback should not run when wmic returns a name")

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)

    assert cpu.detect_cpu_name() == "Intel(R) Core(TM) i9-11900H @ 2.50GHz"
