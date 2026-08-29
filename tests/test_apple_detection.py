"""Tests for Apple Silicon GPU detection on macOS."""

from __future__ import annotations

import json
import subprocess

from whichllm.hardware import apple


def test_detect_apple_gpu_caps_unified_memory_at_iogpu_wired_limit(monkeypatch):
    hardware = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M1 Max",
                "physical_memory": "32 GB",
            }
        ]
    }

    def fake_run(args, **kwargs):
        if args == ["system_profiler", "SPHardwareDataType", "-json"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(hardware), stderr=""
            )
        if args == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return subprocess.CompletedProcess(args, 0, stdout="26000\n", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(apple.subprocess, "run", fake_run)

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 26000 * 1024**2
    assert gpus[0].shared_memory is True


def test_detect_apple_gpu_keeps_unified_memory_when_no_budget_signal(
    monkeypatch,
):
    hardware = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M1 Max",
                "physical_memory": "32 GB",
            }
        ]
    }

    def fake_run(args, **kwargs):
        if args == ["system_profiler", "SPHardwareDataType", "-json"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(hardware), stderr=""
            )
        if args == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="unknown oid")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(apple.subprocess, "run", fake_run)
    monkeypatch.setattr(
        apple, "_detect_metal_recommended_working_set_bytes", lambda: None
    )

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 32 * 1024**3


def test_detect_apple_gpu_keeps_unified_memory_when_sysctl_cannot_run(monkeypatch):
    monkeypatch.setattr(
        apple, "_detect_metal_recommended_working_set_bytes", lambda: None
    )
    hardware = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M1 Max",
                "physical_memory": "32 GB",
            }
        ]
    }

    def fake_run(args, **kwargs):
        if args == ["system_profiler", "SPHardwareDataType", "-json"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(hardware), stderr=""
            )
        if args == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            raise PermissionError("sysctl is not permitted")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(apple.subprocess, "run", fake_run)

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 32 * 1024**3


def _hardware_payload(chip: str = "Apple M4", memory: str = "16 GB") -> dict:
    return {"SPHardwareDataType": [{"chip_type": chip, "physical_memory": memory}]}


def _fake_run_with_wired_limit(payload: dict, sysctl_stdout: str):
    def fake_run(args, **kwargs):
        if args == ["system_profiler", "SPHardwareDataType", "-json"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(payload), stderr=""
            )
        if args == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return subprocess.CompletedProcess(args, 0, stdout=sysctl_stdout, stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    return fake_run


def test_detect_apple_gpu_falls_back_to_metal_working_set_on_stock_machine(monkeypatch):
    """iogpu.wired_limit_mb == 0 is the default on every unmodified Mac.

    Before this, the whole of physical memory was reported as GPU-addressable.
    Measured on a 16 GB M4: Metal reports 11.84 GiB, llama.cpp agrees
    (12124.17 MiB), so the old behaviour overstated the budget by 20%.
    """
    monkeypatch.setattr(
        apple.subprocess, "run", _fake_run_with_wired_limit(_hardware_payload(), "0\n")
    )
    monkeypatch.setattr(
        apple,
        "_detect_metal_recommended_working_set_bytes",
        lambda: 12713115648,  # 11.84 GiB, read from Metal on a 16 GB M4
    )

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 12713115648


def test_detect_apple_gpu_prefers_explicit_wired_limit_over_metal(monkeypatch):
    """An explicitly raised wired limit is a deliberate user choice: honour it."""
    monkeypatch.setattr(
        apple.subprocess,
        "run",
        _fake_run_with_wired_limit(
            _hardware_payload("Apple M1 Max", "32 GB"), "26000\n"
        ),
    )
    monkeypatch.setattr(
        apple,
        "_detect_metal_recommended_working_set_bytes",
        lambda: 24 * 1024**3,
    )

    gpus = apple.detect_apple_gpu()

    assert gpus[0].vram_bytes == 26000 * 1024**2


def test_detect_apple_gpu_never_exceeds_physical_memory(monkeypatch):
    """A Metal value above installed RAM must not inflate the budget."""
    monkeypatch.setattr(
        apple.subprocess, "run", _fake_run_with_wired_limit(_hardware_payload(), "0\n")
    )
    monkeypatch.setattr(
        apple, "_detect_metal_recommended_working_set_bytes", lambda: 64 * 1024**3
    )

    gpus = apple.detect_apple_gpu()

    assert gpus[0].vram_bytes == 16 * 1024**3


def test_metal_working_set_probe_returns_none_off_darwin(monkeypatch):
    monkeypatch.setattr(apple.sys, "platform", "linux")

    assert apple._detect_metal_recommended_working_set_bytes() is None
