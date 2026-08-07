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


def test_detect_apple_gpu_keeps_unified_memory_when_wired_limit_is_unavailable(
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

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 32 * 1024**3


def test_detect_apple_gpu_keeps_unified_memory_when_sysctl_cannot_run(monkeypatch):
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
