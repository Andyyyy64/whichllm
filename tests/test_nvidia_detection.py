import builtins
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import whichllm.hardware.nvidia as nvidia
from whichllm.hardware.nvidia import detect_nvidia_gpus


def test_nvidia_smi_fallback_when_pynvml_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA GeForce RTX 5060 Ti, 16303\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 5060 Ti"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 16303 * 1024**2


def test_nvidia_smi_fallback_applies_rtx_a3000_laptop_catalog(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA RTX A3000 Laptop GPU, 6144 MiB\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA RTX A3000 Laptop GPU"
    assert gpus[0].compute_capability == (8, 6)
    assert gpus[0].memory_bandwidth_gbps == 264.0


def test_nvidia_smi_fallback_resolves_laptop_5090_via_dbgpu(monkeypatch):
    # Regression for #74: a laptop RTX 5090 must not inherit the desktop
    # 1792 GB/s bandwidth. dbgpu carries "GeForce RTX 5090 Mobile" (~896).
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA GeForce RTX 5090 Laptop GPU, 24564 MiB\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 5090 Laptop GPU"
    bw = gpus[0].memory_bandwidth_gbps
    assert bw is not None
    assert bw < 1500.0  # mobile ~896, not the 1792 desktop value


def test_nvidia_smi_fallback_when_nvml_init_fails(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    fake_pynvml = SimpleNamespace(
        NVMLError=FakeNVMLError,
        nvmlInit=Mock(side_effect=FakeNVMLError("NVML unavailable")),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            return fake_pynvml
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA DGX Spark, 128000 MiB\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA DGX Spark"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 128000 * 1024**2
    assert gpus[0].shared_memory is True


def test_nvidia_smi_fallback_detects_gb10_with_na_memory(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA GB10, [N/A]\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 128 * 1024**3
    )

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GB10"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 128 * 1024**3
    assert gpus[0].shared_memory is True
    assert gpus[0].compute_capability == (12, 1)
    assert gpus[0].memory_bandwidth_gbps == 273.0


def test_nvml_detects_gb10_when_memory_info_is_unavailable(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    handle = object()
    fake_pynvml = SimpleNamespace(
        NVMLError=FakeNVMLError,
        nvmlInit=Mock(),
        nvmlDeviceGetCount=Mock(return_value=1),
        nvmlSystemGetDriverVersion=Mock(return_value=b"580.142"),
        nvmlSystemGetCudaDriverVersion_v2=Mock(return_value=12010),
        nvmlDeviceGetHandleByIndex=Mock(return_value=handle),
        nvmlDeviceGetName=Mock(return_value="NVIDIA GB10"),
        nvmlDeviceGetMemoryInfo=Mock(side_effect=FakeNVMLError("not supported")),
        nvmlShutdown=Mock(),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            return fake_pynvml
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        raise AssertionError("nvidia-smi fallback should not be used")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 128 * 1024**3
    )

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GB10"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 128 * 1024**3
    assert gpus[0].shared_memory is True
    assert gpus[0].compute_capability == (12, 1)
    assert gpus[0].cuda_version == "12.1"
    assert gpus[0].memory_bandwidth_gbps == 273.0


def test_nvidia_smi_fallback_returns_empty_on_command_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert detect_nvidia_gpus() == []


@pytest.mark.parametrize(
    "name", ["NVIDIA Jetson AGX Xavier", "NVIDIA Jetson Xavier NX"]
)
def test_xavier_smi_uses_system_ram_when_dedicated_memory_is_unavailable(
    monkeypatch, name
):
    monkeypatch.setattr(
        nvidia.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{name}, [N/A]\n"),
    )
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 16 * 1024**3
    )

    gpus = nvidia._detect_nvidia_gpus_via_smi()

    assert len(gpus) == 1
    assert gpus[0].name == name
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 16 * 1024**3


@pytest.mark.parametrize(
    "name", [b"NVIDIA Jetson AGX Xavier", "NVIDIA Jetson Xavier NX"]
)
def test_xavier_nvml_uses_system_ram_when_memory_query_fails(monkeypatch, name):
    class FakeNVMLError(Exception):
        pass

    fake = SimpleNamespace(
        NVMLError=FakeNVMLError,
        nvmlInit=Mock(),
        nvmlDeviceGetCount=Mock(return_value=1),
        nvmlDeviceGetHandleByIndex=Mock(return_value=object()),
        nvmlDeviceGetName=Mock(return_value=name),
        nvmlDeviceGetMemoryInfo=Mock(side_effect=FakeNVMLError("not supported")),
        nvmlShutdown=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(
        "whichllm.hardware.memory.detect_ram_bytes", lambda: 32 * 1024**3
    )

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == (name.decode() if isinstance(name, bytes) else name)
    assert gpus[0].shared_memory is True
    assert gpus[0].vram_bytes == 32 * 1024**3
    fake.nvmlShutdown.assert_called_once()
