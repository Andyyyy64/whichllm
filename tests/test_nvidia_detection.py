import builtins
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

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
