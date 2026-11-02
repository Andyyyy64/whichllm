"""Where HF_TOKEN is sent, and the --ram-bandwidth note for discrete GPUs."""

from __future__ import annotations

import pytest

import whichllm.models.hf as hf
from whichllm.cli_hardware import _apply_ram_bandwidth
from whichllm.hardware.types import GPUInfo, HardwareInfo


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("WHICHLLM_HF_TOKEN_TO_ENDPOINT", raising=False)
    monkeypatch.setattr(hf, "_token_warning_shown", False)


def test_token_goes_to_hugging_face():
    assert hf._hf_headers()["Authorization"] == "Bearer hf_test_token"


def test_token_is_not_sent_to_a_mirror_by_default(monkeypatch, caplog):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert "Authorization" not in hf._hf_headers()
    assert "WHICHLLM_HF_TOKEN_TO_ENDPOINT" in caplog.text


def test_a_mirror_gets_the_token_when_asked(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("WHICHLLM_HF_TOKEN_TO_ENDPOINT", "1")
    assert hf._hf_headers()["Authorization"] == "Bearer hf_test_token"


def test_token_never_goes_over_http(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://huggingface.co")
    monkeypatch.setenv("WHICHLLM_HF_TOKEN_TO_ENDPOINT", "1")
    assert "Authorization" not in hf._hf_headers()


def _hardware(*gpus: GPUInfo) -> HardwareInfo:
    return HardwareInfo(gpus=list(gpus), cpu_name="CPU", ram_bytes=64 * 1024**3)


def test_ram_bandwidth_on_a_discrete_gpu_says_what_it_sets():
    dgpu = GPUInfo(
        name="RTX 4090",
        vendor="nvidia",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1008,
    )
    hw = _apply_ram_bandwidth(_hardware(dgpu), 96.0)
    assert hw.ram_bandwidth_gbps == 96.0
    assert dgpu.memory_bandwidth_gbps == 1008
    assert any("--bandwidth" in note for note in hw.budget_notes)


def test_ram_bandwidth_on_an_igpu_needs_no_note():
    igpu = GPUInfo(name="Radeon 780M", vendor="amd", vram_bytes=0, shared_memory=True)
    hw = _apply_ram_bandwidth(_hardware(igpu), 68.0)
    assert igpu.memory_bandwidth_gbps == 68.0
    assert hw.budget_notes == []
