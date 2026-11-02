"""Free RAM, the GPU's PCIe link and measured RAM bandwidth: shown, never used."""

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

from rich.console import Console
from typer.testing import CliRunner

import whichllm.hardware.ram_measure as ram_measure
from whichllm.cli import app
from whichllm.hardware.nvidia import _pcie_link, detect_nvidia_gpus
from whichllm.hardware.types import GPUInfo, HardwareInfo

_GiB = 1024**3


def _desktop(**changes) -> HardwareInfo:
    fields = dict(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 5080",
                vendor="nvidia",
                vram_bytes=16 * _GiB,
                memory_bandwidth_gbps=960.0,
                compute_capability=(12, 0),
                pcie_generation=5,
                pcie_lanes=16,
            )
        ],
        cpu_name="Intel(R) Core(TM) Ultra 9 285K",
        cpu_cores=24,
        ram_bytes=64 * _GiB,
        ram_available_bytes=41 * _GiB,
        ram_bandwidth_gbps=89.6,
        ram_bandwidth_source="detected",
        ram_bandwidth_detail="DDR5-5600, 128-bit",
        disk_free_bytes=500 * _GiB,
        os="windows",
    )
    fields.update(changes)
    return HardwareInfo(**fields)


def _shown(hardware: HardwareInfo, monkeypatch) -> str:
    from whichllm.output import _console as console_mod
    from whichllm.output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(
        console_mod, "console", Console(file=buf, force_terminal=False, width=200)
    )
    display_mod.display_hardware(hardware)
    return " ".join(buf.getvalue().split())


def test_the_panel_shows_free_ram_pcie_and_the_measurement(monkeypatch):
    shown = _shown(_desktop(ram_bandwidth_measured_gbps=77.4), monkeypatch)

    assert "(CC 12.0, PCIe 5.0 x16)" in shown
    assert "RAM: 64.0 GB (41.0 GB free) — BW: 90 GB/s" in shown
    assert "(DDR5-5600, 128-bit, measured 77 GB/s)" in shown


def test_nothing_known_adds_nothing(monkeypatch):
    hardware = _desktop(ram_available_bytes=None, ram_bandwidth_detail=None)
    hardware.gpus[0].pcie_generation = None
    shown = _shown(hardware, monkeypatch)

    assert "PCIe" not in shown
    assert "GB free" not in shown
    assert "measured" not in shown


def test_free_ram_sits_beside_a_budget(monkeypatch):
    shown = _shown(_desktop(ram_budget_bytes=32 * _GiB), monkeypatch)

    assert "RAM: 64.0 GB (41.0 GB free, budget 32.0 GB)" in shown


def test_json_carries_free_ram_and_the_pcie_link(monkeypatch):
    from whichllm.output import _console as console_mod
    from whichllm.output.json_output import display_json

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=10_000))
    display_json([], _desktop())
    hardware = json.loads(buf.getvalue())["hardware"]

    assert hardware["ram_available_bytes"] == 41 * _GiB
    assert hardware["gpus"][0]["pcie_generation"] == 5
    assert hardware["gpus"][0]["pcie_lanes"] == 16


def test_pcie_link_reads_the_maximum_the_slot_allows():
    nvml = SimpleNamespace(
        nvmlDeviceGetMaxPcieLinkGeneration=Mock(return_value=4),
        nvmlDeviceGetMaxPcieLinkWidth=Mock(return_value=8),
    )

    assert _pcie_link(nvml, object()) == (4, 8)


def test_pcie_link_unknown_without_an_answer():
    def unsupported(handle):
        raise RuntimeError("Not Supported")

    refuses = SimpleNamespace(
        nvmlDeviceGetMaxPcieLinkGeneration=unsupported,
        nvmlDeviceGetMaxPcieLinkWidth=unsupported,
    )
    zero = SimpleNamespace(
        nvmlDeviceGetMaxPcieLinkGeneration=Mock(return_value=0),
        nvmlDeviceGetMaxPcieLinkWidth=Mock(return_value=16),
    )

    assert _pcie_link(refuses, object()) == (None, None)
    assert _pcie_link(zero, object()) == (None, None)


def _fake_nvml(name: str) -> SimpleNamespace:
    class FakeNVMLError(Exception):
        pass

    return SimpleNamespace(
        NVMLError=FakeNVMLError,
        NVML_CLOCK_MEM=2,
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetCount=Mock(return_value=1),
        nvmlSystemGetDriverVersion=Mock(return_value="580.0"),
        nvmlSystemGetCudaDriverVersion_v2=Mock(return_value=13000),
        nvmlDeviceGetHandleByIndex=Mock(return_value=object()),
        nvmlDeviceGetName=Mock(return_value=name),
        nvmlDeviceGetMemoryInfo=Mock(return_value=SimpleNamespace(total=16 * _GiB)),
        nvmlDeviceGetMaxClockInfo=Mock(return_value=15000),
        nvmlDeviceGetMaxPcieLinkGeneration=Mock(return_value=5),
        nvmlDeviceGetMaxPcieLinkWidth=Mock(return_value=16),
    )


def test_nvml_detection_records_the_pcie_link(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml("NVIDIA GeForce RTX 5080"))

    gpus = detect_nvidia_gpus()

    assert (gpus[0].pcie_generation, gpus[0].pcie_lanes) == (5, 16)


def test_a_unified_memory_gpu_has_no_pcie_link(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml("NVIDIA GB10"))
    monkeypatch.setattr("whichllm.hardware.memory.detect_ram_bytes", lambda: 128 * _GiB)

    gpus = detect_nvidia_gpus()

    assert gpus[0].shared_memory is True
    assert gpus[0].pcie_generation is None


def test_the_measurement_copies_real_memory(monkeypatch):
    monkeypatch.setattr(ram_measure, "_COPY_BYTES", 32 * 1024**2)

    rate = ram_measure.measure_ram_bandwidth_gbps()

    assert rate is not None and rate > 1.0


def test_too_little_free_ram_skips_the_measurement():
    assert ram_measure.measure_ram_bandwidth_gbps(available_bytes=16 * 1024**2) is None


def test_the_measurement_stops_when_more_threads_stop_helping(monkeypatch):
    rates = {2: 40.0, 4: 70.0, 8: 60.0, 16: 90.0}
    tried = []

    def rate(source, target, span, threads):
        tried.append(threads)
        return rates[threads]

    monkeypatch.setattr(ram_measure, "_COPY_BYTES", 64 * 1024**2)
    monkeypatch.setattr(ram_measure, "_usable_cpus", lambda: 16)
    monkeypatch.setattr(ram_measure, "_copy_rate", rate)

    assert ram_measure.measure_ram_bandwidth_gbps() == 70.0
    assert tried == [2, 4, 8]


def _hardware_run(monkeypatch, hardware: HardwareInfo) -> tuple[str, Mock]:
    measure = Mock(return_value=77.4)
    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", lambda: hardware)
    monkeypatch.setattr(
        "whichllm.hardware.ram_measure.measure_ram_bandwidth_gbps", measure
    )
    result = CliRunner().invoke(app, ["hardware"], terminal_width=200)
    assert result.exit_code == 0, result.output
    return " ".join(result.output.replace("│", " ").split()), measure


def test_the_hardware_command_measures_ram(monkeypatch):
    output, measure = _hardware_run(monkeypatch, _desktop())

    measure.assert_called_once_with(41 * _GiB)
    assert "measured 77 GB/s" in output


def test_unified_memory_is_not_measured(monkeypatch):
    # A CPU copy cannot reach what the GPU reads from unified memory.
    hardware = _desktop(
        gpus=[],
        ram_bandwidth_gbps=400.0,
        ram_bandwidth_source="unified",
        ram_bandwidth_detail=None,
    )
    output, measure = _hardware_run(monkeypatch, hardware)

    measure.assert_not_called()
    assert "measured" not in output
