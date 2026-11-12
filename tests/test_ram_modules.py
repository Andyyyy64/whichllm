"""RAM bandwidth from udev on Linux, from Windows on WSL, and its cache."""

from __future__ import annotations

import whichllm.hardware.ram_bandwidth as ram_bandwidth
from whichllm.hardware.ram_bandwidth import (
    _cached_windows_memory_modules,
    _is_wsl,
    _read_udev_modules,
    bandwidth_from_modules,
    detect_ram_bandwidth,
)

# Two DDR5 DIMMs and two empty slots, the way systemd writes the DMI table.
_UDEV = """\
E:MEMORY_ARRAY_LOCATION=System Board Or Motherboard
E:MEMORY_DEVICE_0_PRESENT=1
E:MEMORY_DEVICE_0_TYPE=DDR5
E:MEMORY_DEVICE_0_SPEED_MTS=5600
E:MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS=5600
E:MEMORY_DEVICE_0_DATA_WIDTH=64
E:MEMORY_DEVICE_1_PRESENT=0
E:MEMORY_DEVICE_2_PRESENT=1
E:MEMORY_DEVICE_2_TYPE=DDR5
E:MEMORY_DEVICE_2_SPEED_MTS=5600
E:MEMORY_DEVICE_2_CONFIGURED_SPEED_MTS=5200
E:MEMORY_DEVICE_2_DATA_WIDTH=64
E:MEMORY_DEVICE_3_PRESENT=0
"""


def test_udev_dmi_properties_give_the_modules(tmp_path):
    path = tmp_path / "+dmi:id"
    path.write_text(_UDEV)
    modules = _read_udev_modules(path)
    assert len(modules) == 2
    result = bandwidth_from_modules(modules, "AMD Ryzen 9 9950X")
    # The slower module sets the clock: DDR5-5200 on a 128-bit bus.
    assert result.gbps == 83.2
    assert result.detail == "DDR5-5200, 128-bit"


def test_missing_udev_data_gives_nothing(tmp_path):
    assert _read_udev_modules(tmp_path / "absent") == []


def _proc_version(monkeypatch, system, text, distro=None):
    """Stand in for the OS, /proc/version (None when absent) and WSL_DISTRO_NAME."""

    class Proc:
        def __init__(self, path):
            assert path == "/proc/version"

        def read_text(self):
            if text is None:
                raise FileNotFoundError(2, "No such file or directory")
            return text

    monkeypatch.setattr(ram_bandwidth.platform, "system", lambda: system)
    monkeypatch.setattr(ram_bandwidth, "Path", Proc)
    if distro is None:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    else:
        monkeypatch.setenv("WSL_DISTRO_NAME", distro)


_WSL_KERNEL = "Linux version 6.6.87.2-microsoft-standard-WSL2 (root@439a258ad544)"
_LINUX_KERNEL = "Linux version 6.8.0-45-generic (buildd@lcy02-amd64-075)"


def test_linux_under_wsl_is_wsl(monkeypatch):
    # _is_wsl here is the real function: conftest patches the module's name for it.
    _proc_version(monkeypatch, "Linux", _WSL_KERNEL)
    assert _is_wsl()
    _proc_version(monkeypatch, "Linux", None, distro="Ubuntu-24.04")
    assert _is_wsl()
    _proc_version(monkeypatch, "Linux", _LINUX_KERNEL)
    assert not _is_wsl()
    _proc_version(monkeypatch, "Linux", None)
    assert not _is_wsl()


def test_windows_on_a_wsl_share_is_not_wsl(monkeypatch):
    # From \\wsl.localhost\Ubuntu\home\me, Windows resolves "/proc/version" to the
    # distro's file, which says microsoft.
    _proc_version(monkeypatch, "Windows", _WSL_KERNEL)
    assert not _is_wsl()
    _proc_version(monkeypatch, "Windows", _WSL_KERNEL, distro="Ubuntu-24.04")
    assert not _is_wsl()
    _proc_version(monkeypatch, "Darwin", None)
    assert not _is_wsl()


def test_wsl_asks_windows_for_the_modules(monkeypatch):
    calls = []

    def windows(executable="powershell"):
        calls.append(executable)
        return [
            {"ConfiguredClockSpeed": 5600, "DataWidth": 64, "SMBIOSMemoryType": 34},
            {"ConfiguredClockSpeed": 5600, "DataWidth": 64, "SMBIOSMemoryType": 34},
        ]

    monkeypatch.setattr(ram_bandwidth, "_is_wsl", lambda: True)
    monkeypatch.setattr(ram_bandwidth, "_windows_memory_modules", windows)

    result = detect_ram_bandwidth("linux", "Intel(R) Core(TM) Ultra 9 285K", [])

    assert calls == ["powershell.exe"]
    assert result.source == "detected"
    assert result.gbps == 89.6


def test_windows_modules_are_cached(monkeypatch):
    calls = []

    def windows(executable="powershell"):
        calls.append(executable)
        return [{"ConfiguredClockSpeed": 3200, "DataWidth": 64}]

    monkeypatch.setattr(ram_bandwidth, "_windows_memory_modules", windows)

    first = _cached_windows_memory_modules("powershell")
    second = _cached_windows_memory_modules("powershell")

    assert first == second
    assert calls == ["powershell"]
