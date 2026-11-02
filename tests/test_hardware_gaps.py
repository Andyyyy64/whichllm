"""AMD APU shared memory (GTT) and GPU simulation memory variants."""

from __future__ import annotations

import whichllm.hardware.amd as amd
from whichllm.hardware.gpu_simulator import create_synthetic_gpu

_GiB = 1024**3


def _amd_card(root, device_id: str, vram: int, gtt: int) -> None:
    card = root / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "device").write_text(f"{device_id}\n")
    (card / "mem_info_vram_total").write_text(str(vram))
    (card / "mem_info_gtt_total").write_text(str(gtt))


def test_apu_gpu_memory_is_carve_out_plus_gtt(tmp_path):
    _amd_card(tmp_path, "0x1586", 512 * 1024**2, 64 * _GiB)
    (gpu,) = amd._detect_from_sysfs(tmp_path)
    assert gpu.shared_memory
    assert gpu.vram_bytes == 512 * 1024**2 + 64 * _GiB


def test_discrete_gpu_ignores_gtt(tmp_path):
    _amd_card(tmp_path, "0x744c", 24 * _GiB, 32 * _GiB)  # RX 7900 XTX
    (gpu,) = amd._detect_from_sysfs(tmp_path)
    assert not gpu.shared_memory
    assert gpu.vram_bytes == 24 * _GiB


def test_rtx_3060_defaults_to_the_12_gb_card():
    gpu = create_synthetic_gpu("RTX 3060")
    assert gpu.vram_bytes == 12 * _GiB
    assert gpu.memory_bandwidth_gbps == 360.0


def test_vram_picks_the_matching_memory_variant():
    gpu = create_synthetic_gpu("RTX 3060", vram_override_gb=8)
    assert gpu.memory_bandwidth_gbps == 240.0
    twelve = create_synthetic_gpu("RTX 3060", vram_override_gb=12)
    assert twelve.memory_bandwidth_gbps == 360.0
