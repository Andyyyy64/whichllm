"""Tests for GPU simulator (--gpu flag) using dbgpu database."""

import pytest

from whichllm.constants import _GiB
from whichllm.hardware.gpu_simulator import create_synthetic_gpu


class TestKnownGPULookup:
    def test_nvidia_rtx_4090(self):
        gpu = create_synthetic_gpu("RTX 4090")
        assert gpu.vram_bytes == 24 * _GiB
        assert gpu.vendor == "nvidia"
        assert gpu.memory_bandwidth_gbps is not None
        assert gpu.compute_capability is not None
        assert gpu.compute_capability[0] >= 8
        assert "(simulated)" in gpu.name

    def test_nvidia_rtx_3060(self):
        gpu = create_synthetic_gpu("RTX 3060")
        assert "RTX 3060" in gpu.name
        assert "Ti" not in gpu.name  # Should NOT match RTX 3060 Ti
        assert gpu.vendor == "nvidia"
        assert gpu.compute_capability is not None

    def test_nvidia_rtx_3060_ti(self):
        gpu = create_synthetic_gpu("RTX 3060 Ti")
        assert "RTX 3060 Ti" in gpu.name
        assert gpu.vram_bytes == 8 * _GiB
        assert gpu.vendor == "nvidia"

    def test_amd_rx_7900_xtx(self):
        gpu = create_synthetic_gpu("RX 7900 XTX")
        assert gpu.vram_bytes == 24 * _GiB
        assert gpu.vendor == "amd"
        assert gpu.memory_bandwidth_gbps is not None

    def test_nvidia_gtx_1080(self):
        gpu = create_synthetic_gpu("GTX 1080")
        assert gpu.vram_bytes == 8 * _GiB
        assert gpu.vendor == "nvidia"
        assert gpu.compute_capability == (6, 1)

    def test_a100_80gb_alias(self):
        gpu = create_synthetic_gpu("A100 80GB")
        assert gpu.vram_bytes == 80 * _GiB
        assert gpu.vendor == "nvidia"
        assert "(simulated)" in gpu.name

    def test_h100_80gb_alias(self):
        gpu = create_synthetic_gpu("H100 80GB")
        assert gpu.vram_bytes == 80 * _GiB
        assert gpu.vendor == "nvidia"
        assert "(simulated)" in gpu.name


class TestAppleSiliconAliases:
    @pytest.mark.parametrize(
        "chip",
        [
            "M1",
            "M1 Max",
            "M1 Ultra",
            "M2",
            "M2 Max",
            "M2 Ultra",
            "M3",
            "M3 Max",
            "M3 Ultra",
            "M4",
            "M4 Max",
            "M4 Ultra",
        ],
    )
    def test_apple_prefixed_alias_matches_plain_chip_name(self, chip):
        plain = create_synthetic_gpu(chip)
        prefixed = create_synthetic_gpu(f"Apple {chip}")

        assert prefixed.name == plain.name
        assert prefixed.vendor == "apple"
        assert prefixed.vram_bytes == plain.vram_bytes
        assert prefixed.memory_bandwidth_gbps == plain.memory_bandwidth_gbps


class TestVRAMOverride:
    def test_override_known_gpu(self):
        gpu = create_synthetic_gpu("RTX 4060 Ti", vram_override_gb=16)
        assert gpu.vram_bytes == 16 * _GiB
        assert gpu.memory_bandwidth_gbps is not None

    def test_override_unknown_gpu(self):
        gpu = create_synthetic_gpu("Nonexistent GPU 9999", vram_override_gb=48)
        assert gpu.vram_bytes == 48 * _GiB
        assert "(simulated)" in gpu.name


class TestUnknownGPU:
    def test_unknown_without_vram_raises(self):
        with pytest.raises(ValueError, match="Unknown GPU"):
            create_synthetic_gpu("Nonexistent GPU 9999")

    def test_unknown_with_vram_succeeds(self):
        gpu = create_synthetic_gpu("Nonexistent GPU 9999", vram_override_gb=24)
        assert gpu.vram_bytes == 24 * _GiB


class TestFuzzySearch:
    def test_partial_name(self):
        """dbgpu fuzzy search should find GPU from partial name."""
        gpu = create_synthetic_gpu("GTX 1080")
        assert "1080" in gpu.name
        assert gpu.vendor == "nvidia"
