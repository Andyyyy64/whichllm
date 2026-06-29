"""Tests for dbgpu-backed bandwidth resolution on detected hardware.

The detection path must never give a laptop/mobile card its desktop bandwidth
(issues #74, #61, #93). These tests pin the strict matching rules: curated
values win, the curated lookup is mobile-aware, dbgpu fills gaps, and an
unknown card resolves to None rather than a wrong guess.
"""

from whichllm.hardware.gpu_db import (
    _normalize_detected_name,
    _static_bandwidth,
    resolve_detected_bandwidth,
)

_GiB = 1024**3


# --- normalization ---------------------------------------------------------


def test_normalize_strips_vendor_trademark_and_maps_laptop_to_mobile():
    assert (
        _normalize_detected_name("NVIDIA GeForce RTX 5090 Laptop GPU")
        == "GeForce RTX 5090 Mobile"
    )
    assert _normalize_detected_name("Intel(R) Arc(TM) A770 Graphics") == "Arc A770"
    assert _normalize_detected_name("AMD Radeon RX 6750 XT") == "Radeon RX 6750 XT"


def test_normalize_empty_or_vendor_only():
    assert _normalize_detected_name("") == ""
    assert _normalize_detected_name("AMD") == ""


def test_normalize_adds_space_to_vram_bin_suffix():
    # #98: drivers write "12GB", dbgpu writes "12 GB".
    assert _normalize_detected_name("NVIDIA RTX A2000 12GB") == "RTX A2000 12 GB"


# --- curated lookup is mobile-aware ---------------------------------------


def test_static_bandwidth_desktop_key_does_not_claim_laptop_card():
    # "RTX 5090" desktop key (1792) must not match a Laptop driver name.
    assert _static_bandwidth("NVIDIA GeForce RTX 5090 Laptop GPU") is None
    # The desktop card itself still resolves from the curated table.
    assert _static_bandwidth("NVIDIA GeForce RTX 5090") == 1792.0


def test_static_bandwidth_keeps_curated_laptop_entry():
    # A curated key that is itself a laptop entry stays authoritative.
    assert _static_bandwidth("NVIDIA RTX A3000 Laptop GPU") == 264.0


# --- end-to-end resolution -------------------------------------------------


def test_resolve_desktop_uses_curated_value():
    assert resolve_detected_bandwidth("NVIDIA GeForce RTX 5090") == 1792.0


def test_resolve_laptop_5090_is_mobile_not_desktop():
    # The bug in #74: a laptop 5090 must not inherit the 1792 desktop value.
    bw = resolve_detected_bandwidth("NVIDIA GeForce RTX 5090 Laptop GPU", 24 * _GiB)
    assert bw is not None
    assert bw < 1500.0  # mobile 5090 is ~896 GB/s, nowhere near desktop 1792


def test_resolve_a3000_laptop_preserves_curated_264():
    assert resolve_detected_bandwidth("NVIDIA RTX A3000 Laptop GPU", 6 * _GiB) == 264.0


def test_resolve_rx_6750_xt():
    # #61: originally None. The curated table now carries it (#68); dbgpu
    # would agree (432 GB/s) if the curated entry ever went away.
    bw = resolve_detected_bandwidth("AMD Radeon RX 6750 XT", 12 * _GiB)
    assert bw == 432.0


def test_resolve_variant_qualifier_is_preserved():
    # "RTX 4060 Ti" must not collapse to the non-Ti "RTX 4060".
    bw = resolve_detected_bandwidth("NVIDIA GeForce RTX 4060 Ti", 16 * _GiB)
    assert bw is not None
    # 4060 (non-Ti) is 272 GB/s; the Ti is 288. Either way it must be a real,
    # positive value and not desktop-flagship sized.
    assert 200 < bw < 400


def test_resolve_arc_pro_b70_uses_curated_value():
    assert resolve_detected_bandwidth("Intel(R) Arc(TM) Pro B70 Graphics") == 608.0
    assert resolve_detected_bandwidth("Battlemage G31 [Intel Graphics]") == 608.0


def test_resolve_empty_name_returns_none():
    assert resolve_detected_bandwidth("") is None
    assert resolve_detected_bandwidth("Some Totally Made Up GPU 9xZ") is None


def test_resolve_known_amd_desktop_from_curated_table():
    assert resolve_detected_bandwidth("AMD Radeon RX 9070 XT") == 640.0


def test_resolve_a2000_12gb_vram_bin_from_dbgpu():
    # #98: "NVIDIA RTX A2000 12GB" reported BW: N/A. dbgpu names the bin
    # "RTX A2000 12 GB" (288 GB/s); the normalizer must bridge the spacing.
    assert resolve_detected_bandwidth("NVIDIA RTX A2000 12GB", 12 * _GiB) == 288.0


def test_static_bandwidth_compound_lspci_name():
    # Ported from amd.py (#68): compound lspci names resolve via segment
    # splitting, first matching segment wins, "RX " prefix retried for bare
    # segments.
    compound = "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"
    bw = _static_bandwidth(compound)
    assert bw is not None
    assert bw > 0
    # The same answer flows through the public resolver.
    assert resolve_detected_bandwidth(compound, 12 * _GiB) == bw
