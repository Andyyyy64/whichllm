"""Lookup tables that runtimes read from disk a row at a time.

Qwen3.8-Flash-Next's n-gram table and DeepSeek V4.1's Engram tables used to
count as dense weights read every token and kept in memory. That put the
model at 61.6B active parameters instead of about 10B, so llama.cpp's row
could not fit 63 GB of RAM and was estimated at about 1 tok/s.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from whichllm.engine.compatibility import check_compatibility
from whichllm.engine.lookup import lazy_lookup_params
from whichllm.engine.performance import estimate_tok_per_sec
from whichllm.engine.vram import estimate_memory_breakdown
from whichllm.engines import load_cards
from whichllm.engines.select import engines_for
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.fetcher import dicts_to_models, models_to_dicts
from whichllm.models.gguf import _extract_gguf_variants
from whichllm.models.moe_fields import fill_moe_from_shape
from whichllm.models.shape import ModelShape, parse_model_shape
from whichllm.models.types import GGUFVariant, ModelInfo

CONFIGS = json.loads(
    (Path(__file__).parent / "fixtures" / "model_configs.json").read_text(
        encoding="utf-8"
    )
)
_GiB = 1024**3


def _shape(name: str) -> ModelShape:
    shape = parse_model_shape(CONFIGS[name])
    assert shape is not None
    return shape


def _flash_next(**overrides) -> ModelInfo:
    """ISTA-DASLab's GSQ-RCO repo of Qwen3.8-Flash-Next, as the index has it."""
    model = ModelInfo(
        id="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        family_id="qwen/qwen3.8-flash-next",
        name="Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        parameter_count=176_943_899_520,
        parameter_count_active=58_507_240_320,
        is_moe=True,
        shape=_shape("Qwen/Qwen3.8-Flash-Next"),
        gguf_variants=[
            GGUFVariant("Q2_0/x-00001-of-00002.gguf", "Q2_0", 66_400_000_000),
            GGUFVariant("IQ3_S/x-00001-of-00002.gguf", "IQ3_S", 83_600_000_000),
        ],
    )
    return dataclasses.replace(model, **overrides)


def _machine(vram_gb=16, ram_gb=63.3) -> HardwareInfo:
    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 5080",
        vendor="nvidia",
        vram_bytes=int(vram_gb * _GiB),
        compute_capability=(12, 0),
        memory_bandwidth_gbps=960,
    )
    return HardwareInfo(
        gpus=[gpu],
        has_avx2=True,
        ram_bytes=int(ram_gb * _GiB),
        ram_bandwidth_gbps=89.6,
        os="windows",
    )


def test_configs_declare_their_lookup_tables():
    # 20M n-gram rows x 2560, the 51.2B ISTA-DASLab counts.
    assert _shape("Qwen/Qwen3.8-Flash-Next").lookup_params == 51_200_000_000
    # Two Engram layers, 768M rows x 256: 196.6B, which the 111 GB IQ4_NL
    # Engram file holds at 4.5 bits.
    assert _shape("deepseek-ai/DeepSeek-V4.1-Flash-1017").lookup_params == (
        (384006168 + 384016682) * 256
    )
    assert _shape("Qwen/Qwen3-30B-A3B").lookup_params == 0


def test_the_lookup_size_survives_the_index_and_old_shapes_read_as_none():
    model = _flash_next()
    restored = dicts_to_models(models_to_dicts([model]))[0]

    assert restored.shape.lookup_params == 51_200_000_000
    old = {k: v for k, v in model.shape.to_dict().items() if k != "lookup_params"}
    assert ModelShape.from_dict(old).lookup_params == 0


def test_active_parameters_leave_the_table_out():
    model = _flash_next()

    fill_moe_from_shape(model)

    # 48 layers x 10 of 512 experts are read per token, not the n-gram table.
    assert model.parameter_count_active / 1e9 == pytest.approx(7.3, abs=0.1)


def test_resident_weights_match_the_shard_that_must_be_resident():
    model = _flash_next()
    fill_moe_from_shape(model)
    q2, iq3 = model.gguf_variants

    # ISTA-DASLab's table: shard 1 must be resident, 37.6 GB for Q2_0 and
    # 54.8 GB for IQ3_S; shard 2, the 28.8 GB n-gram table, stays on disk.
    assert estimate_memory_breakdown(model, q2).weights / 1e9 == pytest.approx(
        37.6, abs=0.1
    )
    assert estimate_memory_breakdown(model, iq3).weights / 1e9 == pytest.approx(
        54.8, abs=0.1
    )


def test_llama_cpp_runs_the_pack_from_ram_at_a_real_speed():
    model = _flash_next()
    fill_moe_from_shape(model)
    hardware = _machine()
    iq3 = model.gguf_variants[1]

    result = check_compatibility(model, iq3, hardware, 4096)
    speed = estimate_tok_per_sec(
        model,
        iq3,
        hardware.gpus[0],
        result.fit_type,
        placement=result.placement,
        ram_bandwidth_gbps=hardware.ram_bandwidth_gbps,
    )

    assert result.can_run
    assert result.placement.kind == "expert_offload"
    # Strata measured 53 tok/s on an RTX 5070 and says llama.cpp runs these
    # packs at about a third of its speed; it was about 1 tok/s before.
    assert 15 < speed < 53


def test_a_small_table_stays_resident():
    shape = dataclasses.replace(_shape("Qwen/Qwen3-30B-A3B"), lookup_params=2 * 10**9)
    model = ModelInfo(
        id="org/Small",
        family_id="org/small",
        name="Small",
        parameter_count=30 * 10**9,
        is_moe=True,
        shape=shape,
    )

    # 2B at 4.5 bits is under llama.cpp's 4 GiB lazy-loading threshold.
    assert lazy_lookup_params(model) == 0


def test_only_engines_that_read_tables_from_disk_run_it():
    model = _flash_next()
    fill_moe_from_shape(model)
    hardware = _machine()
    result = check_compatibility(model, model.gguf_variants[1], hardware, 4096)

    engines = [card.id for card in engines_for(result, hardware, load_cards())]

    # llama.cpp documents --lazy-mode; Ollama and LM Studio do not.
    assert engines == ["llama.cpp"]


def test_an_engram_file_is_not_a_build_of_the_model():
    data = {
        "siblings": [
            {
                "rfilename": "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                "size": 69_000_000_000,
            },
            {
                "rfilename": "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf",
                "size": 69_200_000_000,
            },
            {
                "rfilename": "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
                "size": 110_600_000_000,
            },
        ]
    }

    variants = _extract_gguf_variants(data, 763_000_000_000)

    assert [v.quant_type for v in variants] == ["IQ2_XXS"]
