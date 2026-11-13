"""vLLM and SGLang: which rows of Hugging Face weights they serve, and how."""

from pathlib import Path

import pytest

from whichllm.engine.placement import Placement
from whichllm.engine.ranker import rank_models
from whichllm.engine.recipe import build_recipe, build_server_recipe
from whichllm.engine.types import CompatibilityResult
from whichllm.engines import load_cards
from whichllm.engines.cards import CardError, parse_card
from whichllm.engines.detect import Installed
from whichllm.engines.select import assign_engines, engines_for
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output.json_output import _recipe_json

_GiB = 1024**3


def _hardware(
    vram_gb: float = 16,
    os_name: str = "linux",
    capability: tuple[int, int] = (12, 0),
    gpus: int = 1,
) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 5080",
                vendor="nvidia",
                vram_bytes=int(vram_gb * _GiB),
                compute_capability=capability,
                memory_bandwidth_gbps=960.0,
            )
            for _ in range(gpus)
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * _GiB,
        disk_free_bytes=500 * _GiB,
        os=os_name,
    )


def _model(repo: str, params_b: float = 8, **extra) -> ModelInfo:
    return ModelInfo(
        id=repo,
        family_id=repo,
        name=repo.split("/")[1],
        parameter_count=int(params_b * 1e9),
        downloads=1000,
        likes=100,
        **extra,
    )


def _row(model: ModelInfo, *, multi_gpu: bool = False, variant=None):
    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=6 * _GiB,
        vram_available_bytes=16 * _GiB,
        placement=Placement(kind="gpu", gpu_weight_bytes=5 * _GiB, cpu_weight_bytes=0),
        uses_multi_gpu=multi_gpu,
    )


def test_gpu_servers_serve_hugging_face_weights_and_vllm_is_picked_first():
    row = _row(_model("Qwen/Qwen3-8B-AWQ"))
    candidates = engines_for(row, _hardware(), load_cards())
    assert {card.id for card in candidates} == {"sglang", "vllm"}

    assign_engines([row], _hardware(), load_cards(), {})
    assert row.engine == "vllm"
    assert row.engine_installed is False


def test_an_installed_sglang_is_picked_over_vllm():
    row = _row(_model("Qwen/Qwen3-8B-AWQ"))
    installed = {
        "sglang": Installed(path="/usr/bin/sglang", version=None, running=False)
    }
    assign_engines([row], _hardware(), load_cards(), installed)
    assert row.engine == "sglang"
    assert row.engine_installed is True


def test_no_gpu_server_runs_on_native_windows_or_an_old_gpu():
    row = _row(_model("Qwen/Qwen3-8B-AWQ"))
    assert engines_for(row, _hardware(os_name="windows"), load_cards()) == []
    # Both need compute capability 7.5, a Turing card.
    assert engines_for(row, _hardware(capability=(6, 1)), load_cards()) == []


def test_mlx_weights_and_gguf_rows_get_no_gpu_server():
    mlx = _row(_model("mlx-community/Qwen3-8B-4bit", tags=("mlx",)))
    assert engines_for(mlx, _hardware(), load_cards()) == []

    variant = GGUFVariant(
        filename="m-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=5 * _GiB
    )
    gguf = _row(_model("org/M-GGUF"), variant=variant)
    ids = {card.id for card in engines_for(gguf, _hardware(), load_cards())}
    assert "vllm" not in ids and "sglang" not in ids
    assert "llama.cpp" in ids


def test_a_row_split_over_gpus_waits_for_tensor_parallelism():
    row = _row(_model("Qwen/Qwen3-32B-AWQ", 32), multi_gpu=True)
    assert engines_for(row, _hardware(gpus=2), load_cards()) == []


def test_the_run_line_serves_the_repo_and_names_the_other_server():
    row = _row(_model("Qwen/Qwen3-8B-AWQ"))
    assign_engines([row], _hardware(), load_cards(), {})

    assert build_recipe(row, 8192) is None
    recipe = build_server_recipe(row)
    assert recipe.command == "vllm serve Qwen/Qwen3-8B-AWQ --max-model-len auto"
    assert recipe.others == (("sglang", "sglang serve Qwen/Qwen3-8B-AWQ"),)

    as_json = _recipe_json(row, 8192)
    assert as_json["engine"] == "vllm"
    assert as_json["command"] == recipe.command
    assert as_json["quant"] == "AWQ"
    assert as_json["llama_cpp"] is None


def test_hugging_face_weights_that_need_ram_are_left_out():
    # 30B AWQ is about 21 GB: on a 16 GB GPU it would need the experts in RAM,
    # which vLLM and SGLang do not run at a usable speed.
    moe = _model(
        "Qwen/Qwen3-30B-A3B-AWQ",
        30.5,
        parameter_count_active=3_300_000_000,
        is_moe=True,
    )
    small = _model("Qwen/Qwen3-8B-AWQ")
    scores = {"Qwen/Qwen3-30B-A3B-AWQ": 80.0, "Qwen/Qwen3-8B-AWQ": 60.0}

    on_16 = rank_models([moe, small], _hardware(16), top_n=5, benchmark_scores=scores)
    assert [r.model.id for r in on_16] == ["Qwen/Qwen3-8B-AWQ"]
    assert on_16[0].fit_type == "full_gpu"

    on_32 = rank_models([moe, small], _hardware(32), top_n=5, benchmark_scores=scores)
    assert on_32[0].model.id == "Qwen/Qwen3-30B-A3B-AWQ"
    assert on_32[0].fit_type == "full_gpu"


def test_every_gpu_server_card_names_formats_it_reads(tmp_path):
    servers = [card for card in load_cards() if card.safetensors]
    assert {card.id for card in servers} == {"sglang", "vllm"}
    for card in servers:
        assert card.serve and "{repo}" in card.serve
        assert set(card.memory) == {"gpu"}
        assert card.platforms and all(p.os == ("linux",) for p in card.platforms)

    raw = {
        "id": "demo",
        "name": "Demo",
        "kind": "runner",
        "url": "https://example.com/demo",
        "evidence": "documented",
        "upstream": {"status": "active"},
        "checked_at": "2026-11-13",
        "sources": ["https://example.com/demo"],
        "platforms": [{"os": ["linux"], "accel": "cuda"}],
        "safetensors": ["AWQ", "EXL3"],
    }
    with pytest.raises(CardError, match="'safetensors'"):
        parse_card(raw, Path("demo.json"))
