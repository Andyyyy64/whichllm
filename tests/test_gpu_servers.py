"""vLLM and SGLang: which rows of Hugging Face weights they serve, and how."""

from dataclasses import replace
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
from whichllm.models.shape import ModelShape
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output.json_output import _recipe_json

_GiB = 1024**3


def _hardware(
    vram_gb: float = 16,
    os_name: str = "linux",
    capability: tuple[int, int] = (12, 0),
    gpus: int = 1,
    *,
    vram_list: tuple[float, ...] | None = None,
    ram_gb: float = 64,
    wsl: bool = False,
) -> HardwareInfo:
    sizes = vram_list or (vram_gb,) * gpus
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name=f"NVIDIA GeForce RTX {size:g} GB",
                vendor="nvidia",
                vram_bytes=int(size * _GiB),
                compute_capability=capability,
                memory_bandwidth_gbps=960.0,
            )
            for size in sizes
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=int(ram_gb * _GiB),
        disk_free_bytes=500 * _GiB,
        os=os_name,
        wsl=wsl,
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


def test_a_row_split_over_gpus_without_tensor_parallelism_gets_no_server():
    # The fit check splits Hugging Face weights only with tensor parallelism.
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
        assert card.tensor_parallel and "{size}" in card.tensor_parallel
        assert card.platforms and all(p.os == ("linux",) for p in card.platforms)
    by_id = {card.id: card for card in servers}
    assert set(by_id["vllm"].memory) == {"gpu", "tensor_parallel"}
    assert set(by_id["sglang"].memory) == {"gpu", "tensor_parallel", "paged_experts"}
    paged = by_id["sglang"].paged_experts
    assert "BNB_4BIT" not in paged.formats and "INT8" not in paged.formats

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

    raw["safetensors"] = ["AWQ"]
    raw["memory"] = ["gpu", "tensor_parallel"]
    with pytest.raises(CardError, match="lists tensor_parallel"):
        parse_card(raw, Path("demo.json"))
    raw["tensor_parallel"] = "--tp"
    with pytest.raises(CardError, match="takes {size}"):
        parse_card(raw, Path("demo.json"))
    raw["tensor_parallel"] = "--tp {size}"
    raw["memory"] = ["gpu", "tensor_parallel", "paged_experts"]
    raw["paged_experts"] = {"flags": "--paged {resident}", "formats": ["FP8"]}
    with pytest.raises(CardError, match="formats are some"):
        parse_card(raw, Path("demo.json"))
    raw["paged_experts"]["formats"] = ["AWQ"]
    assert (
        parse_card(raw, Path("demo.json")).paged_experts.flags == "--paged {resident}"
    )


# Qwen3-30B-A3B's layout: 48 layers of 128 experts, 8 of them per token.
_QWEN3_MOE = ModelShape(
    model_type="qwen3_moe",
    num_layers=48,
    full_kv_bytes_per_token=98_304,
    hidden_size=2048,
    intermediate_size=6144,
    num_experts=128,
    experts_per_token=8,
    moe_intermediate_size=768,
)
# gpt-oss-20b's: 24 layers of 32 experts, 4 per token.
_GPT_OSS = ModelShape(
    model_type="gpt_oss",
    num_layers=24,
    full_kv_bytes_per_token=49_152,
    hidden_size=2880,
    intermediate_size=2880,
    num_experts=32,
    experts_per_token=4,
    moe_intermediate_size=2880,
)


def _moe(repo: str, params_b: float = 30.5, active_b: float = 3.3, shape=_QWEN3_MOE):
    return _model(
        repo,
        params_b,
        parameter_count_active=int(active_b * 1e9),
        is_moe=True,
        shape=shape,
    )


def _llama_70b_awq() -> ModelInfo:
    # The repository's real size, as the index gives it.
    return _model("org/Llama-3.3-70B-Instruct-AWQ", 70.6, weight_bytes=39_800_000_000)


def _ranked(models, hardware):
    scores = {model.id: 80.0 for model in models}
    rows = rank_models(models, hardware, top_n=5, benchmark_scores=scores)
    assign_engines(rows, hardware, load_cards(), {})
    return rows


def test_a_model_past_one_gpu_is_split_evenly_over_four():
    model = _llama_70b_awq()
    rows = _ranked([model], _hardware(24, gpus=4))
    assert len(rows) == 1
    row = rows[0]
    assert row.fit_type == "full_gpu"
    assert row.uses_multi_gpu and row.tensor_parallel_size == 4
    shares = {device.weight_bytes for device in row.placement.devices}
    assert len(row.placement.devices) == 4 and len(shares) == 1

    assert row.engines == ("sglang", "vllm") and row.engine == "vllm"
    recipe = build_server_recipe(row)
    assert recipe.command == (
        "vllm serve org/Llama-3.3-70B-Instruct-AWQ --max-model-len auto "
        "--tensor-parallel-size 4"
    )
    assert recipe.others == (
        ("sglang", "sglang serve org/Llama-3.3-70B-Instruct-AWQ --tp-size 4"),
    )
    assert recipe.notes == (
        "Splits the weights evenly over 4 GPUs with tensor parallelism.",
    )
    assert _recipe_json(row, 4096)["notes"] == list(recipe.notes)
    assert any("Tensor parallelism reads" in note for note in row.speed_notes)


def test_three_gpus_split_over_two_and_a_small_card_bounds_the_split():
    model = _llama_70b_awq()
    rows = _ranked([model], _hardware(24, gpus=3))
    assert rows[0].tensor_parallel_size == 2
    assert len(rows[0].placement.devices) == 2

    # Each card holds half, which the 12 GB card cannot; a layer split could.
    assert _ranked([model], _hardware(vram_list=(24, 12))) == []


def test_an_moe_past_vram_runs_with_paged_experts_on_one_gpu():
    model = _moe("Qwen/Qwen3-30B-A3B-GPTQ-Int4")
    rows = _ranked([model], _hardware(16))
    assert len(rows) == 1
    row = rows[0]
    assert row.fit_type == "partial_offload"
    assert row.placement.kind == "expert_offload"
    assert 8 <= row.paged_experts_resident < 128
    assert row.engines == ("sglang",) and row.engine == "sglang"
    assert any("pages the rest in" in warning for warning in row.warnings)
    assert any("SGLang pages experts in" in note for note in row.speed_notes)

    recipe = build_server_recipe(row)
    assert recipe.command == (
        "sglang serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 --enable-paged-experts "
        f"--paged-experts-num-resident {row.paged_experts_resident} "
        "--mem-fraction-static 0.88"
    )
    assert recipe.others == ()
    assert recipe.notes[0].startswith(
        f"Keeps {row.paged_experts_resident} of 128 experts per MoE layer"
    )


def test_paged_experts_add_the_flags_a_format_or_shared_experts_need():
    mxfp4 = _moe("org/Moe-20B-MXFP4", 21, 3.6, _GPT_OSS)
    row = _ranked([mxfp4], _hardware(8))[0]
    assert row.paged_experts_resident
    assert build_server_recipe(row).command.endswith(
        f"--paged-experts-num-resident {row.paged_experts_resident} "
        "--mem-fraction-static 0.88 --moe-runner-backend marlin"
    )

    shared = replace(_QWEN3_MOE, shared_expert_intermediate_size=4096)
    row = _ranked([_moe("org/Moe-30B-A3B-AWQ", shape=shared)], _hardware(16))[0]
    assert build_server_recipe(row).command.endswith("--disable-shared-experts-fusion")


def test_paged_experts_need_a_format_they_page_and_every_expert_pinned():
    # bitsandbytes experts do not page; the model's GGUF covers them.
    assert _ranked([_moe("org/Moe-30B-A3B-bnb-4bit")], _hardware(16)) == []
    # Without the layout, K cannot be worked out.
    no_shape = _moe("Qwen/Qwen3-30B-A3B-GPTQ-Int4", shape=None)
    assert _ranked([no_shape], _hardware(16)) == []

    # About 16 GB of experts: 24 GB of RAM pins them, but WSL pins about half.
    model = _moe("Qwen/Qwen3-30B-A3B-GPTQ-Int4")
    assert _ranked([model], _hardware(16, ram_gb=24))[0].paged_experts_resident
    assert _ranked([model], _hardware(16, ram_gb=24, wsl=True)) == []


def test_paged_experts_run_on_one_card_of_several():
    # Too big for an even split over a 16 and an 8 GB card: one card pages.
    model = _moe("Qwen/Qwen3-30B-A3B-GPTQ-Int4")
    row = _ranked([model], _hardware(vram_list=(16, 8)))[0]
    assert not row.uses_multi_gpu and row.tensor_parallel_size is None
    assert row.paged_experts_resident and row.engine == "sglang"
