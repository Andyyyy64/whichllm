"""Choosing the engine for a ranked row, and the recipe that follows from it."""

import json
from io import StringIO

import pytest
from rich.console import Console

from whichllm.engine.placement import Placement
from whichllm.engine.recipe import build_recipe
from whichllm.engine.types import CompatibilityResult
from whichllm.engines import load_cards
from whichllm.engines.detect import Installed
from whichllm.engines.select import (
    assign_engines,
    engines_for,
    pick_engine,
    reads_quant,
)
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3
CARDS = load_cards()
BY_ID = {card.id: card for card in CARDS}


def _row(quant="Q4_K_M", kind="gpu", repo="org/Model-GGUF") -> CompatibilityResult:
    model = ModelInfo(
        id=repo,
        family_id=repo,
        name="Model",
        parameter_count=8_000_000_000,
        gguf_variants=[GGUFVariant(f"Model-{quant}.gguf", quant, 5 * _GiB)],
    )
    placement = Placement(
        kind,
        gpu_weight_bytes=5 * _GiB,
        cpu_weight_bytes=0 if kind == "gpu" else 2 * _GiB,
        n_cpu_moe=12 if kind == "expert_offload" else None,
        num_layers=36,
    )
    return CompatibilityResult(
        model=model,
        gguf_variant=model.gguf_variants[0],
        can_run=True,
        vram_required_bytes=5 * _GiB,
        vram_available_bytes=16 * _GiB,
        placement=placement,
        fit_type="full_gpu" if kind == "gpu" else "partial_offload",
    )


def _rtx() -> HardwareInfo:
    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 5080",
        vendor="nvidia",
        vram_bytes=16 * _GiB,
        compute_capability=(12, 0),
    )
    return HardwareInfo(gpus=[gpu], has_avx2=True, ram_bytes=64 * _GiB, os="linux")


@pytest.mark.parametrize(
    ("engine", "quant", "reads"),
    [
        ("llama.cpp", "Q4_K_M", True),
        ("llama.cpp", "IQ4_KS", False),
        ("ollama", "IQ4_KS", False),
        ("ik_llama.cpp", "IQ4_KS", True),
        ("ik_llama.cpp", "IQ4_KS_R4", True),
        ("ik_llama.cpp", "Q4_K_M", True),
        ("ik_llama.cpp", "NVFP4", True),
        ("ik_llama.cpp", "TQ2_0", False),
        ("llama.cpp", "NVFP4", True),
    ],
)
def test_who_reads_which_quant(engine, quant, reads):
    assert reads_quant(BY_ID[engine], quant, CARDS) is reads


def test_a_plain_gguf_runs_on_every_engine_here():
    ids = {card.id for card in engines_for(_row(), _rtx(), CARDS)}

    assert ids == {
        "llama.cpp",
        "ik_llama.cpp",
        "ollama",
        "lm-studio",
        "koboldcpp",
        "jan",
        "lemonade",
    }


def test_expert_offload_leaves_out_engines_that_cannot_hold_it():
    ids = {card.id for card in engines_for(_row(kind="expert_offload"), _rtx(), CARDS)}

    assert "lemonade" not in ids
    assert "llama.cpp" in ids


def test_an_ik_quant_runs_only_on_ik_llama_cpp():
    candidates = engines_for(_row(quant="IQ4_KS"), _rtx(), CARDS)

    assert [card.id for card in candidates] == ["ik_llama.cpp"]
    assert pick_engine(candidates, {}).id == "ik_llama.cpp"


def test_without_installs_llama_cpp_is_the_recipe():
    assert pick_engine(engines_for(_row(), _rtx(), CARDS), {}).id == "llama.cpp"


def test_an_installed_engine_comes_first_and_a_running_one_before_it():
    candidates = engines_for(_row(), _rtx(), CARDS)
    ollama = {"ollama": Installed("/usr/bin/ollama", None, running=False)}
    both = {
        "llama.cpp": Installed("/usr/bin/llama-server", None, running=False),
        "lm-studio": Installed(None, "0.4.28", running=True),
    }

    assert pick_engine(candidates, ollama).id == "ollama"
    assert pick_engine(candidates, both).id == "lm-studio"


def test_jan_and_lemonade_are_never_the_recipe():
    candidates = [BY_ID["jan"], BY_ID["lemonade"]]

    assert (
        pick_engine(candidates, {"jan": Installed("/usr/bin/jan", None, False)}) is None
    )


def test_assign_engines_records_the_choice_and_the_binary():
    row = _row()
    installed = {"llama.cpp": Installed("/home/me/.local/bin/llama", "b9986", False)}

    assign_engines([row], _rtx(), CARDS, installed)

    assert row.engine == "llama.cpp"
    assert row.engine_installed is True
    assert row.engine_binary == "llama"
    assert "ollama" in row.engines


def test_a_simulated_machine_leaves_installed_unknown():
    row = _row()

    assign_engines([row], _rtx(), CARDS, None)

    assert row.engine == "llama.cpp"
    assert row.engine_installed is None


def test_recipes_follow_the_engine():
    row = _row(quant="IQ4_KS")
    row.engine = "ik_llama.cpp"
    assert build_recipe(row, 4096).command == (
        "llama-server -hfr org/Model-GGUF -hff Model-IQ4_KS.gguf -c 4096 -ngl 99 -fa on"
    )

    row = _row()
    row.engine = "ollama"
    assert build_recipe(row, 4096).command == "ollama run hf.co/org/Model-GGUF:Q4_K_M"

    row = _row()
    row.engine, row.engine_binary = "llama.cpp", "llama"
    recipe = build_recipe(row, 4096)
    assert (
        recipe.command == "llama serve -hf org/Model-GGUF:Q4_K_M -c 4096 -ngl 99 -fa on"
    )
    assert recipe.llama_cpp == recipe.command


def _printed(row, monkeypatch) -> str:
    from whichllm.output import _console as console_mod
    from whichllm.output import ranking

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=200))
    monkeypatch.setattr(ranking.platform, "system", lambda: "Linux")
    ranking._print_recipe(row, 4096)
    return buf.getvalue()


def test_the_run_line_names_an_installed_engine(monkeypatch):
    row = _row()
    row.engine, row.engine_installed = "ollama", True

    shown = _printed(row, monkeypatch)

    assert "Run #1 with Ollama (installed):" in shown
    assert "ollama run hf.co/org/Model-GGUF:Q4_K_M" in shown
    assert "llama.cpp: llama-server -hf org/Model-GGUF:Q4_K_M" in shown
    assert "not installed" not in shown


def test_a_missing_engine_comes_with_its_install_command(monkeypatch):
    row = _row()
    row.engine, row.engine_installed = "llama.cpp", False

    shown = _printed(row, monkeypatch)

    assert "Run #1 with llama.cpp:" in shown
    assert (
        "llama.cpp is not installed here; install with "
        "curl -LsSf https://llama.app/install.sh | sh"
    ) in shown


def test_json_rows_carry_the_engine(monkeypatch):
    from whichllm.output import _console as console_mod
    from whichllm.output.json_output import display_json

    row = _row()
    assign_engines([row], _rtx(), CARDS, {"ollama": Installed(None, "0.40.14", True)})
    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=10_000))

    display_json([row], _rtx(), 4096)

    data = json.loads(buf.getvalue())["models"][0]
    assert data["engine"] == "ollama"
    assert data["engine_installed"] is True
    assert "llama.cpp" in data["engines"]
    assert data["recipe"]["engine"] == "ollama"
    assert data["recipe"]["command"] == "ollama run hf.co/org/Model-GGUF:Q4_K_M"
