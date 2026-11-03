"""Fixed-model engines: their packs, and which memory tier each runs from here."""

import copy
import json
from pathlib import Path

import pytest

from whichllm.engines import load_cards
from whichllm.engines.cards import ENGINE_DIR, CardError, parse_card
from whichllm.engines.packs import pack_fit, pack_options
from whichllm.hardware.types import GPUInfo, HardwareInfo

_GiB = 1024**3
BY_ID = {card.id: card for card in load_cards()}


def _pack(engine: str, model: str, pack: str):
    found = next(m for m in BY_ID[engine].models if m.id == model)
    return next(p for p in found.packs if p.id == pack)


def _machine(
    vram_gb=16, ram_gb=64, cc=(12, 0), vendor="nvidia", os="windows", disk_gb=0
):
    gpus = []
    if vram_gb:
        gpus.append(
            GPUInfo(
                name="GPU",
                vendor=vendor,
                vram_bytes=int(vram_gb * _GiB),
                compute_capability=cc if vendor == "nvidia" else None,
            )
        )
    return HardwareInfo(
        gpus=gpus,
        has_avx2=True,
        ram_bytes=int(ram_gb * _GiB),
        disk_free_bytes=int(disk_gb * _GiB),
        os=os,
    )


def _raw(card_id: str) -> dict:
    return json.loads((ENGINE_DIR / f"{card_id}.json").read_text(encoding="utf-8"))


def test_fixed_model_cards_list_models_and_runners_do_not():
    for card in load_cards():
        if card.kind == "fixed_model":
            assert card.models, card.id
            for model in card.models:
                assert "/" in model.base
                assert model.packs
        else:
            assert card.models == (), card.id


def test_strata_and_maya_carry_their_models():
    strata = {m.base for m in BY_ID["strata"].models}
    maya = {m.base for m in BY_ID["project-maya"].models}

    assert strata == {
        "Qwen/Qwen3.8-Flash-Next",
        "Qwen/Qwen3.6-35B-A3B",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    }
    assert maya == {
        "zai-org/GLM-5.3-Flash",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "MiniMaxAI/MiniMax-M3.1-Flash",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("models"), "needs at least one model"),
        (lambda d: d["models"][0]["packs"][0]["requires"].pop("disk_gb"), "disk_gb"),
        (
            lambda d: d["models"][0]["packs"][0].update(
                requires={"disk_gb": 10, "vram_gb": 12}
            ),
            "says no memory it runs with",
        ),
        (
            lambda d: d["models"][0]["packs"][0]["requires"].update(ram_gb=-1),
            "positive number",
        ),
        (
            lambda d: d["models"][0]["packs"].append(
                copy.deepcopy(d["models"][0]["packs"][0])
            ),
            "repeats a pack id",
        ),
        (lambda d: d["models"][0].update(base="not a repo"), "owner/repo"),
        (lambda d: d["models"][0]["packs"][0].update(size=1), "unknown key"),
    ],
)
def test_a_broken_fixed_model_card_is_refused(change, message):
    data = _raw("strata")
    change(data)

    with pytest.raises(CardError, match=message):
        parse_card(data, Path("strata.json"))


def test_a_runner_cannot_list_models():
    data = _raw("llama.cpp")
    data["models"] = _raw("strata")["models"]

    with pytest.raises(CardError, match="only a fixed_model card"):
        parse_card(data, Path("llama.cpp.json"))


def test_qwen_q2_0_fits_in_memory_with_enough_ram():
    fit = pack_fit(_pack("strata", "qwen3.8-flash-next", "q2_0"), _machine())

    assert fit.tier == "memory"
    assert fit.reason is None


def test_short_ram_reads_experts_from_the_ssd_only_with_the_documented_gpu():
    pack = _pack("strata", "qwen3.8-flash-next", "q2_0")

    big_card = pack_fit(pack, _machine(vram_gb=24, ram_gb=32))
    small_card = pack_fit(pack, _machine(vram_gb=16, ram_gb=39))

    assert big_card.tier == "ssd"
    assert "from the SSD" in big_card.reason
    assert small_card.tier is None
    assert small_card.reason == "needs 44 GB of RAM, or 32 GB with a 24 GB GPU"


def test_too_little_ram_says_how_much_is_needed():
    fit = pack_fit(_pack("strata", "qwen3.8-flash-next", "iq3_s"), _machine(ram_gb=48))

    assert fit.reason == "needs 60 GB of RAM, this machine has 48 GB"


def test_a_small_gpu_is_refused():
    fit = pack_fit(_pack("strata", "qwen3.8-flash-next", "q2_0"), _machine(vram_gb=8))

    assert fit.reason == "needs a GPU with 12 GB, the largest here has 8 GB"


def test_too_little_free_disk_is_refused():
    fit = pack_fit(_pack("strata", "qwen3.8-flash-next", "q2_0"), _machine(disk_gb=50))

    assert fit.reason == "needs 72.4 GB of disk, 50 GB free"


def test_ram_and_vram_together_hold_a_maya_pack():
    pack = _pack("project-maya", "glm-5.3-flash", "glm-maya-s")

    assert pack_fit(pack, _machine(vram_gb=24, ram_gb=96)).tier == "memory"
    assert pack_fit(pack, _machine(vram_gb=16, ram_gb=64)).tier == "ssd"
    assert pack_fit(pack, _machine(vram_gb=16, ram_gb=16)).reason == (
        "needs 32 GB of RAM, this machine has 16 GB"
    )


def test_unified_memory_counts_once():
    pack = _pack("project-maya", "glm-5.3-flash", "glm-maya-s")
    big_mac = _machine(vram_gb=96, ram_gb=128, vendor="apple", os="darwin")
    # 64 GB of RAM and a 48 GB GPU would hold the 96.5 GB pack, but on a Mac
    # the GPU's 48 GB are part of the 64.
    mac = _machine(vram_gb=48, ram_gb=64, vendor="apple", os="darwin")

    assert pack_fit(pack, big_mac).tier == "memory"
    assert pack_fit(pack, mac).tier == "ssd"


def test_a_ram_budget_caps_the_ram_a_pack_can_use():
    pack = _pack("strata", "qwen3.8-flash-next", "q2_0")
    machine = _machine()
    machine.ram_budget_bytes = 40 * _GiB

    assert pack_fit(pack, machine).tier is None


def test_options_follow_each_models_platforms():
    options = pack_options(load_cards(), _machine(cc=(7, 5), vram_gb=16))
    by_key = {(o.card.id, o.model.id, o.pack.id): o for o in options}

    qwen = by_key[("strata", "qwen3.8-flash-next", "q2_0")]
    deepseek = by_key[("strata", "deepseek-v4.1-flash-1017", "ist-das-2.00bpw")]

    assert qwen.fit.tier == "memory"
    assert qwen.platform.accel == "cuda"
    assert deepseek.fit.tier is None
    assert deepseek.fit.reason == ("needs CUDA compute capability 8.6, this GPU is 7.5")


def test_maya_has_no_mac_build_outside_glm():
    options = pack_options(
        load_cards(), _machine(vram_gb=96, ram_gb=128, vendor="apple", os="darwin")
    )
    maya = {o.model.id: o for o in options if o.card.id == "project-maya"}

    assert maya["glm-5.3-flash"].fit.runs
    assert maya["minimax-m3.1-flash"].fit.reason == "no macOS build"


def test_maya_does_not_run_under_wsl():
    linux = _machine(vram_gb=24, ram_gb=128, os="linux")
    under_wsl = {o.card.id: o.fit for o in pack_options(load_cards(), linux, wsl=True)}
    on_linux = {o.card.id: o.fit for o in pack_options(load_cards(), linux)}

    assert under_wsl["project-maya"].reason == "does not run under WSL"
    assert under_wsl["strata"].runs
    assert on_linux["project-maya"].runs


def test_ds4_needs_a_gpu_that_shares_system_ram():
    pack = _pack("ds4", "qwen3.8-flash-next", "qwen38-q2")
    desktop = _machine(vram_gb=24, ram_gb=128, os="linux")
    mac = _machine(vram_gb=48, ram_gb=64, vendor="apple", os="darwin")

    assert pack_fit(pack, desktop).reason == (
        "needs a GPU that shares system RAM (a Mac, DGX Spark or Strix Halo)"
    )
    assert pack_fit(pack, mac).tier == "memory"


def test_the_engines_table_counts_the_packs_that_fit():
    from whichllm.engines import availability
    from whichllm.engines.packs import with_packs

    desktop = _machine(vram_gb=16, ram_gb=64, os="linux")
    ds4 = with_packs(availability(BY_ID["ds4"], desktop), desktop)
    strata = with_packs(availability(BY_ID["strata"], desktop), desktop)
    total = sum(len(m.packs) for m in BY_ID["strata"].models)

    assert not ds4.runs
    assert ds4.reason == (
        "no pack fits, needs a GPU that shares system RAM (a Mac, DGX Spark or "
        "Strix Halo)"
    )
    assert strata.runs
    assert strata.reason.endswith(f" of {total} packs fit")
