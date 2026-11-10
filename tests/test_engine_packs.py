"""Fixed-model engines: their packs, and which memory tier each runs from here."""

import copy
import json
import re
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
        "zai-org/GLM-5.3",
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
        (lambda d: d["upstream"].update(track="patch"), "'track' is 'patch'"),
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


def test_a_gpu_a_little_under_its_nominal_size_counts_as_that_size():
    pack = _pack("strata", "qwen3.8-flash-next", "q2_0")
    # An RTX 5070 "12 GB" reports 11.9 GiB, an RTX 4090 "24 GB" 23.99.
    assert pack_fit(pack, _machine(vram_gb=11.94)).tier == "memory"
    assert pack_fit(pack, _machine(vram_gb=23.99, ram_gb=32)).tier == "ssd"
    assert pack_fit(pack, _machine(vram_gb=11)).tier is None


def test_ram_a_little_under_its_nominal_size_counts_as_that_size():
    maya = _pack("project-maya", "glm-5.3-flash", "glm-maya-s")
    coder = _pack("strata", "qwen3.8-flash-next", "coder-iq1_m")
    # A "32 GB" PC reports about 31.2 GiB, and Maya runs from 32 GB of RAM.
    assert pack_fit(maya, _machine(vram_gb=16, ram_gb=31.2)).tier == "ssd"
    assert pack_fit(maya, _machine(vram_gb=16, ram_gb=24)).reason == (
        "needs 32 GB of RAM, this machine has 24 GB"
    )
    # Strata's Coder needs 33 GB of RAM, more than a 32 GB PC has even on paper,
    # but it reads from the SSD with 32.
    assert pack_fit(coder, _machine(vram_gb=12, ram_gb=31.2)).tier == "ssd"
    # A 24 GB GPU with 64 GB is the PC Maya's docs give for Maya-XS, 88 GB together.
    xs = _pack("project-maya", "glm-5.3-flash", "glm-maya-xs")
    assert pack_fit(xs, _machine(vram_gb=23.99, ram_gb=62.7)).tier == "memory"
    assert pack_fit(xs, _machine(vram_gb=15.9, ram_gb=62.7)).tier == "ssd"


def test_too_little_free_disk_is_refused():
    fit = pack_fit(_pack("strata", "qwen3.8-flash-next", "q2_0"), _machine(disk_gb=50))

    assert fit.reason == "needs 72.4 GB of disk, 50 GB free"


def test_ram_and_vram_together_hold_a_maya_pack():
    pack = _pack("project-maya", "glm-5.3-flash", "glm-maya-s")

    # Maya's docs recommend 128 GB of VRAM and RAM together for Maya-S.
    assert pack_fit(pack, _machine(vram_gb=32, ram_gb=96)).tier == "memory"
    assert pack_fit(pack, _machine(vram_gb=24, ram_gb=96)).tier == "ssd"
    assert pack_fit(pack, _machine(vram_gb=16, ram_gb=64)).tier == "ssd"
    assert pack_fit(pack, _machine(vram_gb=16, ram_gb=16)).reason == (
        "needs 32 GB of RAM, this machine has 16 GB"
    )


def test_unified_memory_counts_once():
    pack = _pack("project-maya", "glm-5.3-flash", "glm-maya-s")
    big_mac = _machine(vram_gb=96, ram_gb=128, vendor="apple", os="darwin")
    # 96 GB of RAM and a 48 GB GPU would hold the 128 GB the pack needs, but
    # on a Mac the GPU's 48 GB are part of the 96.
    mac = _machine(vram_gb=48, ram_gb=96, vendor="apple", os="darwin")
    pc = _machine(vram_gb=48, ram_gb=96)

    assert pack_fit(pack, big_mac).tier == "memory"
    assert pack_fit(pack, pc).tier == "memory"
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


def test_maya_has_a_mac_build_for_glm_flash_and_minimax_only():
    options = pack_options(
        load_cards(), _machine(vram_gb=96, ram_gb=128, vendor="apple", os="darwin")
    )
    maya = {o.model.id: o for o in options if o.card.id == "project-maya"}

    assert maya["glm-5.3-flash"].fit.runs
    assert maya["minimax-m3.1-flash"].fit.runs
    assert maya["glm-5.3"].fit.reason == "no macOS build"
    assert maya["deepseek-v4.1-flash-1017"].fit.reason == "no macOS build"


_NAMED_PC_RE = re.compile(r"-(\d+)G(\d+)G-GGUF$")


def _abyss_packs_named_for_a_pc():
    for model in BY_ID["abyss"].models:
        for pack in model.packs:
            named = _NAMED_PC_RE.search(pack.repo)
            if named:
                yield pytest.param(
                    model.id,
                    pack,
                    int(named.group(1)),
                    int(named.group(2)),
                    id=f"{model.id}:{pack.id}",
                )


@pytest.mark.parametrize(
    ("model", "pack", "gpu_gb", "ram_gb"), _abyss_packs_named_for_a_pc()
)
def test_an_abyss_pack_runs_on_the_pc_it_is_named_for(model, pack, gpu_gb, ram_gb):
    # What such a PC reports: a "64 GB" one has about 62.5 GiB, a "16 GB" GPU 15.8.
    named = _machine(vram_gb=gpu_gb - 0.2, ram_gb=ram_gb * 0.975, os="linux")
    smaller = _machine(vram_gb=gpu_gb - 0.2, ram_gb=ram_gb * 0.7, os="linux")

    fit = pack_fit(pack, named)

    # V4.1's experts do not fit the 32 GB + 96 GB PC its packs are named for.
    assert fit.tier == ("ssd" if model == "deepseek-v4.1-flash-1017" else "memory")
    assert pack_fit(pack, smaller).tier != "memory"


def test_abyss_names_the_setup_a_v41_pack_was_measured_on():
    pack = _pack("abyss", "deepseek-v4.1-flash-1017", "132G")

    assert pack_fit(pack, _machine(vram_gb=24, ram_gb=128)).reason == (
        "needs 84 GB of RAM with a 32 GB GPU"
    )
    assert pack_fit(pack, _machine(vram_gb=32, ram_gb=64)).reason == (
        "needs 84 GB of RAM, this machine has 64 GB"
    )


def test_a_pool_figure_is_not_called_ram_alone():
    pack = _pack("abyss", "glm-5.3-flash", "32G128G")

    assert pack_fit(pack, _machine(vram_gb=24, ram_gb=94)).reason == (
        "needs 155 GB of RAM and VRAM together, or 84 GB of RAM with a 32 GB GPU"
    )


def test_abyss_reads_its_default_file_from_the_ssd_on_a_16_gb_gpu():
    pack = _pack("abyss", "deepseek-v4-flash-0731", "UD-Q2_K_XL")

    assert pack_fit(pack, _machine(vram_gb=32, ram_gb=94)).tier == "memory"
    assert pack_fit(pack, _machine(vram_gb=16, ram_gb=62.5)).tier == "ssd"


def test_abyss_does_not_run_under_wsl_or_on_a_mac():
    linux = _machine(vram_gb=16, ram_gb=62.5, os="linux")
    mac = _machine(vram_gb=96, ram_gb=128, vendor="apple", os="darwin")
    abyss = [BY_ID["abyss"]]

    assert {o.fit.reason for o in pack_options(abyss, linux, wsl=True)} == {
        "does not run under WSL"
    }
    assert {o.fit.reason for o in pack_options(abyss, mac)} == {"no macOS build"}
    assert any(o.fit.tier == "memory" for o in pack_options(abyss, linux))


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
