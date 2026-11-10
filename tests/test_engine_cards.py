"""Engine cards: every card follows the schema, and availability reads them right."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from whichllm.cli import app
from whichllm.engines import availability, load_cards
from whichllm.engines.cards import CardError, parse_card
from whichllm.hardware.types import GPUInfo, HardwareInfo

_GiB = 1024**3


def _card(**changes) -> dict:
    card = {
        "id": "demo",
        "name": "Demo",
        "kind": "runner",
        "url": "https://example.com/demo",
        "evidence": "documented",
        "upstream": {"status": "active"},
        "checked_at": "2026-11-04",
        "sources": ["https://example.com/demo/docs"],
        "platforms": [{"os": ["linux"], "accel": "cpu"}],
    }
    card.update(changes)
    return card


def _machine(os="linux", gpu=None, avx2=True) -> HardwareInfo:
    return HardwareInfo(
        gpus=[gpu] if gpu else [],
        cpu_name="CPU",
        cpu_cores=8,
        has_avx2=avx2,
        ram_bytes=64 * _GiB,
        os=os,
    )


def _nvidia(name: str, capability: tuple[int, int]) -> GPUInfo:
    return GPUInfo(
        name=name, vendor="nvidia", vram_bytes=16 * _GiB, compute_capability=capability
    )


def _by_id(hardware: HardwareInfo) -> dict:
    return {card.id: availability(card, hardware) for card in load_cards()}


def test_every_card_loads_with_its_sources():
    cards = load_cards()

    assert {card.id for card in cards} >= {
        "llama.cpp",
        "ik_llama.cpp",
        "ollama",
        "lm-studio",
        "koboldcpp",
        "jan",
        "lemonade",
    }
    for card in cards:
        assert card.sources, card.id
        assert all(source.startswith("https://") for source in card.sources)
        assert card.checked_at >= "2026-11-01"


def test_ik_quants_do_not_load_in_mainline():
    cards = {card.id: card for card in load_cards()}

    assert cards["llama.cpp"].gguf.extra == ()
    assert "IQ4_KS" in cards["ik_llama.cpp"].gguf.extra
    assert "TQ2_0" in cards["ik_llama.cpp"].gguf.missing
    # ik_llama.cpp loads mainline NVFP4 files since November 2026.
    assert "NVFP4" not in cards["ik_llama.cpp"].gguf.missing


@pytest.mark.parametrize(
    ("changes", "complaint"),
    [
        ({"colour": "red"}, "unknown key(s) in the card: colour"),
        ({"kind": "plugin"}, "'kind' is 'plugin'"),
        ({"evidence": "rumoured"}, "'evidence' is 'rumoured'"),
        ({"checked_at": "4 Nov 2026"}, "'checked_at' must be YYYY-MM-DD"),
        ({"sources": ["http://example.com"]}, "'sources' needs at least one https URL"),
        ({"platforms": []}, "'platforms' needs at least one entry"),
        ({"platforms": [{"os": ["beos"], "accel": "cpu"}]}, "platform os"),
        ({"platforms": [{"os": ["linux"], "accel": "tpu"}]}, "'accel' is 'tpu'"),
        (
            {
                "platforms": [
                    {
                        "os": ["linux"],
                        "accel": "vulkan",
                        "min_compute_capability": "7.5",
                    }
                ]
            },
            "min_compute_capability is a CUDA",
        ),
        ({"memory": ["swap"]}, "'memory'"),
        ({"port": 70000}, "'port' 70000"),
        (
            {"upstream": {"status": "active", "stars": 5}},
            "unknown key(s) in 'upstream'",
        ),
    ],
)
def test_a_card_that_breaks_the_schema_names_the_file_and_field(changes, complaint):
    with pytest.raises(CardError) as error:
        parse_card(_card(**changes), Path("demo.json"))

    assert str(error.value).startswith("demo.json: ")
    assert complaint in str(error.value)


def test_a_card_is_named_after_its_id():
    with pytest.raises(CardError, match="named after the id"):
        parse_card(_card(), Path("other.json"))


def test_a_recent_nvidia_card_runs_every_cuda_engine_on_cuda():
    rows = _by_id(_machine(gpu=_nvidia("NVIDIA GeForce RTX 5080", (12, 0))))
    cuda = {
        row.card.id: row
        for row in rows.values()
        if any(p.accel == "cuda" for p in row.card.platforms)
    }

    assert len(cuda) >= 9
    assert all(row.runs and row.platform.accel == "cuda" for row in cuda.values())
    assert all(row.reason is None for row in cuda.values())


def test_engines_without_a_cuda_build_say_what_they_run_on():
    rows = _by_id(_machine(gpu=_nvidia("NVIDIA GeForce RTX 5080", (12, 0))))

    # WARP runs on the CPU only, and TurboFieldfare only on a Mac.
    assert rows["warp"].platform.accel == "cpu"
    assert rows["warp"].reason == "no cuda build"
    assert not rows["turbo-fieldfare"].runs


def test_an_old_nvidia_card_falls_back_and_says_why():
    rows = _by_id(_machine(gpu=_nvidia("NVIDIA GeForce GTX 1060", (6, 1))))
    why = "needs CUDA compute capability 7.5, this GPU is 6.1"

    assert rows["llama.cpp"].platform.accel == "cuda"
    assert rows["jan"].platform.accel == "vulkan"
    assert rows["jan"].reason == why
    assert rows["ik_llama.cpp"].platform.accel == "cpu"
    assert rows["ik_llama.cpp"].reason == why


def test_a_cpu_without_avx2_cannot_run_lm_studio():
    rows = _by_id(_machine(os="windows", avx2=False))

    assert not rows["lm-studio"].runs
    assert rows["lm-studio"].reason == "needs a CPU with AVX2"
    assert rows["llama.cpp"].platform.accel == "cpu"


def test_an_apple_silicon_mac_uses_metal_and_has_no_ik_build():
    apple = GPUInfo(name="Apple M3 Max", vendor="apple", vram_bytes=36 * _GiB)
    rows = _by_id(_machine(os="darwin", gpu=apple))

    assert rows["llama.cpp"].platform.accel == "metal"
    assert rows["lm-studio"].platform.accel == "metal"
    assert not rows["ik_llama.cpp"].runs
    assert rows["ik_llama.cpp"].reason == "no macOS build"


def test_an_intel_mac_has_no_lm_studio():
    rows = _by_id(_machine(os="darwin"))

    assert not rows["lm-studio"].runs
    assert rows["llama.cpp"].platform.accel == "cpu"


def test_engines_json_lists_where_each_engine_runs(monkeypatch):
    hardware = _machine(gpu=_nvidia("NVIDIA GeForce GTX 1060", (6, 1)))
    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", lambda: hardware)

    result = CliRunner().invoke(app, ["engines", "--json"])

    assert result.exit_code == 0, result.output
    engines = {row["id"]: row for row in json.loads(result.output)["engines"]}
    assert engines["llama.cpp"]["accel"] == "cuda"
    assert engines["jan"]["accel"] == "vulkan"
    assert engines["jan"]["reason"].startswith("needs CUDA compute capability 7.5")
    assert engines["llama.cpp"]["port"] == 9931
    assert engines["llama.cpp"]["sources"]


def test_a_simulated_apple_gpu_is_treated_as_a_mac(monkeypatch):
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _machine(os="linux")
    )

    result = CliRunner().invoke(app, ["engines", "--gpu", "Apple M3 Max", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    engines = {row["id"]: row for row in data["engines"]}
    assert data["hardware"]["os"] == "darwin"
    assert engines["llama.cpp"]["accel"] == "metal"


def test_json_before_engines_is_refused(monkeypatch):
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware",
        Mock(side_effect=AssertionError("must stop before detecting")),
    )

    result = CliRunner().invoke(app, ["--json", "engines"])

    assert result.exit_code == 2
    assert "whichllm engines --json" in " ".join(result.output.split())
