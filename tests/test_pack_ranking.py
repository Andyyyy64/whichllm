"""Rows under the ranking for models a fixed-model engine runs here."""

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from whichllm.engine.pack_ranking import (
    attach_engine_alternatives,
    merge_pack_rows,
    pack_tiers,
    rank_pack_rows,
)
from whichllm.engine.types import CompatibilityResult
from whichllm.engines import load_cards
from whichllm.engines.detect import Installed
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.moe_fields import fill_moe_from_shape
from whichllm.models.shape import parse_model_shape
from whichllm.models.types import GGUFVariant, ModelInfo

_GiB = 1024**3
CARDS = load_cards()
STRATA = tuple(card for card in CARDS if card.id == "strata")
BASE = "Qwen/Qwen3.8-Flash-Next"
PACKS = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
# The real config.json, so the packs are timed as the MoE they are.
SHAPE = parse_model_shape(
    json.loads(
        (Path(__file__).parent / "fixtures" / "model_configs.json").read_text(
            encoding="utf-8"
        )
    )[BASE]
)


def _catalog() -> list[ModelInfo]:
    family = "qwen/qwen3.8-flash-next"
    base = ModelInfo(
        id=BASE,
        family_id=family,
        name="Qwen3.8-Flash-Next",
        parameter_count=180_000_000_000,
        is_moe=True,
        shape=SHAPE,
        context_length=262_144,
        downloads=500_000,
        published_at="2026-09-20T00:00:00Z",
        license="apache-2.0",
    )
    packs = ModelInfo(
        id=PACKS,
        family_id=family,
        name="Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        parameter_count=176_900_000_000,
        is_moe=True,
        shape=SHAPE,
        base_model=BASE,
        base_model_relation="quantized",
        gguf_variants=[
            GGUFVariant("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf", "Q2_0", 66 * _GiB),
            GGUFVariant("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf", "IQ3_S", 84 * _GiB),
        ],
    )
    for model in (base, packs):
        fill_moe_from_shape(model)
    return [base, packs]


def _machine(ram_gb=64, vram_gb=16) -> HardwareInfo:
    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 5080",
        vendor="nvidia",
        vram_bytes=vram_gb * _GiB,
        compute_capability=(12, 0),
        memory_bandwidth_gbps=960,
    )
    return HardwareInfo(
        gpus=[gpu],
        has_avx2=True,
        ram_bytes=ram_gb * _GiB,
        ram_bandwidth_gbps=89.6,
        os="windows",
    )


def _shown(score: float, family: str = "other/family") -> CompatibilityResult:
    model = ModelInfo(id="org/Other", family_id=family, name="Other", parameter_count=1)
    return CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=score,
    )


def _rows(hardware=None, shown=(), cards=STRATA, **kwargs):
    return rank_pack_rows(
        _catalog(),
        hardware or _machine(),
        cards,
        shown=list(shown),
        benchmark_scores={BASE: 80.0},
        **kwargs,
    )


def test_a_pack_the_ranking_could_not_place_gets_a_row():
    rows = _rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.model.id == BASE
    assert row.artifact_model.id == PACKS
    assert row.pack_tier == "memory"
    # 64 GB holds IQ3_S's experts, which lose less than Q2_0's.
    assert row.pack_name == "ISTA-DASLab IQ3_S"
    assert row.gguf_variant.filename == "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf"
    assert row.pack_launch == "strata launch --family qwen --model iq3_s"
    assert row.benchmark_status == "direct"
    # Timed as llama.cpp would run the file, with Strata's own number beside it.
    assert row.estimated_tok_per_sec > 0
    assert row.pack_docs_speed == (53.0, "RTX 5070 12 GB, Ryzen 5 7600, 64 GB RAM")


def test_less_ram_picks_the_pack_that_still_fits():
    rows = _rows(_machine(ram_gb=48))

    assert rows[0].pack_name == "ISTA-DASLab IQ2_XS"


def test_rows_must_beat_the_last_ranked_model():
    score = _rows()[0].quality_score

    assert _rows(shown=[_shown(score - 1)])
    assert _rows(shown=[_shown(score + 1)]) == []


def test_a_better_row_of_the_same_family_keeps_the_pack_out():
    score = _rows()[0].quality_score
    same_family = _shown(score + 1, family="qwen/qwen3.8-flash-next")

    assert _rows(shown=[_shown(1.0), same_family]) == []


def test_reading_from_the_ssd_scores_lower_than_memory():
    in_memory = _rows(_machine(ram_gb=48), quant_filter="Q2_0")[0]
    from_ssd = _rows(_machine(ram_gb=32, vram_gb=24), quant_filter="Q2_0")[0]

    assert from_ssd.pack_tier == "ssd"
    assert from_ssd.quality_score < in_memory.quality_score
    assert "from the SSD" in from_ssd.warnings[0]


def test_filters_apply_to_pack_rows():
    speed = _rows()[0].estimated_tok_per_sec

    # A speed floor uses each pack's llama.cpp estimate.
    assert _rows(min_speed=speed - 1)
    assert _rows(min_speed=1000.0) == []
    # A pack read from the SSD is not timed, so any speed floor leaves it out.
    ssd = _machine(ram_gb=32, vram_gb=24)
    assert _rows(ssd, quant_filter="Q2_0")[0].pack_tier == "ssd"
    assert _rows(ssd, quant_filter="Q2_0", min_speed=1.0) == []
    assert _rows(fit_filter="full_gpu") == []
    assert _rows(quant_filter="Q2_0")[0].pack_name == "ISTA-DASLab Q2_0"


def test_a_model_the_ranking_has_gets_no_second_row():
    ranked = _shown(1.0, family="qwen/qwen3.8-flash-next")

    assert _rows(shown=[ranked]) == []


def test_installed_is_known_only_when_detection_ran():
    assert _rows()[0].engine_installed is None
    assert _rows(installed={})[0].engine_installed is False
    found = {"strata": Installed("/home/me/.local/bin/strata", None, False)}
    assert _rows(installed=found)[0].engine_installed is True


def _console(monkeypatch, width=200) -> StringIO:
    from whichllm.output import _console as console_mod
    from whichllm.output import ranking

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=width))
    monkeypatch.setattr(ranking.platform, "system", lambda: "Windows")
    return buf


def test_pack_rows_take_their_place_by_score():
    row = _rows()[0]
    above, below = _shown(row.quality_score + 5), _shown(row.quality_score - 5)
    last = _shown(row.quality_score - 9)

    merged = merge_pack_rows([above, below, last], [row], top_n=3)

    assert merged == [above, row, below]


def test_the_top_pick_can_be_a_pack_row(monkeypatch):
    from whichllm.output.ranking import display_ranking

    buf = _console(monkeypatch)
    row = _rows(installed={})[0]
    ranked = merge_pack_rows([_shown(50.0)], [row], top_n=10)

    display_ranking(ranked, show_status=True, context_length=4096)

    shown = buf.getvalue()
    assert ranked[0] is row
    assert "with Strata, ISTA-DASLab IQ3_S" in shown
    assert "docs 53" in shown
    assert "with <engine> = runs with that engine from its own pack" in shown
    assert "Run #1 with Strata:" in shown
    assert "strata launch --family qwen --model iq3_s" in shown
    assert "Strata is not installed here; install with irm" in shown
    assert "+ =" not in shown


def test_a_pack_row_below_the_top_pick_names_its_rank(monkeypatch):
    from whichllm.output.ranking import display_ranking

    buf = _console(monkeypatch)
    row = _rows(installed={})[0]
    ranked = merge_pack_rows([_shown(row.quality_score + 5)], [row], top_n=10)

    display_ranking(ranked, show_status=True, context_length=4096)

    assert "Run #2 with Strata:" in buf.getvalue()


def test_json_rows_carry_the_pack_and_its_command(monkeypatch):
    from whichllm.output.json_output import display_json

    buf = _console(monkeypatch, width=10_000)
    row = _rows()[0]

    display_json(merge_pack_rows([_shown(50.0)], [row], top_n=10), _machine(), 4096)

    data = json.loads(buf.getvalue())
    first = data["models"][0]
    assert "other_engines" not in data
    assert first["rank"] == 1
    assert first["model_id"] == BASE
    assert first["artifact_repo_id"] == PACKS
    assert first["engine"] == "strata"
    assert first["pack"] == {
        "name": "ISTA-DASLab IQ3_S",
        "tier": "memory",
        "docs_tok_per_sec": 53.0,
        "docs_hardware": "RTX 5070 12 GB, Ryzen 5 7600, 64 GB RAM",
    }
    assert first["estimated_tok_per_sec"] > 0
    assert first["recipe"]["engine"] == "strata"
    assert first["recipe"]["command"] == "strata launch --family qwen --model iq3_s"
    assert first["recipe"]["llama_cpp"] is None
    assert data["models"][1]["pack"] is None


def test_markdown_ranks_pack_rows_in_the_table(monkeypatch):
    from whichllm.output import markdown

    written = []
    monkeypatch.setattr(markdown, "_write_markdown", written.append)
    row = _rows()[0]
    ranked = merge_pack_rows([_shown(50.0)], [row], top_n=10)

    markdown.display_markdown(ranked, _machine(), show_status=True, context_length=4096)

    text = written[0]
    assert "| 1 | [Qwen/Qwen3.8-Flash-Next](" in text
    assert "with Strata, ISTA-DASLab IQ3_S" in text
    assert "| GPU+RAM |" in text
    assert " tok/s ?" in text
    assert "Run #1 with Strata (53 tok/s in its docs on RTX 5070" in text
    assert "strata launch --family qwen --model iq3_s" in text


def test_a_pack_in_memory_comes_before_one_read_from_the_ssd():
    # 32 GB of RAM and a 24 GB GPU run Q2_0 from the SSD, and hold Gyro-M's
    # 35 GB in RAM and VRAM together.
    row = _rows(_machine(ram_gb=32, vram_gb=24))[0]

    assert row.pack_name == "Gyro-M"
    assert row.pack_tier == "memory"


def test_a_cpu_engine_is_a_cpu_only_fit():
    # WARP runs on the CPU only; with Qwen3.8-Flash-Next's experts read from
    # the SSD it needs about 3 GB of RAM.
    rows = rank_pack_rows(
        _catalog(),
        _machine(ram_gb=16),
        tuple(c for c in CARDS if c.id == "warp"),
        shown=[],
        benchmark_scores={BASE: 80.0},
    )

    row = rows[0]
    assert row.engine == "warp"
    assert row.fit_type == "cpu_only"
    assert row.pack_tier == "ssd"
    assert pack_tiers(row) == ["RAM", "SSD"]


def test_each_engine_adds_one_row_per_model_best_first():
    rows = _rows(cards=CARDS)
    engines = [row.engine for row in rows]

    assert len(engines) == len(set(engines))
    assert "strata" in engines
    assert [r.quality_score for r in rows] == sorted(
        (r.quality_score for r in rows), reverse=True
    )


def test_a_pack_read_from_the_ssd_is_not_timed():
    row = _rows(_machine(ram_gb=32, vram_gb=24), quant_filter="Q2_0")[0]

    assert row.pack_tier == "ssd"
    assert row.estimated_tok_per_sec is None


def _llama_cpp_row() -> CompatibilityResult:
    catalog = _catalog()
    return CompatibilityResult(
        model=catalog[0],
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        quality_score=80.0,
    )


def test_a_ranked_row_names_the_engines_that_also_run_its_model(monkeypatch):
    from whichllm.output.ranking import _print_engine_alternatives

    row = _llama_cpp_row()
    attach_engine_alternatives([row], _catalog(), _machine(), STRATA)

    assert [(a.engine, a.pack_name) for a in row.engine_alternatives] == [
        ("strata", "ISTA-DASLab IQ3_S")
    ]
    buf = _console(monkeypatch)
    _print_engine_alternatives([row])
    shown = buf.getvalue()
    assert "#1 also runs with Strata" in shown
    assert "53 tok/s in its docs on RTX 5070" in shown
    assert "strata launch --family qwen --model iq3_s" in shown


def test_json_rows_list_engine_alternatives(monkeypatch):
    from whichllm.output.json_output import display_json

    row = _llama_cpp_row()
    attach_engine_alternatives([row], _catalog(), _machine(), STRATA)
    buf = _console(monkeypatch, width=10_000)

    display_json([row], _machine(), 4096)

    alternatives = json.loads(buf.getvalue())["models"][0]["engine_alternatives"]
    assert alternatives == [
        {
            "engine": "strata",
            "pack": "ISTA-DASLab IQ3_S",
            "command": "strata launch --family qwen --model iq3_s",
            "docs_tok_per_sec": 53.0,
            "docs_hardware": "RTX 5070 12 GB, Ryzen 5 7600, 64 GB RAM",
        }
    ]
