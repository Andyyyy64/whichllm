"""Task scores: AA per-benchmark results, scaling, and their use in ranking."""

from __future__ import annotations

import asyncio
import json

import pytest

from whichllm.engine.ranker import rank_models
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models import benchmark_sources
from whichllm.models.benchmark_fetch import fetch_benchmark_data
from whichllm.models.benchmark_sources import aa_evals
from whichllm.models.task_scores import (
    AA_TASK_EVALS,
    MIN_OVERLAP,
    build_task_scores,
    match_names_to_ids,
    name_tokens,
)
from whichllm.models.types import GGUFVariant, ModelInfo


def _page(records: list[dict]) -> str:
    """An AA leaderboard page carrying ``records`` the way the RSC stream does."""
    fragment = '3:["$","$L23",null,{"initialModels":' + json.dumps(records) + "}]\n"
    chunk = json.dumps(fragment)
    return (
        "<html><body><script>self.__next_f.push([0])</script>"
        f"<script>self.__next_f.push([1,{chunk}])</script></body></html>"
    )


def test_display_names_match_repo_names_in_any_order():
    ids = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "google/gemma-4-26B-A4B-it",
        "Qwen/Qwen3-8B-Base",
    ]
    matches = match_names_to_ids(
        ["llama 3.3 instruct 70b", "Gemma 4 26B A4B (Reasoning)", "qwen3 8b"], ids
    )
    assert matches == {
        "llama 3.3 instruct 70b": ["meta-llama/Llama-3.3-70B-Instruct"],
        "Gemma 4 26B A4B (Reasoning)": ["google/gemma-4-26B-A4B-it"],
    }
    assert name_tokens("Qwen3-8B-Base") != name_tokens("qwen3 8b")


def test_reuploads_are_dropped_when_the_original_matches():
    ids = ["Qwen/Qwen3.6-35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert match_names_to_ids(["qwen3.6 35b a3b"], ids) == {
        "qwen3.6 35b a3b": ["Qwen/Qwen3.6-35B-A3B"]
    }
    only_upload = ["unsloth/gemma-4-12b-it-GGUF"]
    assert match_names_to_ids(["gemma 4 12b"], only_upload) == {
        "gemma 4 12b": only_upload
    }


def _population(n: int = 10) -> tuple[dict, dict]:
    """Models whose general score is 40..85 and whose benchmarks track it."""
    general = {f"org/M{i}": 40.0 + 5 * i for i in range(n)}
    evals = {
        model_id: {"scicode": score / 100, "livecodebench": score / 200}
        for model_id, score in general.items()
    }
    return general, evals


def test_benchmarks_are_put_on_the_general_scale():
    general, evals = _population()
    coding = build_task_scores(evals, general)["coding"]
    for model_id, score in general.items():
        assert coding[model_id] == pytest.approx(score, abs=0.1)


def test_a_model_that_codes_better_than_its_general_score_ranks_higher():
    general, evals = _population()
    evals["org/Coder"] = {"scicode": 0.80, "livecodebench": 0.40}
    general_with_coder = {**general, "org/Coder": 50.0}
    coding = build_task_scores(evals, general, general_with_coder)["coding"]
    assert coding["org/Coder"] > 75.0


def test_a_single_benchmark_is_averaged_with_the_general_score():
    general, evals = _population()
    evals["org/One"] = {"scicode": 0.80}
    coding = build_task_scores(evals, general, {**general, "org/One": 50.0})["coding"]
    assert coding["org/One"] == pytest.approx((80.0 + 50.0) / 2, abs=0.5)


def test_too_little_overlap_with_the_general_index_maps_nothing():
    general, evals = _population(MIN_OVERLAP - 1)
    assert build_task_scores(evals, general) == {}


def test_general_scores_need_three_benchmarks():
    general, evals = _population()
    for model_id in general:
        score = general[model_id]
        evals[model_id].update({"gpqa-diamond": score / 100, "ifbench": score / 150})
    evals["org/New"] = {"gpqa-diamond": 0.6, "ifbench": 0.6}
    assert "org/New" not in build_task_scores(evals, general, profiles=["general"]).get(
        "general", {}
    )
    evals["org/New"]["scicode"] = 0.6
    assert (
        "org/New" in build_task_scores(evals, general, profiles=["general"])["general"]
    )


def test_parse_keeps_open_models_and_the_best_variant():
    page = _page(
        [
            {
                "name": "Qwen3.6 35B A3B (Reasoning)",
                "isOpenWeights": True,
                "scores": {"scicode": 0.36, "terminalbench-2-1": 0.45, "made-up": 0.9},
            },
            {
                "name": "Qwen3.6 35B A3B (Non-reasoning)",
                "isOpenWeights": True,
                "scores": {"scicode": 0.40, "terminalbench-2-1": 2.0},
            },
            {
                "name": "Closed Model",
                "isOpenWeights": False,
                "scores": {"scicode": 0.9},
            },
        ]
    )
    assert aa_evals.parse_aa_evals(page, AA_TASK_EVALS) == {
        "qwen3.6 35b a3b": {"scicode": 0.40, "terminalbench-2-1": 0.45}
    }


def test_committed_snapshot_is_readable():
    snapshot = aa_evals.load_aa_eval_snapshot()
    assert len(snapshot) > 50
    assert all(set(r) <= AA_TASK_EVALS for r in snapshot.values())


def test_fetch_builds_task_scores_and_fills_new_models(monkeypatch):
    general, evals = _population()
    aa_results = {f"m{i}": evals[f"org/M{i}"] for i in range(10)}
    for i in range(10):
        aa_results[f"m{i}"].update(
            {
                "gpqa-diamond": general[f"org/M{i}"] / 100,
                "ifbench": general[f"org/M{i}"] / 150,
            }
        )
    # A release AA scores that the general sources have not.
    aa_results["fresh 9b"] = {"scicode": 0.7, "gpqa-diamond": 0.7, "ifbench": 0.7 / 1.5}

    async def empty(client):
        return {}

    async def aa_index(client):
        return dict(general)

    async def fetch_evals(client, wanted):
        return aa_results

    for name in ("fetch_leaderboard_with_fallback", "fetch_arena_scores"):
        monkeypatch.setattr(benchmark_sources, name, empty)
    monkeypatch.setattr(benchmark_sources, "fetch_aa_index_scores", aa_index)
    monkeypatch.setattr(benchmark_sources, "fetch_vision_scores", lambda c: empty(c))
    monkeypatch.setattr(benchmark_sources, "get_livebench_data", lambda: {})
    monkeypatch.setattr(aa_evals, "fetch_aa_eval_scores", fetch_evals)

    data = asyncio.run(fetch_benchmark_data([*general, "org/Fresh-9B"]))

    assert data.general["org/M3"] == general["org/M3"]
    assert data.general["org/Fresh-9B"] == pytest.approx(70.0, abs=3.0)
    assert data.tasks["coding"]["org/M3"] == pytest.approx(general["org/M3"], abs=1.0)


def test_vision_index_no_longer_raises_the_general_score(monkeypatch):
    async def empty(client):
        return {}

    async def vision(client):
        return {"Qwen/Qwen3-VL-32B-Instruct": 57.0}

    for name in (
        "fetch_leaderboard_with_fallback",
        "fetch_arena_scores",
        "fetch_aa_index_scores",
    ):
        monkeypatch.setattr(benchmark_sources, name, empty)
    monkeypatch.setattr(benchmark_sources, "fetch_vision_scores", vision)
    monkeypatch.setattr(benchmark_sources, "get_livebench_data", lambda: {})

    data = asyncio.run(fetch_benchmark_data())
    assert "Qwen/Qwen3-VL-32B-Instruct" not in data.general


def _hw() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=1000.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=1024**4,
        os="linux",
    )


def _ranked(model_id: str) -> ModelInfo:
    model = ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=30_000_000_000,
        downloads=1000,
    )
    model.gguf_variants = [
        GGUFVariant(
            filename=f"{model.name}-Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=17 * 1024**3,
        )
    ]
    return model


def test_coding_profile_ranks_on_coding_scores():
    general_model = _ranked("org/Generalist-30B")
    coder = _ranked("org/Strong-Coder-30B")
    general = {general_model.id: 75.0, coder.id: 65.0}
    coding = {general_model.id: 66.0, coder.id: 78.0}

    by_general = rank_models(
        [general_model, coder], _hw(), benchmark_scores=general, task_profile="coding"
    )
    by_task = rank_models(
        [general_model, coder],
        _hw(),
        benchmark_scores=general,
        task_scores=coding,
        task_profile="coding",
    )

    assert by_general[0].model.id == general_model.id
    assert by_task[0].model.id == coder.id
    assert by_task[0].benchmark_basis == "coding"


def test_name_bonus_only_applies_without_coding_evidence():
    plain = _ranked("Qwen/Qwen3-30B-A3B")
    coder = _ranked("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    general = {plain.id: 70.0, coder.id: 70.0}

    results = rank_models(
        [plain, coder],
        _hw(),
        benchmark_scores=general,
        task_scores={plain.id: 70.0, coder.id: 70.0},
        task_profile="coding",
    )

    assert results[0].quality_score == results[1].quality_score
