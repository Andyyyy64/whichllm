"""The 8 tok/s usability bar and hiding older generations of a model line."""

from __future__ import annotations

from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _newer_generation
from whichllm.engine.ranking_score import USABLE_SPEED_TOK_S, _compute_quality_score
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _model(model_id: str, params_b: float, active_b: float | None = None) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=int(params_b * 1e9),
        parameter_count_active=int(active_b * 1e9) if active_b else None,
        is_moe=active_b is not None,
        downloads=1000,
    )


def _score(tok_per_sec: float, fit_type: str = "full_gpu") -> float:
    return _compute_quality_score(
        _model("org/Model-27B", 27),
        None,
        tok_per_sec,
        fit_type,
        benchmark_avg=78.0,
        benchmark_source="direct",
    )


def test_every_placement_shares_one_usable_bar():
    assert USABLE_SPEED_TOK_S == 8.0
    for fit in ("full_gpu", "partial_offload", "cpu_only"):
        assert _score(4.0, fit) < _score(8.0, fit)


def test_slow_generation_scales_quality_down():
    # Halving the speed below the bar costs far more than the old flat
    # 8-point band: a benchmark lead cannot carry a 4 tok/s model.
    assert _score(8.0) - _score(4.0) > 15
    assert _score(2.0) < _score(4.0)


def _hw(vram_gb: int, bandwidth: float) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                memory_bandwidth_gbps=bandwidth,
            )
        ],
        cpu_name="CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=1024**4,
        os="linux",
    )


def _ranked(model_id: str, params_b: float, size_gb: float, active_b=None) -> ModelInfo:
    model = _model(model_id, params_b, active_b)
    model.gguf_variants = [
        GGUFVariant(
            filename=f"{model.name}-Q4_K_M.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=int(size_gb * 1024**3),
        )
    ]
    return model


def test_a_fast_model_outranks_a_slightly_stronger_slow_one():
    # A dense 27B at ~3 tok/s against a MoE at ~13 tok/s.
    dense = _ranked("Qwen/Qwen3.8-27B", 27, 14.5)
    moe = _ranked("google/gemma-4-26B-A4B-it", 26, 14.5, active_b=4)
    results = rank_models(
        [dense, moe],
        _hw(16, 60.0),
        benchmark_scores={dense.id: 77.8, moe.id: 54.6},
        task_profile="any",
    )
    assert results[0].model.id == moe.id
    assert results[0].estimated_tok_per_sec >= USABLE_SPEED_TOK_S


def test_newer_generation_of_the_same_line():
    q38 = _model("Qwen/Qwen3.8-27B", 27.3)
    q36 = _model("Qwen/Qwen3.6-27B", 27.8)
    q36_moe = _model("Qwen/Qwen3.6-35B-A3B", 35, active_b=3)
    q35_moe = _model("Qwen/Qwen3.5-35B-A3B", 35, active_b=3)
    coder = _model("Qwen/Qwen3-Coder-30B-A3B-Instruct", 30, active_b=3)
    distill = _model("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 32)

    assert _newer_generation(q38, q36)
    assert not _newer_generation(q36, q38)
    assert _newer_generation(q36_moe, q35_moe)
    # A different size, a dense/MoE mismatch, a specialization or a model
    # that belongs to two lineages is a different line.
    assert not _newer_generation(q38, q36_moe)
    assert not _newer_generation(q36_moe, coder)
    assert not _newer_generation(q38, distill)


def test_older_generations_are_hidden_behind_the_newer_one():
    q38 = _ranked("Qwen/Qwen3.8-27B", 27.3, 16)
    q36 = _ranked("Qwen/Qwen3.6-27B", 27.8, 16)
    q35 = _ranked("Qwen/Qwen3.5-27B", 27.8, 16)
    other = _ranked("google/gemma-4-31B-it", 31, 17)

    results = rank_models(
        [q38, q36, q35, other],
        _hw(24, 1000.0),
        benchmark_scores={q38.id: 77.8, q36.id: 66.0, q35.id: 65.0, other.id: 56.2},
        task_profile="any",
    )

    assert [r.model.id for r in results] == [q38.id, other.id]
    assert results[0].older_generations == [q36.id, q35.id]


def test_an_older_generation_that_outranks_the_newer_one_is_kept():
    # The newer release only fits at a speed or with evidence that ranks it
    # lower; hiding the older one would hide the better option.
    q38 = _ranked("Qwen/Qwen3.8-27B", 27.3, 16)
    q36 = _ranked("Qwen/Qwen3.6-27B", 27.8, 16)

    results = rank_models(
        [q38, q36],
        _hw(24, 1000.0),
        benchmark_scores={q36.id: 66.0},
        task_profile="any",
    )

    assert [r.model.id for r in results] == [q36.id, q38.id]
    assert results[0].older_generations == []
