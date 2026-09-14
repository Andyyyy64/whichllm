"""Coverage for the Qwen3.5 / Qwen3.6 generation and the task profiles (#24).

- newer Qwen releases were never fetched, so they could not be ranked at all
- ``Qwen/Qwen3-Coder-Next`` (80B total / 3B active) hit the generic
  expert-fraction estimate and came out as a ~33B-active model
- ``--profile math`` only kept repos with "math" in the name, which is a thin
  2024-era pool
"""

from __future__ import annotations

from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _generation_bonus, _matches_profile
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.benchmark_sources.aa_index import (
    AA_INDEX_FALLBACK_2026_09_14,
    AA_NAME_TO_HF_IDS,
    get_aa_curated_fallback,
)
from whichllm.models.hf import _FRONTIER_MODEL_IDS
from whichllm.models.parser import _parse_model
from whichllm.models.types import GGUFVariant, ModelInfo

_NEWER_QWEN = (
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-Coder-Next",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
)


def _gguf(quant: str, size_gb: float) -> GGUFVariant:
    return GGUFVariant(
        filename=f"model-{quant}.gguf",
        quant_type=quant,
        file_size_bytes=int(size_gb * 1e9),
    )


def _model(model_id: str, params_b: float, size_gb: float) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name=model_id.split("/")[-1],
        parameter_count=int(params_b * 1e9),
        downloads=10_000,
        gguf_variants=[_gguf("Q4_K_M", size_gb)],
    )


def _hw(vram_gb: int = 24) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="Test GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1000.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


class TestNewerQwenMetadata:
    def test_moe_metadata_from_a_b_naming(self):
        model = _parse_model({"id": "Qwen/Qwen3.6-35B-A3B", "config": {}})
        assert model is not None
        assert model.is_moe
        assert model.parameter_count == 35_000_000_000
        assert model.parameter_count_active == 3_000_000_000

    def test_coder_next_active_params_override_expert_heuristic(self):
        # 512 experts / top-10 routing: the generic estimate lands near 33B,
        # but the model card states 80B total and 3B activated.
        model = _parse_model(
            {
                "id": "Qwen/Qwen3-Coder-Next",
                "config": {"num_experts": 512, "num_experts_per_tok": 10},
                "safetensors": {"total": 79_674_391_296},
            }
        )
        assert model is not None
        assert model.is_moe
        assert model.parameter_count_active == 3_000_000_000

    def test_newer_releases_are_fetched(self):
        for model_id in _NEWER_QWEN:
            assert model_id in _FRONTIER_MODEL_IDS, f"{model_id} is never fetched"


class TestNewerQwenBenchmarkEvidence:
    def test_curated_snapshot_covers_newer_releases(self):
        for model_id in _NEWER_QWEN:
            assert model_id in AA_INDEX_FALLBACK_2026_09_14, f"{model_id} missing"

    def test_live_aa_names_map_to_newer_releases(self):
        mapped = {i for ids in AA_NAME_TO_HF_IDS.values() for i in ids}
        for model_id in _NEWER_QWEN:
            assert model_id in mapped, f"{model_id} unmapped from live AA names"

    def test_newer_generations_outrank_older_ones(self):
        fallback = get_aa_curated_fallback()
        assert fallback["Qwen/Qwen3.8-27B"] > fallback["Qwen/Qwen3.6-27B"]
        # AA v4.3 ranks Qwen3.6-27B just below Qwen3.5-27B, so no 3.6 > 3.5 check.
        assert fallback["Qwen/Qwen3.8-27B"] > fallback["Qwen/Qwen3.5-27B"]
        assert fallback["Qwen/Qwen3.6-35B-A3B"] > fallback["Qwen/Qwen3-30B-A3B"]
        assert fallback["Qwen/Qwen3.5-9B"] > fallback["Qwen/Qwen3-8B"]
        assert (
            fallback["Qwen/Qwen3-Coder-Next"]
            > fallback["Qwen/Qwen2.5-Coder-32B-Instruct"]
        )

    def test_generation_bonus_tracks_the_newest_qwen_line(self):
        # Without a qwen3.8 lineage row the newest generation falls through to
        # the generic "qwen3" pattern and is ranked below Qwen3.5 / Qwen3.6.
        assert _generation_bonus("Qwen/Qwen3.8-27B") > _generation_bonus(
            "Qwen/Qwen3.6-27B"
        )
        assert _generation_bonus("Qwen/Qwen3.6-27B") > _generation_bonus(
            "Qwen/Qwen3-32B"
        )


class TestMathProfile:
    def test_math_keeps_specialists_and_general_reasoning_models(self):
        assert _matches_profile(_model("Qwen/Qwen2.5-Math-7B", 7, 4.5), "math")
        assert _matches_profile(_model("Qwen/Qwen3.6-27B", 27, 17.0), "math")

    def test_math_still_excludes_coding_and_vision_specialists(self):
        assert not _matches_profile(_model("Qwen/Qwen3-Coder-Next", 80, 45.0), "math")
        assert not _matches_profile(_model("Qwen/Qwen3-VL-8B", 8, 5.0), "math")

    def test_general_profile_still_excludes_math_specialists(self):
        assert not _matches_profile(_model("Qwen/Qwen2.5-Math-7B", 7, 4.5), "general")

    def test_math_ranking_is_not_limited_to_math_named_repos(self):
        models = [
            _model("Qwen/Qwen2.5-Math-1.5B", 1.5, 1.6),
            _model("Qwen/Qwen3.6-27B", 27, 17.0),
        ]
        scores = {"Qwen/Qwen3.6-27B": 85.0, "Qwen/Qwen2.5-Math-1.5B": 22.0}
        results = rank_models(
            models, _hw(), task_profile="math", benchmark_scores=scores
        )
        assert [r.model.id for r in results][0] == "Qwen/Qwen3.6-27B"
