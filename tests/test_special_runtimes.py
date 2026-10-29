"""Draft heads, diffusion LMs and GPU-specific checkpoint formats."""

from __future__ import annotations

from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _is_auxiliary_model
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo


def _model(model_id: str, *, tags=(), gguf: bool = True) -> ModelInfo:
    model = ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=8_000_000_000,
        downloads=1000,
        tags=tuple(tags),
    )
    if gguf:
        model.gguf_variants = [
            GGUFVariant(
                filename=f"{model.name}-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=5 * 1024**3,
            )
        ]
    return model


def _linux_nvidia(cc: tuple[int, int]) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=900.0,
                compute_capability=cc,
            )
        ],
        cpu_name="CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=1024**4,
        os="linux",
    )


def test_draft_heads_and_diffusion_models_are_auxiliary():
    assert _is_auxiliary_model(_model("AtomicChat/Qwen3-8B-DFlash-GGUF"))
    assert _is_auxiliary_model(_model("org/Qwen2.5-0.5B-Instruct-draft"))
    assert _is_auxiliary_model(_model("google/diffusiongemma-26B-A4B-it"))
    assert _is_auxiliary_model(_model("org/Model-8B", tags=("diffusion_gemma",)))
    assert not _is_auxiliary_model(_model("Qwen/Qwen3-8B"))
    assert not _is_auxiliary_model(_model("org/Drafter-Writer-8B"))


def test_auxiliary_models_are_never_ranked():
    draft = _model("AtomicChat/Qwen3-8B-DFlash-GGUF")
    diffusion = _model("google/diffusiongemma-26B-A4B-it")
    chat = _model("Qwen/Qwen3-8B")
    results = rank_models(
        [draft, diffusion, chat],
        _linux_nvidia((8, 9)),
        benchmark_scores={draft.id: 90.0, diffusion.id: 90.0, chat.id: 50.0},
        task_profile="any",
    )
    assert [r.model.id for r in results] == [chat.id]


def _ranked_ids(model: ModelInfo, cc: tuple[int, int]) -> list:
    return rank_models(
        [model],
        _linux_nvidia(cc),
        benchmark_scores={model.id: 60.0},
        task_profile="any",
    )


def test_nvfp4_needs_ampere_and_notes_marlin_before_blackwell():
    nvfp4 = _model("someone/Model-8B-NVFP4", gguf=False)

    assert _ranked_ids(nvfp4, (7, 5)) == []

    ada = _ranked_ids(nvfp4, (8, 9))
    assert ada and any("Marlin" in w for w in ada[0].warnings)

    blackwell = _ranked_ids(nvfp4, (12, 0))
    assert blackwell and not any("Marlin" in w for w in blackwell[0].warnings)


def test_fp8_needs_ampere():
    fp8 = _model("someone/Model-8B-FP8", gguf=False)
    assert _ranked_ids(fp8, (7, 5)) == []
    assert _ranked_ids(fp8, (8, 0))
