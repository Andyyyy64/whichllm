"""Tests for resolving concrete downloadable artifacts."""

from whichllm.engine.types import CompatibilityResult
from whichllm.models.artifacts import attach_resolved_artifacts
from whichllm.models.types import GGUFVariant, ModelInfo


def test_attach_resolved_artifacts_maps_synthetic_quant_to_real_gguf_repo():
    base = ModelInfo(
        id="Qwen/Qwen3-4B-Thinking-2507",
        family_id="qwen3-4b-thinking",
        name="Qwen3-4B-Thinking-2507",
        parameter_count=4_000_000_000,
        downloads=494_000,
    )
    gguf_repo = ModelInfo(
        id="MaziyarPanahi/Qwen3-4B-Thinking-2507-GGUF",
        family_id="qwen3-4b-thinking",
        name="Qwen3-4B-Thinking-2507-GGUF",
        parameter_count=4_000_000_000,
        downloads=26_000,
        base_model="Qwen/Qwen3-4B-Thinking-2507",
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3-4B-Thinking-2507-Q3_K_M.gguf",
                quant_type="Q3_K_M",
                file_size_bytes=2_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3-4B-Thinking-2507.Q3_K_M.gguf",
        quant_type="Q3_K_M",
        file_size_bytes=2_000_000_000,
    )
    result = CompatibilityResult(
        model=base,
        gguf_variant=synthetic,
        can_run=True,
        vram_required_bytes=3_000_000_000,
        vram_available_bytes=8_000_000_000,
    )

    attach_resolved_artifacts([result], [base, gguf_repo])

    assert result.artifact_model is gguf_repo
    assert result.artifact_variant is gguf_repo.gguf_variants[0]
