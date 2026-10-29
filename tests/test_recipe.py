"""Run recipes follow the placement a recommendation was estimated with."""

from __future__ import annotations

import io
import json

from rich.console import Console

from whichllm.cli_models import _generate_chat_script
from whichllm.engine.placement import Placement
from whichllm.engine.recipe import build_recipe, python_gpu_layers
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.output.display import display_json, display_markdown, display_ranking


def _result(placement: Placement, *, estimated: bool = False, **kw):
    model = ModelInfo(
        id="ggml-org/gpt-oss-120b-GGUF",
        family_id="gpt-oss-120b",
        name="gpt-oss-120b-GGUF",
        parameter_count=116_800_000_000,
    )
    variant = GGUFVariant(
        filename="gpt-oss-120b-MXFP4.gguf",
        quant_type="MXFP4",
        file_size_bytes=63 * 10**9,
        is_estimated=estimated,
    )
    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        placement=placement,
        **kw,
    )


def _experts(n_cpu_moe):
    return Placement("expert_offload", 10, 50, n_cpu_moe=n_cpu_moe, num_layers=36)


def test_expert_offload_recipe_names_the_layer_count():
    recipe = build_recipe(_result(_experts(29)), 8192)
    assert recipe is not None
    assert recipe.llama_cpp == (
        "llama-server -hf ggml-org/gpt-oss-120b-GGUF:MXFP4 -c 8192 "
        "-ngl 99 --n-cpu-moe 29 -fa on"
    )
    assert "force MoE weights onto CPU: 29" in recipe.lm_studio
    assert recipe.ollama == "ollama run hf.co/ggml-org/gpt-oss-120b-GGUF:MXFP4"
    assert any("Ollama" in note for note in recipe.notes)


def test_unknown_split_keeps_every_expert_on_the_cpu():
    recipe = build_recipe(_result(_experts(None)), 4096)
    assert recipe is not None
    assert "--cpu-moe" in recipe.llama_cpp
    assert "--n-cpu-moe" not in recipe.llama_cpp
    # LM Studio has no all-experts toggle: the layer count goes to the maximum.
    assert recipe.lm_studio.endswith("force MoE weights onto CPU: 36")


def test_all_experts_without_a_layer_count_says_max():
    placement = Placement("expert_offload", 10, 50, n_cpu_moe=None, num_layers=None)
    recipe = build_recipe(_result(placement), 4096)
    assert recipe is not None
    assert recipe.lm_studio.endswith("force MoE weights onto CPU: max")


def test_ollama_note_only_when_the_model_is_split():
    full = build_recipe(_result(Placement("gpu", 60, 0, n_gpu_layers=36)), 4096)
    layers = build_recipe(
        _result(Placement("layer_offload", 40, 20, n_gpu_layers=24, num_layers=36)),
        4096,
    )
    assert full is not None and full.notes == ()
    assert layers is not None
    assert any("Ollama picks its own" in note for note in layers.notes)


def test_full_gpu_layer_offload_and_cpu_recipes():
    full = build_recipe(_result(Placement("gpu", 60, 0, n_gpu_layers=36)), 4096)
    layers = build_recipe(
        _result(Placement("layer_offload", 40, 20, n_gpu_layers=24, num_layers=36)),
        4096,
    )
    cpu = build_recipe(_result(Placement("cpu", 0, 60, n_gpu_layers=0)), 4096)
    assert full is not None and full.llama_cpp.endswith("-c 4096 -ngl 99 -fa on")
    assert layers is not None and layers.llama_cpp.endswith("-c 4096 -ngl 24")
    assert layers.lm_studio == "GPU Offload: 24 layers"
    assert cpu is not None and cpu.llama_cpp.endswith("-c 4096 -ngl 0")


def test_synthetic_variant_needs_a_resolved_gguf_repo():
    synthetic = _result(Placement("gpu", 60, 0), estimated=True)
    assert build_recipe(synthetic, 4096) is None

    synthetic.artifact_model = ModelInfo(
        id="bartowski/openai_gpt-oss-120b-GGUF",
        family_id="gpt-oss-120b",
        name="openai_gpt-oss-120b-GGUF",
        parameter_count=116_800_000_000,
    )
    synthetic.artifact_variant = GGUFVariant(
        filename="openai_gpt-oss-120b-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=63 * 10**9,
    )
    recipe = build_recipe(synthetic, 4096)
    assert recipe is not None
    assert "-hf bartowski/openai_gpt-oss-120b-GGUF:Q4_K_M" in recipe.llama_cpp


def test_python_layer_split_follows_the_placement():
    assert python_gpu_layers(_result(Placement("gpu", 60, 0))) == -1
    assert python_gpu_layers(_result(Placement("cpu", 0, 60))) == 0
    layers = Placement("layer_offload", 40, 20, n_gpu_layers=24, num_layers=36)
    assert python_gpu_layers(_result(layers)) == 24
    # llama-cpp-python cannot split experts: keep the same bytes on the GPU.
    assert python_gpu_layers(_result(_experts(29))) == 6
    no_shape = Placement("expert_offload", 10, 50)
    assert python_gpu_layers(_result(no_shape)) == -1


def test_chat_script_uses_the_layer_split_unless_cpu_only():
    model = _result(_experts(29)).model
    variant = _result(_experts(29)).gguf_variant
    assert "n_gpu_layers=6," in _generate_chat_script(
        model, variant, 4096, cpu_only=False, n_gpu_layers=6
    )
    assert "n_gpu_layers=0," in _generate_chat_script(
        model, variant, 4096, cpu_only=True, n_gpu_layers=6
    )
    assert "n_gpu_layers=-1," in _generate_chat_script(
        model, variant, 4096, cpu_only=False
    )


def _capture(render) -> str:
    import whichllm.output._console as console_mod

    buf = io.StringIO()
    original = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=200)
    try:
        render()
    finally:
        console_mod.console = original
    return buf.getvalue()


def test_outputs_carry_the_top_pick_recipe():
    result = _result(_experts(29))
    hw = HardwareInfo(ram_bytes=64 * 1024**3)

    data = json.loads(_capture(lambda: display_json([result], hw, 8192)))
    assert "--n-cpu-moe 29" in data["models"][0]["recipe"]["llama_cpp"]

    markdown = _capture(lambda: display_markdown([result], hw, context_length=8192))
    assert "```bash\nllama-server -hf ggml-org/gpt-oss-120b-GGUF:MXFP4" in markdown

    table = _capture(lambda: display_ranking([result], context_length=8192))
    assert "Run #1 with llama.cpp:" in table
    assert "--n-cpu-moe 29" in table
