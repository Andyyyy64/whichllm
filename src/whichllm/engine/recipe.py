"""How to run a recommendation the way its fit and speed were estimated.

A recipe turns a ranked result's placement into runtime settings: the
llama.cpp server command, the matching LM Studio load settings and the
Ollama command, with a note where a runtime cannot place weights the same
way. It needs a real GGUF repository; ranked safetensors repositories get
one only after artifact resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whichllm.engine.types import CompatibilityResult

# Above any real layer count: llama.cpp clamps -ngl to the model's layers.
_ALL_LAYERS = 99


@dataclass(frozen=True)
class Recipe:
    gguf_repo: str
    quant: str
    llama_cpp: str
    lm_studio: str
    ollama: str
    notes: tuple[str, ...] = field(default=())


def _gguf_source(result: CompatibilityResult) -> tuple[str, str] | None:
    if result.artifact_model is not None and result.artifact_variant is not None:
        return result.artifact_model.id, result.artifact_variant.quant_type
    variant = result.gguf_variant
    if variant is not None and not variant.is_estimated:
        return result.model.id, variant.quant_type
    return None


def build_recipe(result: CompatibilityResult, context_length: int) -> Recipe | None:
    """Runtime settings for a ranked result, or None without a known GGUF."""
    source = _gguf_source(result)
    if source is None:
        return None
    repo, quant = source
    placement = result.placement
    kind = placement.kind if placement else result.fit_type

    flags = [f"-c {context_length}"]
    notes: list[str] = []
    if kind in ("gpu", "full_gpu"):
        flags += [f"-ngl {_ALL_LAYERS}", "-fa on"]
        lm_studio = "GPU Offload: max"
    elif kind == "expert_offload" and placement is not None:
        layers = placement.n_cpu_moe
        all_experts = layers is None or layers >= (placement.num_layers or 0)
        flags += [
            f"-ngl {_ALL_LAYERS}",
            "--cpu-moe" if all_experts else f"--n-cpu-moe {layers}",
            "-fa on",
        ]
        lm_studio = (
            "GPU Offload: max, Force Model Expert Weights onto CPU: on"
            if all_experts
            else (
                "GPU Offload: max, Number of layers for which to force MoE "
                f"weights onto CPU: {layers}"
            )
        )
        notes.append(
            "Ollama offloads whole layers, not experts, so it runs slower than "
            "this estimate."
        )
    elif kind in ("layer_offload", "partial_offload"):
        layers = placement.n_gpu_layers if placement else None
        if layers is not None:
            flags.append(f"-ngl {layers}")
            lm_studio = f"GPU Offload: {layers} layers"
        else:
            lm_studio = "GPU Offload: as many layers as fit"
    else:
        flags.append("-ngl 0")
        lm_studio = "GPU Offload: 0"

    return Recipe(
        gguf_repo=repo,
        quant=quant,
        llama_cpp=f"llama-server -hf {repo}:{quant} " + " ".join(flags),
        lm_studio=lm_studio,
        ollama=f"ollama run hf.co/{repo}:{quant}",
        notes=tuple(notes),
    )


def python_gpu_layers(result: CompatibilityResult) -> int:
    """llama-cpp-python ``n_gpu_layers`` for a ranked result (-1 = all).

    llama-cpp-python cannot keep only the MoE experts in RAM, so an expert
    offload becomes the whole-layer split that keeps the same bytes on the
    GPU. That runs slower than the estimate; llama-server matches it.
    """
    placement = result.placement
    if placement is None or placement.kind == "gpu":
        return -1
    if placement.kind == "cpu":
        return 0
    if placement.kind == "layer_offload":
        return placement.n_gpu_layers if placement.n_gpu_layers is not None else -1
    total = placement.gpu_weight_bytes + placement.cpu_weight_bytes
    if not placement.num_layers or total <= 0:
        return -1
    return int(placement.num_layers * placement.gpu_weight_bytes / total)
