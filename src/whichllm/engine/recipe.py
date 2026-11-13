"""How to run a recommendation the way its fit and speed were estimated.

A recipe turns a ranked result's placement into runtime settings: the
llama.cpp server command, the matching LM Studio load settings and the
Ollama command. Ollama picks its own GPU/CPU split, so a recipe that splits
the model says its speed can differ. It needs a real GGUF repository; ranked
safetensors repositories get one only after artifact resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.types import CompatibilityResult
from whichllm.engines.cards import card_by_id

# Above any real layer count: llama.cpp clamps -ngl to the model's layers.
_ALL_LAYERS = 99

# Ollama 0.40 sets the split itself (on an RTX 5080 it kept the experts of 14
# of Qwen3.6-35B-A3B's 40 layers in RAM and ran faster than llama-server),
# and older releases offload whole layers. Either way it is not this estimate.
_OLLAMA_SPLIT_NOTE = (
    "Ollama picks its own GPU/CPU split, so its speed can differ from this estimate."
)


@dataclass(frozen=True)
class Recipe:
    gguf_repo: str
    quant: str
    llama_cpp: str
    lm_studio: str
    ollama: str
    notes: tuple[str, ...] = field(default=())
    # The engine chosen for this row and its command (LM Studio's is its
    # load settings).
    engine: str = "llama.cpp"
    command: str = ""


def _gguf_source(result: CompatibilityResult) -> tuple[str, str, str] | None:
    if result.artifact_model is not None and result.artifact_variant is not None:
        variant = result.artifact_variant
        return result.artifact_model.id, variant.quant_type, variant.filename
    variant = result.gguf_variant
    if variant is not None and not variant.is_estimated:
        return result.model.id, variant.quant_type, variant.filename
    return None


def build_recipe(result: CompatibilityResult, context_length: int) -> Recipe | None:
    """Runtime settings for a ranked result, or None without a known GGUF."""
    source = _gguf_source(result)
    if source is None:
        return None
    repo, quant, filename = source
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
        # LM Studio 0.4 has no all-experts switch; its layer count at the
        # model's layer count (the slider's maximum) does the same.
        cpu_layers = placement.num_layers if all_experts else layers
        lm_studio = (
            "GPU Offload: max, Number of layers for which to force MoE weights "
            f"onto CPU: {cpu_layers or 'max'}"
        )
        notes.append(_OLLAMA_SPLIT_NOTE)
    elif kind in ("layer_offload", "partial_offload"):
        layers = placement.n_gpu_layers if placement else None
        if layers is not None:
            flags.append(f"-ngl {layers}")
            lm_studio = f"GPU Offload: {layers} layers"
        else:
            lm_studio = "GPU Offload: as many layers as fit"
        notes.append(_OLLAMA_SPLIT_NOTE)
    else:
        flags.append("-ngl 0")
        lm_studio = "GPU Offload: 0"

    server = "llama serve" if result.engine_binary == "llama" else "llama-server"
    llama_cpp = f"{server} -hf {repo}:{quant} " + " ".join(flags)
    ollama = f"ollama run hf.co/{repo}:{quant}"
    engine = result.engine or "llama.cpp"
    commands = {
        "llama.cpp": llama_cpp,
        # ik_llama.cpp predates "-hf repo:quant" and takes the repo and the
        # file apart; the placement flags are the same.
        "ik_llama.cpp": f"llama-server -hfr {repo} -hff {filename} " + " ".join(flags),
        "ollama": ollama,
        "lm-studio": lm_studio,
    }
    return Recipe(
        gguf_repo=repo,
        quant=quant,
        llama_cpp=llama_cpp,
        lm_studio=lm_studio,
        ollama=ollama,
        notes=tuple(notes),
        engine=engine,
        command=commands.get(engine, llama_cpp),
    )


@dataclass(frozen=True)
class ServerRecipe:
    """How a GPU server (vLLM, SGLang) serves a repository of Hugging Face
    weights, and the commands of the other servers that read it."""

    engine: str
    repo: str
    weights: str  # the format, as effective_quant_type names it
    command: str
    others: tuple[tuple[str, str], ...] = field(default=())  # (engine, command)


def _serve(engine: str, repo: str) -> str | None:
    card = card_by_id(engine)
    if card is None or not card.safetensors or not card.serve:
        return None
    return card.serve.format(repo=repo)


def build_server_recipe(result: CompatibilityResult) -> ServerRecipe | None:
    """The serve command for a row of Hugging Face weights, or None for a GGUF
    row or one no GPU server here runs."""
    if result.gguf_variant is not None or result.artifact_variant is not None:
        return None
    if result.engine is None:
        return None
    repo = result.model.id
    command = _serve(result.engine, repo)
    if command is None:
        return None
    others = tuple(
        (engine, other)
        for engine in result.engines
        if engine != result.engine
        for other in [_serve(engine, repo)]
        if other is not None
    )
    return ServerRecipe(
        engine=result.engine,
        repo=repo,
        weights=effective_quant_type(result.model, None),
        command=command,
        others=others,
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
