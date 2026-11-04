"""Which engine runs a ranked row here, and which one its recipe uses."""

from __future__ import annotations

import re
from pathlib import Path

from whichllm.engine.quantization import effective_quant_type
from whichllm.engine.types import CompatibilityResult
from whichllm.engine.lookup import lazy_lookup_params
from whichllm.engines.availability import availability
from whichllm.engines.cards import EVIDENCE, EngineCard
from whichllm.engines.detect import Installed
from whichllm.hardware.types import HardwareInfo

# Engines whichllm can write a recipe for. Jan and Lemonade load models by their
# own names rather than a Hugging Face repo and quant, and KoboldCpp needs a
# local file, so they are listed as able to run a row but never chosen for it.
RECIPE_ENGINES = ("llama.cpp", "ik_llama.cpp", "ollama", "lm-studio")

_MEMORY_FOR_PLACEMENT = {
    "gpu": "gpu",
    "full_gpu": "gpu",
    "expert_offload": "expert_offload",
    "layer_offload": "layer_offload",
    "partial_offload": "layer_offload",
    "cpu": "cpu",
    "cpu_only": "cpu",
}
_REPACKED_RE = re.compile(r"_R\d+$")


def _gguf_quant(result: CompatibilityResult) -> str | None:
    variant = result.artifact_variant or result.gguf_variant
    if variant is None:
        return None
    return effective_quant_type(result.artifact_model or result.model, variant).upper()


def reads_quant(card: EngineCard, quant: str, cards: tuple[EngineCard, ...]) -> bool:
    """Whether the engine loads a GGUF of this quant type.

    A type some engine adds beyond mainline llama.cpp (ik_llama.cpp's IQ4_KS,
    its repacked _R4 forms) loads only in an engine that lists it; any other
    type loads everywhere except where an engine says it is missing.
    """
    if card.gguf is None:
        return False
    base = _REPACKED_RE.sub("", quant)
    added = {t for c in cards if c.gguf for t in c.gguf.extra}
    if base in added or quant in added:
        return base in card.gguf.extra or quant in card.gguf.extra
    return quant not in card.gguf.missing and base not in card.gguf.missing


def engines_for(
    result: CompatibilityResult, hardware: HardwareInfo, cards: tuple[EngineCard, ...]
) -> list[EngineCard]:
    """Engines that run here, read the row's quant and hold its placement.

    The memory estimate leaves a large lookup table on disk, so a model with
    one needs an engine documented to read it from there.
    """
    quant = _gguf_quant(result)
    if quant is None:
        return []
    kind = result.placement.kind if result.placement else result.fit_type
    needed = {_MEMORY_FOR_PLACEMENT.get(kind, "gpu")}
    if lazy_lookup_params(result.artifact_model or result.model):
        needed.add("lazy_lookup")
    return [
        card
        for card in cards
        if card.kind == "runner"
        and needed <= set(card.memory)
        and reads_quant(card, quant, cards)
        and availability(card, hardware).runs
    ]


def pick_engine(
    candidates: list[EngineCard], installed: dict[str, Installed]
) -> EngineCard | None:
    """The engine a recipe uses: one already running, else installed, else the
    best evidence, with llama.cpp first among equals."""
    usable = [card for card in candidates if card.id in RECIPE_ENGINES]
    if not usable:
        return None

    def preference(card: EngineCard) -> tuple:
        found = installed.get(card.id)
        return (
            not (found and found.running),
            found is None,
            EVIDENCE.index(card.evidence),
            card.id != "llama.cpp",
            card.id,
        )

    return min(usable, key=preference)


def assign_engines(
    results: list[CompatibilityResult],
    hardware: HardwareInfo,
    cards: tuple[EngineCard, ...],
    installed: dict[str, Installed] | None,
) -> None:
    """Set each row's engine, the engines that can run it, and whether the
    chosen one is installed. ``installed`` is None when that is unknown, as
    for a simulated GPU."""
    for result in results:
        candidates = engines_for(result, hardware, cards)
        result.engines = tuple(card.id for card in candidates)
        chosen = pick_engine(candidates, installed or {})
        result.engine = chosen.id if chosen else None
        if chosen is None or installed is None:
            continue
        found = installed.get(chosen.id)
        result.engine_installed = found is not None
        if found is not None and found.path:
            result.engine_binary = Path(found.path).stem


__all__ = [
    "RECIPE_ENGINES",
    "assign_engines",
    "engines_for",
    "pick_engine",
    "reads_quant",
]
