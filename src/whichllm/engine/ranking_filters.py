"""Filtering helpers for ranking candidate models."""

from __future__ import annotations

import re

from whichllm.constants import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
)
from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import ModelInfo

# Pre-compile lineage regex tables once at import time.
_LINEAGE_REGEX: dict[str, list[tuple[re.Pattern[str], int]]] = {
    family: [(re.compile(pat), idx) for pat, idx in entries]
    for family, entries in MODEL_LINEAGE_VERSIONS.items()
}
_LINEAGE_FAMILY_MAX: dict[str, int] = {
    family: max(idx for _, idx in entries) for family, entries in _LINEAGE_REGEX.items()
}

# Orgs whose repositories ship CI fixtures, deprecated research artifacts, or
# debug binaries that are not viable production LLMs. Exclude them outright so
# they cannot occupy ranking slots regardless of download counts.
_EXCLUDED_ORGS = frozenset(
    {
        "openai-community",  # gpt2 family, 2019 research
        "distilbert",  # distilgpt2 etc.
        "facebook",  # opt-125m research scaffolds
        "EleutherAI",  # pythia/gpt-neo research
        "trl-internal-testing",  # TRL CI fixtures
        "hmellor",  # random tiny test models
        "HuggingFaceH4",  # often staging / fixtures
        "transformersbook",
        "togethercomputer",  # mostly inference endpoints, no GGUFs
    }
)
# hf-, trl-, peft- and optimum-intel-internal-testing keep CI fixtures.
_FIXTURE_ORG_SUFFIX = "-internal-testing"

# Substring patterns in *names* that strongly suggest non-production usage.
# A bare "tiny" is not one: Ling-3.0-tiny, Granite 4.0 Tiny and DeepSeek-VL2-tiny
# are real models, and fixtures score too low to rank without a name rule.
_EXCLUDED_NAME_PATTERNS = (
    "test-only",
    "debug-",
    "playground",
    "-fixture",
    "for-testing",
    "tiny-random",
    "ci-",
)

# Naming patterns that indicate a fine-tune / merge / "uncensoring" derivative
# of a real base model. These derivatives inherit the base model's benchmark
# score via line_interp, but the derivative itself is rarely benchmarked
# independently and frequently degrades quality. Apply a soft score penalty
# rather than full exclusion so they can still surface when nothing better is
# available.
_DUBIOUS_DERIVATIVE_PATTERNS = (
    "heretic",
    "abliterat",
    "uncensored",
    "obliterat",
    "abliter",
    "horror",
    "erotic",
    "nsfw",
    "rp-",
    "-rp",
    "roleplay",
    "darkidol",
    "darkforest",
    "tiefigh",
    "smaug",
    "personalityengine",
    "lexi",
    "violence",
    "violet",
    "schizo",
    "dark-",
    "twilight",
    "celeste",
    "midnight-rose",
    "moistral",
    "stheno",
    "fimbulvetr",
    "wizard-vicuna",
    "kunoichi",
)


def _derivative_name_penalty(model_id: str) -> float:
    """Return a score penalty (in raw quality points) for fine-tune /
    "uncensored" / merge derivatives that ride on a real base model's
    benchmark line. The penalty is gentle (≤ 12pt) so a derivative can
    still win when its size class has no better option.
    """
    if not model_id:
        return 0.0
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in _DUBIOUS_DERIVATIVE_PATTERNS:
        if pat in name:
            return -10.0
    return 0.0


def _is_excluded_model(model_id: str) -> bool:
    """Return True for CI/research/fixture models that should never rank."""
    if not model_id:
        return True
    org = model_id.split("/", 1)[0] if "/" in model_id else ""
    if org in _EXCLUDED_ORGS or org.endswith(_FIXTURE_ORG_SUFFIX):
        return True
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in _EXCLUDED_NAME_PATTERNS:
        if pat in name:
            return True
    return False


# Speculative-decoding draft heads (DFlash, "-draft" uploads) only run next to
# their target model; their size comes from the name, so they would otherwise
# pass for the full model.
DRAFT_NAME_RE = re.compile(r"(?i)dflash|(?:^|[-_.])draft(?:[-_.]|$)")
# Diffusion language models (DiffusionGemma, LLaDA, Dream) need their own
# sampler; llama.cpp, LM Studio and Ollama do not run them as chat models.
_DIFFUSION_RE = re.compile(r"(?i)diffusion|(?:^|[-_/])(llada|dream-v\d)")


def _is_auxiliary_model(model: ModelInfo) -> bool:
    """A draft head or a diffusion LM: not something to chat with directly."""
    name = model.id.rsplit("/", 1)[-1]
    if DRAFT_NAME_RE.search(name) or _DIFFUSION_RE.search(name):
        return True
    model_type = model.shape.model_type if model.shape else ""
    return "diffusion" in (model_type or "").lower() or any(
        "diffusion" in tag.lower() for tag in model.tags
    )


# No quant shrinks a model tenfold, since even IQ1_S keeps 1.6 bits of the base's
# 16. A derivative below this share is a head, a broken count or a name misread
# as a size, and every estimate built on it is fiction: one such row claimed
# 14,000 tok/s on an 8 GB Mac.
_MIN_SHARE_OF_BASE = 0.1


def _understates_its_base(model: ModelInfo, params_by_repo: dict[str, int]) -> bool:
    """True when a derivative claims a tiny fraction of the model it derives from."""
    if not model.base_model or not model.parameter_count:
        return False
    base_params = params_by_repo.get(model.base_model.casefold(), 0)
    return model.parameter_count < base_params * _MIN_SHARE_OF_BASE


# Checkpoints vLLM serves through its Marlin kernels, which need Ampere or newer
# (compute capability 8.0). Blackwell (10.0+) runs FP4 natively.
_AMPERE_FORMATS = frozenset({"NVFP4", "FP8"})
_MIN_COMPUTE_CAPABILITY = (8, 0)
_NATIVE_FP4_CAPABILITY = (10, 0)


def _non_gguf_format_note(quant_type: str, hardware: HardwareInfo) -> str | None:
    """Why a non-GGUF checkpoint cannot run here (``""`` if fine), or a note.

    Returns ``None`` when the format cannot run on this machine's GPU, an empty
    string when it runs as expected, and a note when it runs with a caveat.
    """
    if quant_type not in _AMPERE_FORMATS:
        return ""
    capabilities = [
        g.compute_capability
        for g in hardware.gpus
        if g.vendor == "nvidia" and g.compute_capability
    ]
    if not capabilities:
        return ""  # unknown: do not block
    best = max(capabilities)
    if best < _MIN_COMPUTE_CAPABILITY:
        return None
    if quant_type == "NVFP4" and best < _NATIVE_FP4_CAPABILITY:
        return (
            "NVFP4 runs through vLLM's weight-only FP4 (Marlin) kernels here; "
            "only Blackwell GPUs compute in FP4 natively"
        )
    return ""


def _generation_bonus(model_id: str) -> float:
    """Return a small additive bonus reflecting model generation recency."""
    if not model_id:
        return 0.0
    lower = model_id.lower()
    best_bonus = 0.0
    for _family, patterns in _LINEAGE_REGEX.items():
        for regex, idx in patterns:
            if regex.search(lower):
                top = _LINEAGE_FAMILY_MAX[_family]
                if top <= 1:
                    contribution = 0.0
                else:
                    # Map oldest -> -PENALTY_MAX, newest -> +BONUS_MAX.
                    norm = (idx - 1) / (top - 1)  # 0 .. 1
                    span = MODEL_GENERATION_BONUS_MAX + MODEL_GENERATION_PENALTY_MAX
                    contribution = norm * span - MODEL_GENERATION_PENALTY_MAX
                if abs(contribution) > abs(best_bonus):
                    best_bonus = contribution
                break  # first match wins for this family
    return best_bonus


def _lineage_positions(model_id: str) -> dict[str, int]:
    """Every lineage family the id matches, with its generation index."""
    lower = model_id.lower()
    positions: dict[str, int] = {}
    for family, patterns in _LINEAGE_REGEX.items():
        for regex, idx in patterns:
            if regex.search(lower):
                positions[family] = idx
                break
    return positions


def _same_size(a: ModelInfo, b: ModelInfo) -> bool:
    """Same size class: total parameters within 25%, and dense vs MoE alike."""
    if not a.parameter_count or not b.parameter_count:
        return False
    ratio = a.parameter_count / b.parameter_count
    if not 0.8 <= ratio <= 1.25 or a.is_moe != b.is_moe:
        return False
    if a.is_moe and a.parameter_count_active and b.parameter_count_active:
        active = a.parameter_count_active / b.parameter_count_active
        return 0.67 <= active <= 1.5
    return True


def _newer_generation(newer: ModelInfo, older: ModelInfo) -> bool:
    """True when ``newer`` is a later generation of ``older``'s model line.

    Both ids must belong to exactly one lineage family (a distill such as
    DeepSeek-R1-Distill-Qwen matches two and is left alone), carry the same
    specialization tags, and be the same size.
    """
    new_pos = _lineage_positions(newer.id)
    old_pos = _lineage_positions(older.id)
    if len(new_pos) != 1 or new_pos.keys() != old_pos.keys():
        return False
    family = next(iter(new_pos))
    if new_pos[family] <= old_pos[family]:
        return False
    if _detect_specializations(newer.id) != _detect_specializations(older.id):
        return False
    return _same_size(newer, older)


def _detect_specializations(model_id: str) -> set[str]:
    """モデルIDから用途特化タグを検出する。"""
    lower = model_id.lower()
    tags: set[str] = set()
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.add("coding")
    if re.search(r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)", lower):
        tags.add("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.add("math")
    return tags


def _matches_profile(model: ModelInfo, task_profile: str) -> bool:
    """指定プロファイルにモデルが合致するか判定する。"""
    profile = task_profile.lower()
    tags = _detect_specializations(model.id)
    if profile == "any":
        return True
    if profile == "general":
        return len(tags) == 0
    if profile in ("math", "coding"):
        # Name-tagged specialists (*-Coder, *-Math) are a thin and often older
        # pool, while current general models (gpt-oss, Devstral, Qwen3.6) are
        # among the strongest coders and reasoners. Keep both eligible and only
        # exclude the other specializations.
        return not (tags - {profile})
    if profile == "vision":
        # Natively multimodal models (Gemma 3/4, Qwen3.5/3.6) carry no "VL" in
        # their names; the Hub's image-text-to-text tag marks them.
        return "vision" in tags or "image-text-to-text" in model.tags
    return profile in tags


def _effective_params_b(model: ModelInfo) -> float:
    """Return effective parameter size in billions."""
    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active / 1e9
    return model.parameter_count / 1e9


def _knowledge_capacity_b(model: ModelInfo) -> float:
    """Return the knowledge capacity in billions for size filtering."""
    return model.parameter_count / 1e9


def _passes_evidence_filter(source: str, evidence_filter: str) -> bool:
    """判定根拠フィルタに合致するかを返す。"""
    mode = evidence_filter.lower()
    if mode == "strict":
        return source == "direct"
    if mode == "base":
        return source in {"direct", "variant", "base_model"}
    return True


def _is_gguf_only_backend(hardware: HardwareInfo) -> bool:
    """実行基盤の都合でGGUFのみを許可すべきか判定する。"""
    # Apple Silicon(macOS/Metal)とCPU-onlyは、実運用の安定性を優先してGGUFに限定する。
    if hardware.os == "darwin":
        return True
    if not hardware.gpus:
        return True

    # Linux + NVIDIA (CUDA) は AWQ/GPTQ 含む非GGUFも許可する。
    has_linux_nvidia = hardware.os == "linux" and any(
        g.vendor == "nvidia" for g in hardware.gpus
    )
    return not has_linux_nvidia


__all__ = [
    "_DUBIOUS_DERIVATIVE_PATTERNS",
    "_EXCLUDED_NAME_PATTERNS",
    "_EXCLUDED_ORGS",
    "_derivative_name_penalty",
    "_detect_specializations",
    "_effective_params_b",
    "_generation_bonus",
    "_is_auxiliary_model",
    "_is_excluded_model",
    "_is_gguf_only_backend",
    "_knowledge_capacity_b",
    "_matches_profile",
    "_passes_evidence_filter",
]
