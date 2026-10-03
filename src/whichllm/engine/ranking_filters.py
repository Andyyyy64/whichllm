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

# Substring patterns in *names* that strongly suggest non-production usage.
_EXCLUDED_NAME_PATTERNS = (
    "tiny-",
    "-tiny",
    "tiny_",
    "_tiny",
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
    if org in _EXCLUDED_ORGS:
        return True
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in _EXCLUDED_NAME_PATTERNS:
        if pat in name:
            return True
    return False


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
    if profile == "math":
        # Math-tagged repos are a thin, mostly 2024-era pool (Qwen2.5-Math and
        # friends), so a name-only filter leaves --profile math near-empty.
        # Current math ability comes from general reasoning models, so keep the
        # specialists eligible and only exclude the other specializations.
        return not (tags - {"math"})
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
    "_is_excluded_model",
    "_is_gguf_only_backend",
    "_knowledge_capacity_b",
    "_matches_profile",
    "_passes_evidence_filter",
]
