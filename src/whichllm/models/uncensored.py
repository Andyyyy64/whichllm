"""Spot uncensored, abliterated and unfiltered model uploads.

Hugging Face has no field for "this model has had its refusals removed", so the
marker is the name (``Qwen3-8B-abliterated``, ``Llama-3.1-8B-Instruct-heretic``,
``Dolphin3.0-...``) or a tag the uploader added. A quantized upload counts when
the repository it quantizes does, since converters often keep a clean name.

Modes for the ranking, from ``--uncensored`` and ``--include-uncensored``:

- ``off``: today's behaviour. Uncensored derivatives compete with a small
  penalty, because they ride on the base model's benchmark line without being
  benchmarked themselves.
- ``include``: no penalty, so uncensored models rank next to the originals.
- ``only``: nothing but uncensored models, without the penalty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from whichllm.models.types import ModelInfo

UNCENSORED_OFF = "off"
UNCENSORED_INCLUDE = "include"
UNCENSORED_ONLY = "only"
UNCENSORED_MODES = (UNCENSORED_OFF, UNCENSORED_INCLUDE, UNCENSORED_ONLY)

# Unambiguous words: a match anywhere in the repository name is enough.
_SUBSTRING_MARKERS = (
    "uncensor",
    "abliterat",
    "abliter",
    "obliterat",
    "unfiltered",
    "heretic",
)
# Names that mark an uncensored line only at the start of a word, so that
# "Dolphin3.0" matches and an unrelated word that merely contains it does not.
_WORD_START_RE = re.compile(r"(?i)(?:^|[-_.\s])(dolphin|lexi|josiefied)")
# Hugging Face tags that say the same thing.
_TAG_MARKERS = ("uncensor", "abliterat", "heretic", "unfiltered")

# Name fragments that make a derivative uncensored, for the penalty rule in
# ``ranking_filters``. Other derivative names (roleplay, nsfw, merges) are not
# about refusals and keep their penalty in every mode.
UNCENSORED_PENALTY_PATTERNS = frozenset(
    {"heretic", "abliterat", "abliter", "uncensored", "obliterat", "lexi"}
)


def _repo_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower()


def _name_is_uncensored(model_id: str) -> bool:
    name = _repo_name(model_id)
    if any(marker in name for marker in _SUBSTRING_MARKERS):
        return True
    return _WORD_START_RE.search(name) is not None


def is_uncensored(
    model_id: str,
    tags: Iterable[str] = (),
    base_model: str | None = None,
) -> bool:
    """True when the repository, its tags or the repo it derives from say so."""
    if not model_id:
        return False
    if _name_is_uncensored(model_id):
        return True
    if base_model and _name_is_uncensored(base_model):
        return True
    for tag in tags:
        lowered = tag.lower()
        if any(marker in lowered for marker in _TAG_MARKERS):
            return True
    return False


def model_is_uncensored(model: ModelInfo) -> bool:
    """:func:`is_uncensored` for a catalog entry."""
    return is_uncensored(model.id, model.tags, model.base_model)


def uncensored_mode(include: bool, only: bool) -> str:
    """The ranking mode for the two CLI flags (``only`` wins when both are set)."""
    if only:
        return UNCENSORED_ONLY
    if include:
        return UNCENSORED_INCLUDE
    return UNCENSORED_OFF


__all__ = [
    "UNCENSORED_INCLUDE",
    "UNCENSORED_MODES",
    "UNCENSORED_OFF",
    "UNCENSORED_ONLY",
    "UNCENSORED_PENALTY_PATTERNS",
    "is_uncensored",
    "model_is_uncensored",
    "uncensored_mode",
]
