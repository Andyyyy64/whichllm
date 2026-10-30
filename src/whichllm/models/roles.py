"""Find pretrain-only (base) checkpoints, which are not chat models.

Publishers rarely label a base checkpoint in metadata (Qwen's base repos even
carry the ``conversational`` tag), but they do publish an instruct version
derived from it. A checkpoint counts as a base when:

- its name ends in ``-Base`` or ``-pt`` (Qwen3 and Gemma conventions), or
- the same organisation publishes an instruct, ``-it`` or chat model that
  declares it as its fine-tune parent, or that is named after it
  (``Qwen2.5-7B`` → ``Qwen2.5-7B-Instruct``).

A few older bases fit neither rule, because their chat release has another
name or the catalog does not list it, so they are named in ``_KNOWN_BASES``.

A quantized or converted upload counts as a base when its model name is the
name of such a base. Names that carry an instruct marker are never bases;
uploaders often tag the wrong ``base_model``, so tags alone do not decide.
"""

from __future__ import annotations

import re

from whichllm.models.types import ModelInfo

_INSTRUCT_MARKER = re.compile(r"(?i)(^|[-_.])(instruct|it|chat|thinking)([-_.]|$)")
_NAMED_SIBLING = re.compile(r"(?i)^(.+?)[-_.](instruct|it|chat)$")
_BASE_SUFFIX = re.compile(r"(?i)[-_](base|pt)$")
_CONVERSION_SUFFIX = re.compile(
    r"(?i)([-_.](gguf|awq|gptq|fp8|fp16|bf16|mlx|exl2|int4|int8|\d+bit))+$"
)
_FINETUNE_TAG = "base_model:finetune:"
# Lower-case repo ids. Olmo's chat release is Olmo-3-7B-Instruct, Mistral's is
# Mistral-7B-Instruct-v0.1, Llama 2's is Llama-2-7b-chat-hf, and the Gemma -it
# releases are often outside the catalog.
_KNOWN_BASES = frozenset(
    {
        "allenai/olmo-3-1025-7b",
        "google/gemma-2-2b",
        "google/gemma-2b",
        "google/gemma-7b",
        "google/paligemma-3b-pt-224",
        "meta-llama/llama-2-7b-hf",
        "mistralai/mistral-7b-v0.1",
    }
)


def _org(model_id: str) -> str:
    return model_id.split("/", 1)[0].lower()


def _name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _subject_name(model_id: str) -> str:
    """Model name without conversion suffixes such as -GGUF or -AWQ."""
    return _CONVERSION_SUFFIX.sub("", _name(model_id)).lower()


def _is_instruct(model_id: str) -> bool:
    return bool(_INSTRUCT_MARKER.search(_name(model_id)))


def find_base_checkpoints(models: list[ModelInfo]) -> set[str]:
    """Ids of models in this catalog that are pretrain-only checkpoints."""
    ids = {m.id.lower(): m.id for m in models}
    base_ids: set[str] = set()

    for model in models:
        if not _is_instruct(model.id):
            continue
        org = _org(model.id)
        parents = [
            tag[len(_FINETUNE_TAG) :]
            for tag in model.tags
            if tag.startswith(_FINETUNE_TAG)
        ]
        for parent in parents:
            if _org(parent) == org and not _is_instruct(parent):
                base_ids.add(parent.lower())
        # Named sibling: org/Name-Instruct implies org/Name is its base. A
        # dated refresh (Qwen3-4B-Instruct-2507) says nothing about Qwen3-4B,
        # so the marker must end the name.
        sibling = _NAMED_SIBLING.match(_name(model.id))
        if sibling:
            candidate = f"{model.id.split('/', 1)[0]}/{sibling.group(1)}".lower()
            if candidate in ids and not _is_instruct(candidate):
                base_ids.add(candidate)

    for model in models:
        if not _is_instruct(model.id) and _BASE_SUFFIX.search(
            _CONVERSION_SUFFIX.sub("", _name(model.id))
        ):
            base_ids.add(model.id.lower())
    base_ids |= _KNOWN_BASES & ids.keys()

    # A fine-tune of a -Base checkpoint is the post-trained model (Qwen3-4B
    # from Qwen3-4B-Base), whatever else refers to it.
    by_id = {m.id.lower(): m for m in models}
    base_ids = {
        i
        for i in base_ids
        if not (
            i in by_id
            and any(
                tag.startswith(_FINETUNE_TAG)
                and _BASE_SUFFIX.search(tag[len(_FINETUNE_TAG) :])
                for tag in by_id[i].tags
            )
        )
    }

    base_names = {_subject_name(i) for i in base_ids}
    result = {ids[i] for i in base_ids if i in ids}
    for model in models:
        if (
            model.id not in result
            and not _is_instruct(model.id)
            and model.base_model_relation == "quantized"
            and _subject_name(model.id) in base_names
        ):
            result.add(model.id)
    return result
