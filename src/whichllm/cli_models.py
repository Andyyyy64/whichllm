"""Model loading, lookup, and runtime script helpers for the CLI."""

from __future__ import annotations

import re
import sys

import typer

from whichllm.cli_shared import _format_fetch_error, _run_async, console
from whichllm.models.types import ModelInfo


_SIZE_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)([bm])$", re.IGNORECASE)


_ID_SIZE_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)(b|m)(?:[-_.]|$)", re.IGNORECASE)

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Hugging Face uses these responses for missing and private repositories.
_HF_MISSING_REPO_MESSAGES = (
    "invalid username or password",
    "invalid credentials",
    "repository not found",
)


def _load_models(refresh: bool, include_vision: bool = True):
    """Load models from cache or fetch from HuggingFace."""
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts

    cached_data = None if refresh else load_cache()
    if cached_data is not None:
        return dicts_to_models(cached_data)
    try:
        models = _run_async(fetch_models(include_vision=include_vision))
        save_cache(models_to_dicts(models))
        return models
    except Exception as e:
        console.print(f"[red]Error fetching models:[/] {_format_fetch_error(e)}")
        sys.exit(1)


def _parse_size_tokens(
    terms: list[str],
) -> tuple[list[str], float | None]:
    """Split query terms into non-size terms and an optional size in billions.

    Returns (remaining_terms, size_b) where size_b is None if no size token
    was found.  Only the first size token is used; subsequent size tokens are
    kept as plain text terms.  Handles 'b' (billions) and 'm' (millions).
    """
    remaining = []
    size_b: float | None = None
    for t in terms:
        m = _SIZE_TOKEN_RE.match(t)
        if m and size_b is None:
            value = float(m.group(1))
            if value <= 0:
                remaining.append(t)
                continue
            unit = m.group(2).lower()
            size_b = value if unit == "b" else value / 1000.0
        else:
            remaining.append(t)
    return remaining, size_b


def _extract_id_size_b(model_id: str) -> float | None:
    """Extract the size label from a model ID string, in billions.

    Scans for patterns like '7B', '1.7B', '500M' at word boundaries in the
    model ID.  Returns the first match converted to billions, or None.
    """
    for m in _ID_SIZE_RE.finditer(model_id):
        value = float(m.group(1))
        if value <= 0:
            continue
        unit = m.group(2).lower()
        return value if unit == "b" else value / 1000.0
    return None


def _size_compatible(model: ModelInfo, size_b: float) -> bool:
    """Check whether a model's parameter count is compatible with a query size.

    Uses a tolerance band of [0.7x, 1.5x] to accommodate rounding differences
    (e.g. a 7B query matching a model with 7.6B actual parameters) while
    rejecting adjacent model sizes (e.g. 7B vs 4B or 12B).
    """
    if model.parameter_count <= 0:
        return True
    actual_b = model.parameter_count / 1e9
    ratio = actual_b / size_b
    return 0.7 <= ratio <= 1.5


def _search_model(models: list, model_name: str):
    """Search for a model by name/ID. Returns single model or exits."""
    query_lower = model_name.lower()
    terms = query_lower.split()
    size_b = None

    matches = [m for m in models if m.id.lower() == query_lower]
    if not matches:
        matches = [m for m in models if m.id.lower().endswith("/" + query_lower)]
    if not matches:
        text_terms, size_b = _parse_size_tokens(terms)
        matches = [
            m
            for m in models
            if all(t in m.id.lower() for t in text_terms)
            and (size_b is None or _size_compatible(m, size_b))
        ]

    if not matches:
        console.print(f"[red]No model found matching '{model_name}'.[/]")
        suggestions = [m for m in models if any(t in m.id.lower() for t in terms)]
        if suggestions:
            suggestions.sort(key=lambda m: m.downloads, reverse=True)
            console.print("\n[yellow]Did you mean:[/]")
            for m in suggestions[:5]:
                p = (
                    f"{m.parameter_count / 1e9:.1f}B"
                    if m.parameter_count >= 1e9
                    else f"{m.parameter_count / 1e6:.0f}M"
                )
                console.print(f"  • {m.id} ({p})")
        raise typer.Exit(code=1)

    if size_b is not None:

        def _sort_key(m: ModelInfo) -> tuple:
            id_size = _extract_id_size_b(m.id)
            has_id_size = id_size is not None
            id_dist = abs(id_size - size_b) if has_id_size else float("inf")
            pc_dist = (
                abs(m.parameter_count / 1e9 - size_b)
                if m.parameter_count > 0
                else float("inf")
            )
            return (
                0 if (has_id_size or m.parameter_count > 0) else 1,
                id_dist,
                pc_dist,
                -m.downloads,
            )

        matches.sort(key=_sort_key)
    else:
        matches.sort(key=lambda m: m.downloads, reverse=True)
    model = matches[0]
    if len(matches) > 1:
        console.print(f"[dim]Found {len(matches)} matches, using: {model.id}[/]")
    return model


def _pick_gguf_variant(model, quant_filter: str | None = None):
    """Pick the best GGUF variant for a model."""
    from whichllm.constants import QUANT_PREFERENCE_ORDER

    if not model.gguf_variants:
        return None

    if quant_filter:
        for v in model.gguf_variants:
            if v.quant_type.upper() == quant_filter.upper():
                return v
        console.print(
            f"[yellow]Warning:[/] {quant_filter} not available, using best match."
        )

    # Pick by preference order
    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def _resolve_model_deps(model, variant) -> tuple[list[str], str]:
    """Determine pip dependencies and script type for a model.

    Returns (deps, script_type) where script_type is 'gguf' or 'transformers'.
    """
    if variant:
        return ["llama-cpp-python", "huggingface-hub"], "gguf"

    from whichllm.engine.quantization import infer_non_gguf_quant_type

    qt = infer_non_gguf_quant_type(model.id)
    base = ["transformers", "torch", "accelerate"]
    if qt == "AWQ":
        return [*base, "autoawq"], "transformers"
    if qt == "GPTQ":
        return [*base, "auto-gptq"], "transformers"
    return base, "transformers"


def _generate_chat_script(model, variant, context_length: int, cpu_only: bool) -> str:
    """Generate a self-contained Python chat script for any model type."""
    if variant:
        n_gpu = 0 if cpu_only else -1
        return f"""\
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_id = {model.id!r}
filename = {variant.filename!r}
quant_type = {variant.quant_type!r}
print(f"Downloading {{model_id}} ({{quant_type}})...")
model_path = hf_hub_download(repo_id=model_id, filename=filename)
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print("Ready! Type 'exit' to quit.\\n")
messages = []
while True:
    try:
        user_input = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if user_input.strip().lower() in ("exit", "quit", "q"):
        break
    if not user_input.strip():
        continue
    messages.append({{"role": "user", "content": user_input}})
    response = llm.create_chat_completion(messages=messages, stream=True)
    full = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
            full += content
    print()
    messages.append({{"role": "assistant", "content": full}})
print("\\nBye!")
"""

    device_map = '"cpu"' if cpu_only else '"auto"'
    dtype = "torch.float32" if cpu_only else '"auto"'
    return f"""\
import shutil
import tempfile
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = {model.id!r}
offload_folder = tempfile.mkdtemp(prefix="whichllm_transformers_offload_")
try:
    print(f"Loading {{model_id}}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={device_map},
        torch_dtype={dtype},
        trust_remote_code=True,
        offload_folder=offload_folder,
    )
    print("Ready! Type 'exit' to quit.\\n")
    messages = []
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in ("exit", "quit", "q"):
            break
        if not user_input.strip():
            continue
        messages.append({{"role": "user", "content": user_input}})
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=model.generate,
            kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer),
        )
        thread.start()
        full = ""
        for text in streamer:
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        messages.append({{"role": "assistant", "content": full}})
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
"""


def _looks_like_hf_repo_id(value: str) -> bool:
    """Return whether a CLI value has the shape of a Hugging Face repo ID."""
    return _HF_REPO_ID_RE.fullmatch(value) is not None


def _hf_error_message(error: Exception) -> str:
    """Return the Hugging Face error message from a failed response, if any."""
    response = getattr(error, "response", None)
    if response is None:
        return ""
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    try:
        text = response.text
    except Exception:
        return ""
    return text.strip() if isinstance(text, str) else ""


def _hf_reports_missing_repo(error: Exception) -> bool:
    """Whether Hugging Face refused the request because the repo is not visible."""
    message = _hf_error_message(error).casefold()
    return any(marker in message for marker in _HF_MISSING_REPO_MESSAGES)


def _raise_repo_fetch_error(model_id: str, error: Exception) -> None:
    """Print a repository-specific fetch error and exit the CLI."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 404 or _hf_reports_missing_repo(error):
        console.print(
            f"[red]Repository not found on Hugging Face:[/] {model_id} "
            "(it does not exist, or it is private or gated)."
        )
    elif status_code in {401, 403}:
        console.print(
            f"[red]Cannot access Hugging Face repository:[/] {model_id} "
            "(it may be private or gated)."
        )
    else:
        console.print(
            f"[red]Error fetching Hugging Face repository '{model_id}':[/] "
            f"{_format_fetch_error(error)}"
        )
    raise typer.Exit(code=1)


__all__ = [
    "_load_models",
    "_parse_size_tokens",
    "_extract_id_size_b",
    "_size_compatible",
    "_search_model",
    "_pick_gguf_variant",
    "_resolve_model_deps",
    "_generate_chat_script",
    "_looks_like_hf_repo_id",
    "_hf_error_message",
    "_hf_reports_missing_repo",
    "_raise_repo_fetch_error",
]
