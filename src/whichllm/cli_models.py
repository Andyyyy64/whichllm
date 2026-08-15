"""Model loading, lookup, and runtime script helpers for the CLI."""

from __future__ import annotations

import re
import sys

import typer

from whichllm.cli_shared import _format_fetch_error, _run_async, console
from whichllm.models.types import ModelInfo


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


_SIZE_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)([bm])$", re.IGNORECASE)


def _parse_size_tokens(
    terms: list[str],
) -> tuple[list[str], float | None]:
    """Split query terms into non-size terms and an optional size in billions."""
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


_ID_SIZE_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)(b|m)(?:[-_.]|$)", re.IGNORECASE)


def _extract_id_size_b(model_id: str) -> float | None:
    """Extract the size label from a model ID string, in billions."""
    for m in _ID_SIZE_RE.finditer(model_id):
        value = float(m.group(1))
        if value <= 0:
            continue
        unit = m.group(2).lower()
        return value if unit == "b" else value / 1000.0
    return None


def _size_compatible(model: ModelInfo, size_b: float) -> bool:
    """Check whether a model's parameter count is compatible with a query size."""
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

    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def _resolve_model_deps(model, variant) -> tuple[list[str], str]:
    """Determine pip dependencies and script type for a model."""
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


__all__ = [
    "_extract_id_size_b",
    "_generate_chat_script",
    "_load_models",
    "_parse_size_tokens",
    "_pick_gguf_variant",
    "_resolve_model_deps",
    "_search_model",
    "_size_compatible",
]
