"""Implementation bodies for Typer CLI commands."""

from __future__ import annotations


import typer

from whichllm.cli_models import (
    _load_models,
    _pick_gguf_variant,
    _resolve_model_deps,
    _search_model,
)
from whichllm.cli_shared import console


def snippet_command(
    *,
    model_name: str | None,
    quant: str | None,
    refresh: bool,
) -> None:
    """Print a ready-to-run Python script for a model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    if model_name:
        model = _search_model(models, model_name)
    else:
        gguf_models = [m for m in models if m.gguf_variants]
        if not gguf_models:
            console.print("[red]No GGUF models found.[/]")
            raise typer.Exit(code=1)
        gguf_models.sort(key=lambda m: m.downloads, reverse=True)
        model = gguf_models[0]

    variant = _pick_gguf_variant(model, quant)
    deps, _ = _resolve_model_deps(model, variant)

    if variant:
        code = f"""\
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id={model.id!r},
    filename={variant.filename!r},
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = all layers on GPU, 0 = CPU only
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(output["choices"][0]["message"]["content"])
"""
    else:
        code = f"""\
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = {model.id!r}
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
)

inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichllm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))
