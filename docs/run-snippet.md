# Run and snippet

whichllm can do more than print recommendations:

- `whichllm run` starts an interactive chat with a selected model.
- `whichllm snippet` prints a Python script for manual use.

Both commands use the same model loading helpers in `cli_models.py`, and
without a model name both take the same top pick.

## `whichllm run`

```bash
whichllm run [MODEL_NAME]
```

If `MODEL_NAME` is provided, whichllm searches the fetched model list for an
exact ID, suffix match, or term match. When several repos match, it takes the
one whose size label is closest to a size in the query (`8b`), then the
closest name, then the original over a quantized upload, then the most
downloaded. Speculative-decoding draft heads and pretrain-only checkpoints
are picked only when the query names them or nothing else matches. Add a
format word (`gguf`, `awq`) to get an upload, or pass the repo ID to pick one
exactly.

If `MODEL_NAME` is omitted, whichllm ranks models for the current hardware and
uses the top result.

Examples:

```bash
whichllm run
whichllm run "qwen 2.5 1.5b gguf"
whichllm run "phi 3 mini gguf" --cpu-only
whichllm run "llama 3 8b gguf" --quant Q5_K_M
whichllm run "qwen 2.5 1.5b gguf" --lm-studio-path /mnt/models/lm-studio
```

A GGUF model already installed in LM Studio is opened from that library instead
of being downloaded again. The match needs the resolved repository, the exact
artifact path, and every part of a split GGUF beside the entry file. Libraries
are only read; `--lm-studio-path` adds a legacy or custom one and can be
repeated. The reuse and the fallback when the files change under it are covered
under [How `run` executes](#how-run-executes).

## How `run` executes

`run` needs llama.cpp's `llama-cli` (for GGUF models) or `uv` in `PATH`.

At runtime, whichllm loads the models, selects a model and quantization, and
then takes one of two paths.

With `llama-cli` in `PATH` and a GGUF model, it runs

```bash
llama-cli -hf <repo>:<quant> -c <context> <flags>
```

where the flags are the printed recipe's (`-ngl 99 --n-cpu-moe 12 -fa on`, for
example). llama-cli downloads the file into its own cache, keeps MoE experts in
RAM when the ranking did, and uses the GPU. With a local LM Studio match the
`-hf <repo>:<quant>` source becomes `-m <entry part>`, and the flags stay the
same. A named model gets no placement
flags: llama.cpp's `--fit`, on by default, sizes the GPU layers and experts to
the free memory. `--cpu-only` adds `-ngl 0`. On an RTX 5080, `whichllm run
unsloth/Qwen3.5-2B-GGUF` this way answered in 4 seconds once the file was
cached, at about 340 tok/s.

Otherwise it:

1. Generates a temporary Python chat script.
2. Runs that script with `uv run --no-project`.
3. Adds the required dependencies with repeated `--with` flags.
4. Deletes the temporary script after the chat exits.

With a local LM Studio match, the script opens the entry file and checks every
part again before loading: the isolated build runs first and can take minutes, so
a file that went away in the meantime prints a note and falls back to the
Hugging Face download instead of failing. `huggingface-hub` stays in the
dependency list as that fallback, and the reuse skips the download, not the
package.

This keeps the project environment clean. The temporary runtime dependencies
are not added to `pyproject.toml`.

PyPI ships `llama-cpp-python` only as source, so on this path the first GGUF
run compiles it. On a Core Ultra 9 the first reply took 22.7 minutes, and the
build has no GPU support. `run` prints a note about this before it starts.
Later runs reuse uv's cached build.

## Supported model paths

### GGUF

Without llama-cli, GGUF models use:

- `llama-cpp-python`
- `huggingface-hub`

The generated script downloads the selected GGUF file with `hf_hub_download`
and loads it through `llama_cpp.Llama`.

GPU behavior, which only matters for a llama-cpp-python build with GPU support:

- default: `n_gpu_layers=-1`, or the ranking's layer split for an auto pick
- `--cpu-only`: `n_gpu_layers=0`

### AWQ

AWQ repos are inferred from the model ID and use:

- `transformers`
- `torch`
- `accelerate`
- `autoawq`

### GPTQ

GPTQ repos are inferred from the model ID and use:

- `transformers`
- `torch`
- `accelerate`
- `auto-gptq`

### FP16, BF16, FP8, INT8, BNB 4-bit

Other non-GGUF repos use the Transformers path. The generated script uses
`device_map="auto"` unless `--cpu-only` is set.

## Quantization selection

For GGUF repos, whichllm chooses a variant by this preference order unless
`--quant` is provided:

```text
Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q3_K_M, Q3_K_L, Q8_0, ...
```

This order favors a practical balance of memory and quality. Very low-bit
variants are available when explicitly requested but are not preferred by
default.

If the requested quantization is not available, `run` warns and falls back to
the best available match.

## Chat behavior

GGUF scripts call:

```python
llm.create_chat_completion(messages=messages, stream=True)
```

Transformers scripts use:

```python
tokenizer.apply_chat_template(...)
model.generate(...)
```

The chat loop keeps the current conversation history in memory until the
process exits. Type `exit`, `quit`, or `q` to stop. In llama-cli, type `/exit`
or press Ctrl+C.

## `whichllm snippet`

```bash
whichllm snippet [MODEL_NAME]
```

`snippet` prints a short Python example instead of running it.

Examples:

```bash
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

If no model is provided, `snippet` ranks models for the current hardware and
prints code for the top pick, the same pick `run` makes.

## Manual execution

The snippet output includes a suggested `uv run --no-project` command with the
needed `--with` dependencies.

Example shape:

```bash
uv run --no-project --with llama-cpp-python --with huggingface-hub script.py
```

## Practical notes

- First run can take time because dependencies and model weights need to
  download.
- HuggingFace access rules still apply. Gated models may require local
  HuggingFace authentication.
- `run` is a convenience path, not a full model manager.
- `snippet` is better when you want to adapt loading code into your own project.
- Generated Transformers scripts use `trust_remote_code=True`, matching common
  HuggingFace local inference patterns. Review model repos before running code
  from untrusted sources.
