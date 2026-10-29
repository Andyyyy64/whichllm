# Run and snippet

whichllm can do more than print recommendations:

- `whichllm run` starts an interactive chat with a selected model.
- `whichllm snippet` prints a Python script for manual use.

Both commands use the same model loading helpers in `cli.py`.

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
```

## How `run` executes

`run` requires `uv` in `PATH`.

At runtime, whichllm:

1. Loads models from cache or HuggingFace.
2. Selects a model and quantization.
3. Generates a temporary Python chat script.
4. Runs that script with `uv run --no-project`.
5. Adds the required dependencies with repeated `--with` flags.
6. Deletes the temporary script after the chat exits.

This keeps the project environment clean. The temporary runtime dependencies
are not added to `pyproject.toml`.

## Supported model paths

### GGUF

GGUF models use:

- `llama-cpp-python`
- `huggingface-hub`

The generated script downloads the selected GGUF file with `hf_hub_download`
and loads it through `llama_cpp.Llama`.

GPU behavior:

- default: `n_gpu_layers=-1`
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
process exits. Type `exit`, `quit`, or `q` to stop.

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

If no model is provided, `snippet` picks the most-downloaded GGUF model from
the fetched model list. This is different from `run`, which auto-ranks for the
current hardware when no model name is provided.

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
