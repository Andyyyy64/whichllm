# CLI reference

This page describes the commands exposed by `whichllm`. It is based on the
Typer entrypoint in `src/whichllm/cli.py`.

## Main command

```bash
whichllm [OPTIONS]
```

Detects the current machine, loads model and benchmark data, ranks compatible
models, and prints a table.

Common options:

| Option | Meaning |
| --- | --- |
| `--top`, `-n` | Number of ranked models to show. Default: `10` |
| `--context-length`, `-c` | Context length used for KV cache estimation. Accepts integers or `k` shorthand such as `64k`. Default: `4096` |
| `--quant`, `-q` | Keep only a quantization type such as `Q4_K_M` |
| `--min-speed` | Keep only models above an exact tok/s estimate |
| `--speed` | Named speed floor: `any`, `usable` (`10 tok/s`), or `fast` (`30 tok/s`) |
| `--fit` | Runtime fit filter: `any`, `gpu`, or `full-gpu` |
| `--gpu-only` | Alias for `--fit full-gpu`; excludes partial offload and CPU-only candidates |
| `--profile` | Ranking profile: `general`, `coding`, `vision`, `math`, `any` |
| `--evidence` | Benchmark evidence filter: `strict`, `base`, `any` |
| `--direct` | Alias for `--evidence strict` |
| `--status` | Compatibility option. Runtime columns are now shown by default |
| `--details` | Show download metadata instead of runtime columns |
| `--min-params` | Minimum model knowledge capacity in billions of parameters |
| `--json` | Print machine-readable JSON |
| `--markdown`, `-m` | Print a pasteable GitHub-Flavored Markdown table |
| `--refresh` | Ignore caches and fetch models/benchmarks again |
| `--cpu-only` | Ignore GPUs and rank for CPU-only use |
| `--gpu` | Simulate GPU(s) by name. Accepts repeated flags, comma-separated values, and count shorthand |
| `--vram` | Override simulated GPU VRAM in GB. Requires `--gpu` |
| `--vram-headroom` | Reserve per-GPU memory for runtime overhead. Default: `auto`. Accepts `none`, byte values like `1.5GB`, or percentages like `10%` |
| `--ram-budget` | Cap RAM available for partial offload. Accepts `available`, byte values like `8GB`, or percentages like `50%` |
| `--version` | Print the installed package version |

Environment variables:

| Variable | Meaning |
| --- | --- |
| `HF_ENDPOINT` | Hugging Face endpoint root used for whichllm's own model metadata API calls. Example: `https://huggingface.co` or a compatible mirror root |

`--fit any` is the default. It can include full-GPU, partial-offload, and
CPU-only candidates when they are runnable. `--fit gpu`, `--fit full-gpu`, and
`--gpu-only` keep only rows whose `fit_type` is `full_gpu`.

The default table shows memory required, estimated generation speed, fit type,
and published date. Use `--details` when you want download counts instead.
Speed colors are absolute usability hints: red is under `4 tok/s`, yellow is
`4-10 tok/s`, green is `10-30 tok/s`, and bright green is `30+ tok/s`. The `~`
and `?` markers still refer to estimate confidence, not speed quality.

`--vram-headroom auto` subtracts a small budget from each GPU before fit
checks, so near-edge recommendations are less likely to overflow in tools such
as LM Studio. Use `--vram-headroom none` to restore the raw detected VRAM.
`--ram-budget available` caps offload planning to current available RAM.

Examples:

```bash
whichllm
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5060 Ti" --vram 16
whichllm --gpu "2x RTX 4090"
whichllm --gpu "RTX 4090" --gpu "RTX 3090"
whichllm --gpu "RTX 4090, RTX 3090"
whichllm --profile coding --top 5
whichllm --context-length 64k
whichllm --gpu-only
whichllm --fit gpu
whichllm --speed usable
whichllm --speed fast
whichllm --min-speed 4
whichllm --markdown
whichllm --vram-headroom 1.5GB
whichllm --ram-budget available
whichllm --details
whichllm --evidence strict
whichllm --json | jq '.models[0]'
```

`--markdown` is mutually exclusive with `--json`. It prints a plain Markdown
table without the Rich hardware panel, colors, or box-drawing characters.

Ranking JSON model rows include:

| Field | Meaning |
| --- | --- |
| `fit_type` | Runtime fit classification: `full_gpu`, `partial_offload`, or `cpu_only` |
| `vram_required_bytes` | Estimated runtime memory requirement for the candidate |
| `vram_available_bytes` | GPU memory budget used for the fit check |
| `uses_multi_gpu` | Whether the fit check used more than one GPU |
| `multi_gpu_effective_vram_bytes` | Conservative effective VRAM budget for multi-GPU fits, when applicable |
| `estimated_tok_per_sec` | Point estimate used by ranking |
| `speed_confidence` | `high`, `medium`, or `low` |
| `speed_range_tok_per_sec` | Estimated lower/upper tok/s range, when available |
| `speed_notes` | Short reasons for the confidence level |
| `benchmark_status` | Display marker category for benchmark evidence |
| `benchmark_source` | How benchmark evidence was matched: `direct`, `variant`, `base_model`, `line_interp`, `self_reported`, or `none` |
| `benchmark_confidence` | Confidence in the benchmark match, `0.0`–`1.0` |

The top-level `hardware` object also includes `usable_vram_bytes` per GPU,
`ram_budget_bytes`, and `budget_notes` when memory budgets are active.

## `hardware`

```bash
whichllm hardware [OPTIONS]
```

Prints detected hardware without ranking models. The same simulation flags are
available here:

```bash
whichllm hardware
whichllm hardware --cpu-only
whichllm hardware --gpu "Apple M3 Max"
whichllm hardware --gpu "RTX 3060" --vram 12
whichllm hardware --gpu "4x RTX 4090"
```

## `plan`

```bash
whichllm plan MODEL_NAME [OPTIONS]
```

Searches for a model by HuggingFace repo ID or fuzzy terms, then estimates the
VRAM required for several quantization levels and common GPUs.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length for the memory estimate. Accepts integers or `k` shorthand such as `128k`. Default: `4096` |
| `--quant`, `-q` | Target quantization. Default: `Q4_K_M` |
| `--json` | Print the plan as JSON |
| `--refresh` | Ignore model cache and fetch again |

Examples:

```bash
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768
whichllm plan "mistral 7b" --context-length 32k
```

## `upgrade`

```bash
whichllm upgrade TARGET_GPUS... [OPTIONS]
```

Compares the current machine against one or more simulated GPUs. The CPU, RAM,
disk, and OS come from the current machine; only the GPU changes.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length used for ranking. Accepts integers or `k` shorthand such as `64k`. Default: `8192` |
| `--top`, `-n` | Best-N models to compare per GPU. Default: `3` |
| `--profile` | Ranking profile. Default: `general` |
| `--cpu-only` | Use CPU-only as the current baseline |
| `--json` | Print comparison JSON |
| `--refresh` | Ignore caches and fetch again |

Examples:

```bash
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
whichllm upgrade "Apple M4 Max" --top 5
whichllm upgrade "RX 7900 XTX" --profile coding
whichllm upgrade "RTX 4090" --context-length 128k
```

## `run`

```bash
whichllm run [MODEL_NAME] [OPTIONS]
```

Creates a temporary Python script, launches it through `uv run --no-project`,
installs the needed inference packages into that isolated run, and starts an
interactive chat.

If `MODEL_NAME` is omitted, whichllm ranks models for the current hardware and
uses the top result.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length for the generated chat script. Accepts integers or `k` shorthand such as `64k` |
| `--quant`, `-q` | Preferred GGUF quantization |
| `--refresh` | Ignore model cache and fetch again |
| `--cpu-only` | Force CPU-only execution in the generated script |

Examples:

```bash
whichllm run
whichllm run "qwen 2.5 1.5b gguf"
whichllm run "phi 3 mini gguf" --cpu-only
whichllm run "mistral 7b gguf" --context-length 64k
```

`run` requires `uv` in `PATH`.

## `snippet`

```bash
whichllm snippet [MODEL_NAME] [OPTIONS]
```

Prints a ready-to-run Python snippet for the selected model. GGUF models use
`llama-cpp-python`; non-GGUF models use `transformers`.

Options:

| Option | Meaning |
| --- | --- |
| `--quant`, `-q` | Preferred GGUF quantization |
| `--refresh` | Ignore model cache and fetch again |

Examples:

```bash
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

## Evidence filters

`--evidence` controls which benchmark matches are allowed into the ranking.

| Mode | Allows |
| --- | --- |
| `strict` | Exact independent benchmark matches only |
| `base` | Exact, variant, and `cardData.base_model` matches |
| `any` | All evidence levels, including line interpolation and self-reported values |

`--direct` is kept as a shorter alias for `--evidence strict`.

## Profiles

The ranker detects specialization from repository names.

| Profile | Behavior |
| --- | --- |
| `general` | Excludes coding, vision, and math-specialized names |
| `coding` | Keeps coding-specialized names |
| `vision` | Keeps vision or multimodal names and includes VLM candidates |
| `math` | Keeps math-specialized names |
| `any` | Keeps all recognized model types and includes VLM candidates |
