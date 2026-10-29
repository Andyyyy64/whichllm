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
| `--speed` | Named speed floor: `any`, `usable` (`8 tok/s`, the speed the ranking holds every placement to), or `fast` (`30 tok/s`) |
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
| `--refresh` | Revalidate the whichllm index now; with `--live`, ignore caches and fetch models/benchmarks again |
| `--live` | Fetch from HuggingFace and the benchmark sources instead of the daily whichllm index |
| `--cpu-only` | Ignore GPUs and rank for CPU-only use |
| `--gpu` | Simulate GPU(s) by name. Accepts repeated flags, comma-separated values, and count shorthand |
| `--vram` | Override simulated GPU VRAM or detected GPU usable VRAM in GB |
| `--bandwidth` | Override the selected GPU's memory bandwidth in GB/s |
| `--ram-bandwidth` | Override system RAM bandwidth in GB/s. It bounds CPU-only runs, MoE experts in RAM and offloaded layers; for an iGPU or unified memory it also sets the GPU's bandwidth |
| `--gpu-index` | Detected GPU index to override when multiple GPUs are present |
| `--vram-headroom` | Reserve per-GPU memory for runtime overhead. Default: `auto`. Accepts `none`, byte values like `1.5GB`, or percentages like `10%` |
| `--ram-budget` | Cap RAM available for partial offload. Accepts `available`, byte values like `8GB`, or percentages like `50%` |
| `--lm-studio-path` | Scan an additional LM Studio model library. Repeat the option for legacy or multiple custom locations |
| `--version` | Print the installed package version |

Environment variables:

| Variable | Meaning |
| --- | --- |
| `WHICHLLM_INDEX_URL` | Location of the whichllm index (default: GitHub Pages). Set to `off` to always fetch live |
| `HF_TOKEN` | Hugging Face token for higher rate limits and gated configs on the live path. Sent to huggingface.co over https; a mirror in `HF_ENDPOINT` gets it only with `WHICHLLM_HF_TOKEN_TO_ENDPOINT=1`, never over http |
| `WHICHLLM_HF_TOKEN_TO_ENDPOINT` | `1` sends `HF_TOKEN` to the `HF_ENDPOINT` mirror (https only) |
| `HF_ENDPOINT` | Hugging Face endpoint root used for whichllm's own model metadata API calls. Example: `https://huggingface.co` or a compatible mirror root |

`--fit any` is the default. It can include full-GPU, partial-offload, and
CPU-only candidates when they are runnable. `--fit gpu`, `--fit full-gpu`, and
`--gpu-only` keep only rows whose `fit_type` is `full_gpu`.

The default table shows memory required, estimated generation speed, fit type,
and published date. Use `--details` when you want download counts instead.
Speed colors are absolute usability hints: red is under `4 tok/s`, yellow is
`4-8 tok/s`, green is `8-30 tok/s`, and bright green is `30+ tok/s`. The `~`
and `?` markers still refer to estimate confidence, not speed quality.

`--vram-headroom auto` subtracts a small budget from each GPU before fit
checks, so near-edge recommendations are less likely to overflow in tools such
as LM Studio. Use `--vram-headroom none` to restore the raw detected VRAM.
`--ram-budget available` caps offload planning to current available RAM.
For detected iGPU or unified-memory systems, use `--vram` and
`--ram-bandwidth` to override the automatically detected usable memory and
bandwidth. If multiple GPUs are detected, add `--gpu-index`
with the GPU number from `whichllm hardware`.

whichllm scans `~/.lmstudio/models` for GGUF files. A recommendation is marked
`Installed` only when its resolved Hugging Face repository and exact artifact
path match a readable local file. Every part of a split GGUF must be present.
Libraries are only read; whichllm does not move, modify, or delete LM Studio
files. Use `--lm-studio-path` for legacy or custom model libraries; an explicit
path that cannot be read is reported as an error.

Examples:

```bash
whichllm
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5060 Ti" --vram 16
whichllm --vram 8 --ram-bandwidth 68
whichllm --gpu-index 1 --vram 8 --bandwidth 68
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
whichllm --lm-studio-path /mnt/models/lm-studio
whichllm --details
whichllm --evidence strict
whichllm --json | jq '.models[0]'
```

`--json` output has a `data` object that says where the scores came from:
`{"source": "index", "built_at": ..., "benchmark_sources": [...]}`, or
`{"source": "live"}` with `--live`.

`--markdown` is mutually exclusive with `--json`. It prints a plain Markdown
table without the Rich hardware panel, colors, or box-drawing characters.

Below the table, whichllm prints how to run the top pick the way it was
estimated, when a GGUF build of it is known:

```text
  Run #1 with llama.cpp:
    llama-server -hf ggml-org/gpt-oss-120b-GGUF:MXFP4 -c 4096 -ngl 99 --n-cpu-moe 29 -fa on
  LM Studio: GPU Offload: max, Number of layers for which to force MoE weights onto CPU: 29
```

The flags follow the placement: `-ngl 99` for a full-GPU fit,
`--n-cpu-moe N` (or `--cpu-moe`) when MoE experts stay in RAM, `-ngl N` for
layer offload and `-ngl 0` for CPU-only. `--markdown` prints the command as a
code block, and `--json` adds a `recipe` object to every row.

The LM Studio line names the settings in its model loader (checked against
LM Studio 0.4.26). When every expert stays in RAM, the MoE layer count is set
to the model's layer count. Ollama picks its own GPU/CPU split, so for a split
model the recipe says its speed can differ from the estimate. On an RTX 5080,
Ollama 0.40 ran Qwen3.6-35B-A3B at about 100 tok/s, against 88 for
llama-server with `--n-cpu-moe 12`.

Ranking JSON model rows include:

| Field | Meaning |
| --- | --- |
| `fit_type` | Runtime fit classification: `full_gpu`, `partial_offload`, or `cpu_only` |
| `placement` | Where the weights live: `kind` (`gpu`, `expert_offload`, `layer_offload`, `cpu`), `gpu_weight_bytes`, `cpu_weight_bytes`, `n_cpu_moe` (llama.cpp `--n-cpu-moe`; `null` with expert offload means every expert, `--cpu-moe`), `n_gpu_layers`, `num_layers` |
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
| `local_match` | Whether a matching GGUF was found in an LM Studio library |
| `local_path` | Absolute path to the matching local GGUF, or `null` |
| `recipe` | How to run this row: `gguf_repo`, `quant`, `llama_cpp` (a `llama-server` command), `lm_studio` (load settings), `ollama` (command) and `notes`; `null` when no GGUF build is known |

The top-level `hardware` object also includes `usable_vram_bytes` per GPU,
`ram_budget_bytes`, and `budget_notes` when memory budgets are active, plus
`ram_bandwidth_gbps` and `ram_bandwidth_source` (`detected`, `unified`,
`estimated`, `override` or `default`).

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
whichllm hardware --vram 8 --bandwidth 68
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

When `run` picks the model itself, the generated script puts as many layers on
the GPU as the ranking's fit allows (`n_gpu_layers`). llama-cpp-python cannot
keep only MoE experts in RAM, so for an expert-offload pick `run` uses the
equivalent layer split, which is slower than the estimate, and prints the
`llama-server` command that matches it.

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

The ranker detects specialization from repository names, and image support
from the Hub's `image-text-to-text` tag. A profile decides which models are
eligible, and `coding`, `math` and `vision` also rank on benchmarks for that
task: Artificial Analysis's SciCode, Terminal-Bench, LiveCodeBench and
FrontierCode for coding, AIME 2025 for math, and MMMU-Pro plus a curated
vision index for vision. Those results are put on the same 0-100 scale as the
general score, so a model without them keeps its general score. JSON rows
say which one was used in `benchmark_basis`.

| Profile | Behavior |
| --- | --- |
| `general` | Excludes coding, vision, and math-specialized names |
| `coding` | General models plus coding-specialized names (excludes vision and math names), ranked on coding benchmarks where they exist. A code-tuned model without coding results gets a 2-point edge over an otherwise equal general one |
| `vision` | Models that accept images: vision or multimodal names, and natively multimodal models tagged `image-text-to-text` (Gemma 3/4, Qwen3.5/3.6) |
| `math` | General reasoning models plus math-specialized names (excludes coding and vision names) |
| `any` | Keeps all recognized model types and includes VLM candidates |

Pretrain-only (base) checkpoints are never recommended in any profile: names
ending in `-Base` or `-pt`, checkpoints whose publisher ships an instruct
version derived from them (`Qwen2.5-7B` for `Qwen2.5-7B-Instruct`), and
quantized uploads of those. Neither are speculative-decoding draft heads
(DFlash, `-draft` uploads), which only run next to their target model, or
diffusion language models such as DiffusionGemma, which llama.cpp, LM Studio
and Ollama do not run as chat models.

Non-GGUF checkpoints are only considered on Linux with an NVIDIA GPU. NVFP4
and FP8 checkpoints also need compute capability 8.0 or newer (Ampere), since
vLLM serves them through its Marlin kernels there. An NVFP4 pick on a GPU
older than Blackwell carries a note that it runs through weight-only FP4
kernels rather than native FP4.
