# whichllm

[![PyPI version](https://img.shields.io/pypi/v/whichllm)](https://pypi.org/project/whichllm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml/badge.svg)](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml)

**Find the best local LLM that actually runs on your hardware.**

Auto-detects your GPU/CPU/RAM and ranks the top models from HuggingFace that fit your system.

[日本語版はこちら](docs/README.ja.md)

![demo](assets/demo.gif)

## Why whichllm?

**One command. Real answers.** No TUI to learn, no keybindings to memorize.

| | whichllm | Others (TUI-based) |
|---|---|---|
| **Getting results** | `whichllm` — done | Launch TUI → navigate → search → filter |
| **Model data** | Live from HuggingFace API | Static built-in database |
| **Benchmarks** | Real eval scores with confidence | Fixed quality scores |
| **Scriptable** | `whichllm --json \| jq` | Requires special flags |
| **Learning curve** | Zero | Vim keybindings required |

## Features

- **Auto-detect hardware** — NVIDIA, AMD, Apple Silicon, CPU-only
- **Smart ranking** — Scores models by VRAM fit, speed, and benchmark quality
- **One-command chat** — `whichllm run` downloads and starts a chat session instantly
- **Code snippets** — `whichllm snippet` prints ready-to-run Python for any model
- **Live data** — Fetches models directly from HuggingFace (cached for performance)
- **Benchmark-aware** — Integrates real eval scores with confidence-based dampening
- **Task profiles** — Filter by general, coding, vision, or math use cases
- **GPU simulation** — Test with any GPU: `whichllm --gpu "RTX 4090"`
- **Hardware planning** — Reverse lookup: `whichllm plan "llama 3 70b"`
- **JSON output** — Pipe-friendly: `whichllm --json`

## Run & Snippet

**Try any model with a single command.** No manual installs needed — whichllm creates an isolated environment via `uv`, installs dependencies, downloads the model, and starts an interactive chat.

![run demo](assets/demo-run.gif)

```bash
# Chat with a model (auto-picks the best GGUF variant)
whichllm run "qwen 2.5 1.5b gguf"

# Auto-pick the best model for your hardware and chat
whichllm run

# CPU-only mode
whichllm run "phi 3 mini gguf" --cpu-only
```

Works with **all model formats**:
- **GGUF** — via `llama-cpp-python` (lightweight, fast)
- **AWQ / GPTQ** — via `transformers` + `autoawq` / `auto-gptq`
- **FP16 / BF16** — via `transformers`

Get a **copy-paste Python snippet** instead:

```bash
whichllm snippet "qwen 7b"
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
)
print(output["choices"][0]["message"]["content"])
```

## Install

### pipx (recommended)

```bash
pipx install whichllm
```

### pip

```bash
pip install whichllm
```

### Development

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --dev
uv run whichllm
```

## Usage

```bash
# Auto-detect hardware and show best models
whichllm

# Simulate a GPU (e.g. planning a purchase)
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090"

# CPU-only mode
whichllm --cpu-only

# More results / filters
whichllm --top 20
whichllm --quant Q4_K_M
whichllm --min-speed 30
whichllm --evidence base   # allow id/base-model matches
whichllm --evidence strict # id-exact only (same as --direct)
whichllm --direct

# JSON output
whichllm --json

# Force refresh (ignore cache)
whichllm --refresh

# Show hardware info only
whichllm hardware

# Plan: what GPU do I need for a specific model?
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768

# Run: download and chat with a model instantly
whichllm run "qwen 2.5 1.5b gguf"
whichllm run                       # auto-pick best for your hardware

# Snippet: print ready-to-run Python code
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

## Integrations

### Ollama

Find the best model and run it directly:

```bash
# Pick the top model and run it with Ollama
whichllm --top 1 --json | jq -r '.models[0].model_id' | xargs ollama run

# Find the best coding model
whichllm --profile coding --top 1 --json | jq -r '.models[0].model_id' | xargs ollama run
```

### Shell alias

Add to your `.bashrc` / `.zshrc`:

```bash
alias bestllm='whichllm --top 1 --json | jq -r ".models[0].model_id"'
# Usage: ollama run $(bestllm)
```

## Scoring

Each model gets a score from 0 to 100.

| Factor | Points | Description |
|--------|--------|-------------|
| Model size | 0-40 | Larger models generally produce better output |
| Benchmark | 0-10 | Arena ELO / Open LLM Leaderboard scores |
| Speed | 0-20 | Higher tok/s = more practical to use |
| Source trust | -5 to +5 | Official repos get a bonus, repackagers get a penalty |
| Popularity | 0-3 | Downloads and likes as tiebreaker |

Score markers:
- **`~`** (yellow) — No direct benchmark yet. Score estimated from the model family
- **`?`** (yellow) — No benchmark data available

## How it works

### Data pipeline

1. Fetches ~900 popular models from **HuggingFace API** (text-generation, GGUF, multimodal)
2. Fetches benchmark scores from **Chatbot Arena ELO** and **Open LLM Leaderboard**, normalized to 0-100
3. All data cached for 24 hours at `~/.cache/whichllm/`

### Ranking engine

1. **Hardware detection** — GPU (NVIDIA/AMD/Apple Silicon), CPU, RAM, disk
2. **VRAM estimation** — model size + quantization + KV cache overhead
3. **Compatibility check** — Full GPU / Partial Offload / CPU-only classification
4. **Speed estimation** — tok/s based on GPU memory bandwidth
5. **Scoring** — combines size, benchmark, speed, source trust, and popularity
6. **Deduplication** — merges GGUF variants and version differences into model families

### Project structure

```
src/whichllm/
├── cli.py              # Typer CLI entry point
├── constants.py        # GPU bandwidth tables, quantization constants
├── hardware/           # Hardware detection (NVIDIA, AMD, Apple, CPU, RAM)
│   └── gpu_simulator.py  # GPU simulation for --gpu flag
├── models/
│   ├── fetcher.py      # HuggingFace API model fetcher
│   ├── benchmark.py    # Benchmark scores (Arena + Leaderboard)
│   ├── grouper.py      # Model family grouping and dedup
│   └── cache.py        # JSON cache
├── engine/
│   ├── vram.py         # VRAM requirement estimation
│   ├── compatibility.py # Hardware compatibility check
│   ├── performance.py  # Inference speed estimation
│   └── ranker.py       # Scoring and ranking
└── output/
    └── display.py      # Rich table output
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Requirements

- Python 3.11+
- NVIDIA GPU detection via `nvidia-ml-py` (included by default)
- AMD / Apple Silicon detected automatically

## License

MIT
