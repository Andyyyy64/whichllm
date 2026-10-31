# whichllm

[![PyPI version](https://img.shields.io/pypi/v/whichllm)](https://pypi.org/project/whichllm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml/badge.svg)](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/Andyyyy64)

<p align="center">
  <a href="https://trendshift.io/repositories/30336" target="_blank"><img src="https://trendshift.io/api/badge/repositories/30336" alt="Andyyyy64%2Fwhichllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

**The smartest model your PC can actually run, including ones you didn't know it could.**

whichllm reads your GPU, RAM and memory bandwidth, works out which open
models fit and how fast each one will generate, and ranks them by benchmark
results. Then it prints the command that runs the top pick the way it was
estimated.

[日本語版はこちら](docs/README.ja.md)

## Quick start

```bash
uvx whichllm@latest
```

Shopping for a GPU? Simulate one:

```bash
uvx whichllm@latest --gpu "RTX 4090"
```

To keep it around:

```bash
uv tool install whichllm
brew install andyyyy64/whichllm/whichllm
pip install whichllm
```

## Example

A desktop with an 8 GB RTX 4060 and 32 GB of DDR4 (output trimmed):

```text
$ whichllm

 #  Model                         Quant   Fit       Speed     Score
 1  Qwen/Qwen3.6-35B-A3B          Q6_K    GPU+RAM   16 tok/s   68.8
 2  unsloth/gemma-4-12b-it-GGUF   IQ4_XS  Full GPU  37 tok/s   68.1
 3  google/gemma-4-26B-A4B-it     Q4_K_M  GPU+RAM   11 tok/s   64.5
 4  Qwen/Qwen3.5-9B               Q5_K_M  Full GPU  36 tok/s   64.2
 5  zai-org/GLM-4.7-Flash         Q5_K_M  GPU+RAM   15 tok/s   62.3

  #1 Qwen3.6-35B-A3B also stands in for older generations of its line:
  Qwen3.5-35B-A3B, Qwen3-30B-A3B, Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF and 2 more

  Run #1 with llama.cpp:
    llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K -c 4096 -ngl 99 --n-cpu-moe 35 -fa on
```

The top pick is a 35B model on an 8 GB card. It is a mixture-of-experts
model, so each token only uses about 3B of its parameters. whichllm keeps
attention, the shared weights and the KV cache in VRAM and leaves the experts
of 35 of its 40 layers in system RAM, which is what llama.cpp's `--n-cpu-moe`
does. The speed estimate counts only what each token reads from each place.
A fits-in-VRAM check would have skipped the model, and treating it as a
plain partial offload would have rated it far too slow.

On a 24 GB RTX 4090 the same logic picks Qwen3.8-27B, although Qwen3-32B
also fits there at Q4_K_M. The 27B is the newer generation of the same model
line at about the same size and scores higher on current benchmarks, so the
32B is not listed separately; the note under the table names it.

Speeds are estimates from memory bandwidth, not measurements of your
machine. The table marks how confident each estimate is.

## How it ranks

Quality comes from Artificial Analysis and LiveBench, put on one scale and
averaged. `--profile coding`, `math` and `vision` rank on benchmarks for that
task instead, such as SciCode and LiveCodeBench for coding. The Open
LLM Leaderboard and Chatbot Arena are frozen, so they only score models
nothing current covers, and older generations are demoted. Each match
records how it was found (the exact repo, its base model, a same-family
interpolation, or the uploader's own claim) and weaker matches count for
less. whichllm never recommends pretrain-only base checkpoints.

To check the fit, memory is counted as weights plus a KV cache computed
layer by layer from the model's `config.json` (grouped-query and latent
attention, sliding windows, recurrent layers), plus runtime buffers. When a
model does not fit in VRAM, a mixture-of-experts model first moves experts
to RAM; any other model spills whole layers.

Speed is bound by memory bandwidth: the GPU reads its share of the weights
at VRAM speed and the CPU reads the rest at RAM speed. Both sides are
calibrated against llama.cpp runs. Every placement is held to 8 tok/s:
below that, a model's quality is scaled down, so a stronger model that would
answer at 3 tok/s does not outrank one that answers at 25.

The details are in [Scoring](docs/scoring.md) and
[How it works](docs/how-it-works.md).

## How we test recommendations

A wrong recommendation is a bug, so the rankings have regression tests:
[`tests/golden/`](tests/golden/).

- 29 hardware archetypes, from a 4 GB GTX 1650 to a 128 GB M4 Max and a DGX
  Spark, with AMD, Intel and mixed GPU sets, CPU-only boxes and
  32K/128K-context setups, ranked against a frozen copy of the index: 1,441
  models, most GGUF repos with their real file sizes.
- 30 constraints describe what a good answer looks like instead of pinning an
  exact order: no pretrain-only checkpoints, draft heads or diffusion models,
  no outdated generation in the top 5, the top 3 stay within 1 tok/s of the
  8 tok/s bar whenever the machine has that many options that fast, full-GPU
  rows keep their VRAM headroom and stay inside Metal's working set, `run` can
  launch the top rows, the newest dense 27B makes the top 3 on 24 GB+, a big
  MoE with experts in RAM makes the top 5 on a 16 GB card with 64 GB of RAM,
  and coding picks are not limited to models named "coder".
- Speed and memory estimates stay within set bounds of real runs: an RTX 5080
  and its CPU, plus 379 published llama.cpp results on 194 GPUs and 255 LocalScore
  results on 138 CPUs in [`tests/data/`](tests/data/). The estimator cannot drift
  without a test failing.
- The top picks for every archetype are committed in
  [`top_picks.md`](tests/golden/top_picks.md), so any ranking change shows up
  as a diff in review.
- A constraint that is known to fail is recorded with its reason as a strict
  expected failure. Fixing the bug turns it into a test failure until the
  record is removed.

If whichllm picks something you would not run on your hardware, open an
issue with the output. The 6 GB and RTX 3060 archetypes started as reports
like that ([#76](https://github.com/Andyyyy64/whichllm/issues/76),
[#104](https://github.com/Andyyyy64/whichllm/issues/104)).

## Commands

```bash
whichllm                          # best models for this machine
whichllm --gpu "RTX 5060 Ti"      # simulate a GPU (add --vram 16 for a variant)
whichllm --gpu "2x RTX 3090"      # or several
whichllm --profile coding         # general, coding, vision, math, any
whichllm --context-length 64k     # long context changes what fits
whichllm --gpu-only               # only models that fit entirely in VRAM
whichllm --speed usable           # hide anything under 8 tok/s
whichllm --markdown               # pasteable table for issues and chats
whichllm --json                   # rows include placement and the run recipe

whichllm hardware                 # what was detected
whichllm plan "qwen3 32b"         # what GPU a model needs
whichllm upgrade "RTX 4090" "RTX 5090"   # compare against candidate GPUs
whichllm run                      # download the top pick and chat with it
whichllm snippet                  # print Python that loads the top pick
```

Every option is in the [CLI reference](docs/cli.md).

If LM Studio complains that a pick is slightly too large, ask for a more
conservative fit:

```bash
whichllm --gpu-only --speed usable --vram-headroom 1.5GB
```

If an integrated GPU or unified memory is detected with the wrong limits,
override them with `--vram` and `--ram-bandwidth` (see
[Hardware](docs/hardware.md)).

## Run and snippet

`whichllm run` downloads the model and opens a chat. Without a model name it
takes the top pick.

![run demo](assets/demo-run.gif)

```bash
whichllm run
whichllm run "qwen 2.5 1.5b gguf"
whichllm run "phi 3 mini gguf" --cpu-only
```

With llama.cpp installed (`brew install llama.cpp`, `winget install
ggml.llamacpp`), a GGUF model runs in its `llama-cli` with the flags of the
printed recipe, so the chat uses the GPU and keeps MoE experts in RAM the way
the ranking estimated. A named model is left to llama.cpp's automatic fit.

A GGUF model already installed in LM Studio is opened from that library instead
of being downloaded again; `whichllm recommend` marks those rows `Installed`.
Pass `--lm-studio-path` (repeatable) for a legacy or custom library:

```bash
whichllm run "qwen 2.5 1.5b gguf" --lm-studio-path /mnt/models/lm-studio
```

Without llama-cli, `run` creates a throwaway environment with `uv` and uses
`llama-cpp-python` for GGUF models and `transformers` for the rest. PyPI only
ships llama-cpp-python as source, so the first run compiles it, which took
22 minutes on a Core Ultra 9, and that build runs on the CPU only. See
[Run and snippet](docs/run-snippet.md).

## Where the data comes from

whichllm reads a prebuilt index: the Hugging Face model list, each family's
`config.json` and the merged benchmark scores, rebuilt daily on GitHub
Actions and published only if it passes quality checks. A cold start takes
about a second and makes no Hugging Face API calls. `--live` fetches
everything directly instead, and `HF_TOKEN` is used when set.

## Documentation

- [CLI reference](docs/cli.md)
- [How it works](docs/how-it-works.md)
- [Scoring](docs/scoring.md)
- [Hardware detection and simulation](docs/hardware.md)
- [Run and snippet](docs/run-snippet.md)
- [Troubleshooting](docs/troubleshooting.md)

## Development

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --locked --dev
uv run whichllm
uv run pytest
```

Ranking changes need a look at the golden snapshot:
`uv run python scripts/update_golden.py report`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Support

If whichllm saved you a bad download or a bad hardware purchase, sponsoring
helps keep it maintained: hardware reports, benchmark updates, packaging and
support for more machines. It stays open source either way.

A star helps other people find it. I'd also like to hear what it picked for
your machine, so feel free to drop it in
[Issues](https://github.com/Andyyyy64/whichllm/issues).

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=Andyyyy64/whichllm&type=Date)](https://star-history.dera.page/#Andyyyy64/whichllm&Date)

## License

MIT
