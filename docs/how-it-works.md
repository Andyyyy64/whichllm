# How it works

whichllm has one main job: start with the user's hardware, collect candidate
models, estimate what can run, and rank the results.

The implementation is intentionally split into small packages:

```text
src/whichllm/
├── cli.py
├── constants.py
├── data/
├── hardware/
├── models/
├── engine/
└── output/
```

## Request flow

The default `whichllm` command follows this path:

1. Validate CLI flags.
2. Detect hardware.
3. Load the whichllm index: models, shapes and benchmark scores in one file.
   With `--live`, or when no index is reachable or cached, load the model and
   benchmark caches or fetch from HuggingFace and the benchmark sources.
4. Group related model repos into families.
5. Flatten families back into rankable candidates.
6. Rank every candidate variant. On the live path, read `config.json` for the
   top families and rank again.
7. On the live path, backfill missing published dates for top results.
8. Print a Rich table or JSON.

The `plan`, `upgrade`, `run`, `snippet`, and `hardware` subcommands reuse parts
of the same pipeline.

## The whichllm index

A scheduled GitHub Actions job builds the index once a day and publishes it at
`https://andyyyy64.github.io/whichllm/index/v1/index.json.gz` (about 300 KB).
The build runs the same fetchers as a live run, plus `config.json` for every
family and published dates for every repo. It refuses to publish when a
quality gate fails: failed catalog requests, too few models or benchmark ids,
missing curated recent models, low shape coverage, or a large drop from the
previous index. A failed build leaves the last good index online.

The client revalidates the index with its ETag every 6 hours. Offline, it
uses the cached copy of any age, and says how old it is when that is more than
three days. `--refresh` revalidates immediately. `WHICHLLM_INDEX_URL` points
at a mirror, and `WHICHLLM_INDEX_URL=off` always uses the live path.

## Hardware detection

`hardware/detector.py` orchestrates detection. Each detector is fail-safe and
returns an empty result on failure.

| Module | Role |
| --- | --- |
| `hardware/nvidia.py` | Uses `nvidia-ml-py`; falls back to `nvidia-smi`, including optional memory-clock data |
| `hardware/amd.py` | Uses `rocm-smi`; falls back to `lspci` and `/sys/class/drm` |
| `hardware/intel.py` | Detects Linux Intel iGPUs through `lspci` or sysfs |
| `hardware/windows.py` | Detects Windows AMD and Intel fallback GPUs through WMI and registry memory fields |
| `hardware/apple.py` | Uses `system_profiler` on macOS |
| `hardware/cpu.py` | Reads CPU name, physical cores, AVX2, and AVX-512 |
| `hardware/memory.py` | Reads RAM and disk free space |
| `hardware/gpu_simulator.py` | Builds synthetic GPUs for `--gpu` |

The result is a `HardwareInfo` dataclass. GPUs are represented by `GPUInfo`.

## Model fetching

`models/fetcher.py` reads the HuggingFace API and turns responses into
`ModelInfo` objects.

The fetcher combines several queries:

1. Popular `text-generation` models sorted by downloads.
2. Popular GGUF repos.
3. Recently modified GGUF repos.
4. Trending text-generation repos, with and without the GGUF filter.
5. A curated list of frontier and hard-to-find model IDs.
6. Vision candidates from `image-text-to-text` when the active profile needs
   them.

For each repository, the parser extracts:

- repo ID and display name
- parameter count
- active parameter count for known or config-detected MoE models
- architecture and context length
- license, downloads, likes, published date
- GGUF variants and file sizes
- `cardData.base_model`
- conservative HuggingFace `evalResults` values

Parameter counts can come from safetensors metadata, GGUF metadata, config
estimation, name hints, or a curated fallback table for important models with
missing metadata.

## Model shapes and KV cache

The Hub API's model listing only includes `architectures` and `model_type`
from each config, not layer counts or attention heads. After a first ranking
pass, whichllm reads `config.json` for the families near the top (up to 25 per
run), then ranks again. A quantized upload uses its base model's config, and a
gated official repository borrows the shape of a same-size upload in its
family. `plan` reads the config of the model it shows.

With a shape, the KV cache is computed per layer, as llama.cpp allocates it by
default:

- FP16 K/V for attention layers: layers × KV heads × head dim × 2 × context
- sliding-window layers sized to their window, for architectures that honour
  it (Gemma 2/3/4, gpt-oss, Cohere 2, and configs that declare sliding layers)
- the compressed latent for multi-head latent attention (DeepSeek, GLM)
- a fixed per-sequence state for recurrent layers (Gated DeltaNet in
  Qwen3-Next and Qwen3.5/3.6, LFM2 convolution, Mamba2 in Granite 4.0)

Without a shape (no readable config, or a layout whichllm does not
recognise) the estimate falls back to a parameter-count heuristic. That
heuristic undercounts GQA models at long context, so use `whichllm plan` for
the exact figure.

## Caches

These caches normally live under `~/.cache/whichllm/`. If `XDG_CACHE_HOME` is
set to an absolute path, whichllm uses `$XDG_CACHE_HOME/whichllm/` instead.

| File | TTL | Contents |
| --- | --- | --- |
| `index.json.gz`, `index-state.json` | Revalidated every 6 hours | The whichllm index and its ETag |
| `models.json` | 6 hours (30 minutes if the fetch was incomplete) | Serialized `ModelInfo` data |
| `benchmark.json` | 24 hours | Combined benchmark score map |
| `shapes.json` | 7 days (1 day for missing configs) | Model shapes parsed from `config.json` |

`--refresh` bypasses the relevant cache and writes a new one after fetching.

## Benchmark sources

`models/benchmark.py` builds one score map from multiple sources. Scores are
normalized to a 0-100 scale before ranking.

whichllm separates sources into two tiers:

| Tier | Sources | Treatment |
| --- | --- | --- |
| Current | LiveBench, Artificial Analysis, Aider, Vision | Overrides frozen scores for the same model |
| Frozen | Open LLM Leaderboard v2, Chatbot Arena ELO | Kept for older coverage, capped and recency-demoted |

Current sources can use live scrapes when reachable and curated snapshots when
the upstream page shape changes. The snapshot month is printed below rankings.

Within Artificial Analysis, mapped live scores replace stored AA scores even
when lower or zero. The stored snapshot supplies models absent from the live
response. This does not change how scores from different benchmark sources
are combined. Use `--refresh` to bypass the 24-hour benchmark cache.

Frozen-only scores are demoted by model lineage. This prevents an older model
with a stale leaderboard score from outranking a newer generation simply because
the newer model was never added to that frozen leaderboard.

## Benchmark evidence

A model can receive benchmark evidence through several paths:

| Evidence | Meaning |
| --- | --- |
| `direct` | Exact independent benchmark match |
| `variant` | Suffix-stripped or `-Instruct` variant match |
| `base_model` | Match through HuggingFace `cardData.base_model` |
| `line_interp` | Size-aware interpolation within the same model line |
| `self_reported` | Uploader-provided `evalResults` only |
| `none` | No usable evidence |

Inheritance is rejected when the actual model size differs too much from the
reference. This catches small draft heads, MTP heads, and unrelated forks that
would otherwise borrow a larger base model's score.

## Family grouping

`models/grouper.py` groups related repos by:

1. `cardData.base_model`, when the relationship is explicitly `quantized`.
2. Normalized repository names.

The normalizer removes packaging and quantization suffixes such as `-GGUF`,
`-AWQ`, `-GPTQ`, and `-FP16`. It preserves organization namespaces, minor versions, checkpoint dates,
and instruction/chat suffixes. Models that reference a base without an explicit
quantization relationship keep their own repository identity. Quantizations of
those derived models stay with the derived checkpoint rather than its ancestor.

`family_id` reflects these checkpoint distinctions and is recomputed during
grouping, including for models loaded from an existing cache.

Within a family, the ranker evaluates all members and variants but keeps only
the best result for the final table.

## Candidate variants

For each `ModelInfo`, `engine/ranker.py` builds candidate variants:

- Existing GGUF files are evaluated by quantization.
- Extreme low-bit GGUF variants are skipped unless explicitly requested with
  `--quant`.
- Official safetensors-only repos can receive synthetic GGUF estimates for
  common community conversions such as `Q4_K_M`, `Q5_K_M`, `Q6_K`, and `Q8_0`.
- Pre-quantized repos such as AWQ, GPTQ, FP8, and BF16 are not given synthetic
  GGUF variants.

Apple Silicon and CPU-only rankings are restricted to GGUF candidates for
runtime stability. Linux with NVIDIA can also rank non-GGUF AWQ/GPTQ/FP16/BF16
repos.

## Ranking

For each candidate variant:

1. Estimate memory.
2. Check whether it can run.
3. Estimate tok/s and attach speed confidence/range metadata.
4. Resolve benchmark evidence.
5. Compute a quality score.
6. Keep the best variant for the model family.

The final sorting key stays close to the displayed quality score, with a small
direct-benchmark bonus and a CPU-only penalty. Full-GPU candidates are already
favored inside the score through the runtime-fit and speed adjustments, so the
sort key does not add a second full-GPU bonus.

See [Scoring](scoring.md) for the score details.

## Output

Output is split by surface:

- `output/ranking.py` renders hardware panels and recommendation tables.
- `output/json_output.py` renders ranking, `plan`, and `upgrade` JSON.
- `output/plan.py` renders `plan` tables.
- `output/upgrade.py` renders upgrade comparison tables.
- `output/display.py` re-exports those functions for older imports.

Normal ranking tables show memory required, estimated generation speed, fit
type, and published date. `--details` switches to download-oriented metadata.
Speed color is based on absolute usability, while `~` marks estimates with a
range and `?` marks low-confidence, backend-sensitive estimates.
