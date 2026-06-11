# How it works

whichllm has one main job: start with the user's hardware, collect candidate
models, estimate what can run, and rank the results.

The implementation is intentionally split into small packages:

```text
src/whichllm/
├── cli.py
├── constants.py
├── hardware/
├── models/
├── engine/
└── output/
```

## Request flow

The default `whichllm` command follows this path:

1. Validate CLI flags.
2. Detect hardware.
3. Load model cache or fetch from HuggingFace.
4. Load benchmark cache or fetch benchmark sources.
5. Group related model repos into families.
6. Flatten families back into rankable candidates.
7. Rank every candidate variant.
8. Backfill missing published dates for top results.
9. Print a Rich table or JSON.

The `plan`, `upgrade`, `run`, `snippet`, and `hardware` subcommands reuse parts
of the same pipeline.

## Hardware detection

`hardware/detector.py` orchestrates detection. Each detector is fail-safe and
returns an empty result on failure.

| Module | Role |
| --- | --- |
| `hardware/nvidia.py` | Uses `nvidia-ml-py`; falls back to `nvidia-smi` |
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

## Caches

Both caches live under `~/.cache/whichllm/`.

| File | TTL | Contents |
| --- | --- | --- |
| `models.json` | 6 hours | Serialized `ModelInfo` data |
| `benchmark.json` | 24 hours | Combined benchmark score map |

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

1. `cardData.base_model`, when available.
2. Normalized repository names.

The normalizer removes common suffixes such as `-GGUF`, `-AWQ`, `-GPTQ`,
`-Instruct`, `-Chat`, `-FP16`, and date suffixes. It also handles versioned
model lines such as Qwen, Llama, Mistral, and DeepSeek.

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

`output/display.py` renders:

- hardware panels
- recommendation tables
- JSON output
- `plan` tables and JSON
- `upgrade` comparison tables and JSON

Normal ranking tables show published date and downloads. With `--status`, the
table instead shows memory required, estimated speed, and fit type. Speed cells
use `~` for normal estimates with a range and `?` for low-confidence,
backend-sensitive estimates.
