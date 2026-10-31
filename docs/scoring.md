# Scoring

whichllm does not pick the largest model that fits. It ranks candidates by a
composite score that tries to answer a more practical question:

> Of the models that can run here, which one is likely to be the best usable
> choice?

The source of truth is `engine/ranker.py`.

## Inputs

Each candidate score uses:

- model metadata from HuggingFace
- detected or simulated hardware
- estimated VRAM/RAM fit
- estimated tok/s
- quantization type
- benchmark evidence
- downloads and likes
- source organization
- model lineage and generation

The score is capped to `0..100`.

## Benchmark evidence

Independent benchmark matches are not all treated equally.

| Source | Weight | Meaning |
| --- | ---: | --- |
| `direct` | `0.62` | Exact independent benchmark match |
| `base_model` | `0.55` | Match through `cardData.base_model` |
| `variant` | `0.50` | Suffix-stripped variant match |
| `line_interp` | `0.40` | Size-aware model-line interpolation |
| `self_reported` | `0.30` | Uploader-provided HuggingFace eval only |
| `none` | `0.00` | No benchmark evidence |

`self_reported` evidence is intentionally weak. HuggingFace model cards can
contain useful evaluation data, but it is not the same as an independent
leaderboard.

## Size score

Model size is used as a rough world-knowledge proxy:

```text
size_score = 4.2 * log2(params_b) + 9
```

The result is capped at `35`.

For dense models, `params_b` is the parameter count. For MoE models, whichllm
uses total parameters for quality because all experts contribute to stored
knowledge. Active parameters are used later for speed.

## Quantization penalty

Lower-bit quantization can make a larger model fit, but it also reduces quality.
The score core is multiplied by `(1 - quant_penalty)`.

Examples:

| Quant | Penalty |
| --- | ---: |
| `Q8_0` | `0.01` |
| `Q6_K` | `0.02` |
| `Q5_K_M` | `0.03` |
| `Q4_K_M` | `0.05` |
| `Q3_K_M` | `0.08` |
| `Q2_K` | `0.25` |
| `IQ2_XXS` | `0.40` |
| `Q1_0` | `0.55` |

A label an uploader extended (`Q6_K_L`, `UD-Q4_K_XL`, `IQ1_M_XL`) takes the
penalty of its format, so a 1-bit file cannot pass for a 4-bit one.

Extreme low-bit variants are excluded by default when better candidates exist.
They can still be requested explicitly with `--quant`.

## Evidence confidence

After benchmark and size are combined, weak evidence is dampened:

| Evidence state | Multiplier |
| --- | ---: |
| Direct benchmark | `1.00` |
| Inherited evidence | `0.78` |
| Self-reported evidence | `0.55` |
| No benchmark | `0.55` |

For inherited benchmark evidence, the raw score is also scaled by confidence
before entering the scoring function. Line interpolation therefore receives a
double discount: once for its interpolation confidence and once for being
inherited evidence.

## Runtime fit

The candidate's runtime form matters:

| Fit | Multiplier |
| --- | ---: |
| Full GPU | `1.00` |
| MoE experts in RAM (`expert_offload`) | `0.90` |
| Layers in RAM (`layer_offload`) | `0.42`-`0.86`, based on spill ratio |
| CPU-only | `0.50` |

Light layer offload is penalized less than heavy offload. Expert offload keeps
almost the full-GPU multiplier: only the active experts are read from RAM for
each token, and the speed estimate already accounts for that. The small
discount covers the extra runtime setup.

The final family selection key does not add a separate full-GPU bonus. Runtime
fit is already reflected in the quality score through the multiplier above and
the speed adjustment below. CPU-only results receive a small extra sort penalty
when mixed with GPU-backed candidates.

## Speed adjustment

Speed is treated as a usability gate. It is not the main quality signal.

Every placement (full GPU, experts in RAM, layers in RAM, CPU-only) is held
to the same bar, `8 tok/s`: fast enough not to wait on in a chat or a coding
loop.

Below the bar, the quality score is multiplied by `(speed / 8) ** 0.5` and
loses up to another `8` points, so a 4 tok/s model keeps about 70% of its
quality and a 2 tok/s model about half. A benchmark lead cannot carry a model
that answers at 3 tok/s past one that answers at 25. Candidates above the bar
receive up to `+8` points.

After ranking, if any candidate is at least `5 tok/s`, whichllm drops candidates
below `1.5 tok/s`. This avoids recommending models that technically fit but are
not practical to use.

When a newer generation of a model line ranks above an older one of about
the same size (total parameters within 25%, dense or MoE alike, and the same
specialization), the older one is left out of the list and named under the
table instead (`older_generations` in JSON). Qwen3.8-27B, Qwen3.6-27B and
Qwen3.5-27B filling the top three said less than three different models
would. An older generation that ranks above the newer one stays.

The reported speed is a point estimate, not a live benchmark. On a discrete
GPU with a partial fit it adds the GPU's reads at VRAM bandwidth to the CPU's
reads at system RAM bandwidth; with experts in RAM the CPU reads only the
active experts. CPU-only speed follows RAM bandwidth too. The CPU constants
come from llama.cpp runs (see `engine/performance.py`). Ranking also exposes
speed confidence:

| Confidence | Range factor | Typical cases |
| --- | ---: | --- |
| `medium` | `0.60x`-`1.60x` | Normal GPU estimates, CPU-only and expert offload with a known RAM bandwidth, synthetic GGUF estimates, AMD shared-memory APU MoE estimates |
| `low` | `0.35x`-`2.00x` | Layer offload, unknown GPU or RAM bandwidth, Apple Silicon MoE |
| `high` | `0.85x`-`1.20x` | Reserved for future measured-speed data |

Speed cells are colored by absolute usability: red is under `4 tok/s`, yellow
is `4-8 tok/s`, green is `8-30 tok/s`, and bright green is `30+ tok/s`. `~`
marks medium-confidence estimates with a range, and `?` marks low-confidence
estimates. JSON exposes the same uncertainty data as `speed_confidence`,
`speed_range_tok_per_sec`, and `speed_notes`.

## Source trust

The source organization contributes a small adjustment:

- official model organizations receive a small bonus
- trusted GGUF converters can inherit that trust
- known repackagers receive a small penalty

The adjustment is intentionally small. It should break ties, not replace
benchmark and fit signals.

## Popularity

Downloads and likes act as tie-breakers. Their weight is lower when benchmark
evidence is strong and higher when evidence is weak.

Popularity has no effect for direct benchmark matches.

## Generation lineage

Some benchmark sources are frozen. A model released after a frozen leaderboard
cannot appear there, while older models can keep strong but stale scores.

whichllm uses family-specific lineage maps to avoid that inversion. Newer
generations can receive a small bonus; older generations can receive a small
penalty. This is applied carefully so direct benchmark evidence still matters.

Examples of tracked lineages include:

- Qwen
- Llama
- DeepSeek
- Gemma
- Phi
- Mistral
- GLM
- Kimi
- Granite
- OLMo
- T5 (incl. Flan-T5, mT5, ByT5, T5Gemma)

## Benchmark markers

The table score can include a marker:

| Marker | Meaning |
| --- | --- |
| none | Direct independent benchmark evidence |
| `~` | Estimated or inherited benchmark evidence |
| `!sr` | Self-reported HuggingFace eval only |
| `?` | No benchmark evidence |

Top-pick confidence is computed from the score gap, benchmark status, and fit
type. Partial-offload and CPU-only top picks are reported with lower confidence
than full-GPU direct-benchmark winners.

## Why a smaller model can win

A smaller model can outrank a larger one when it has:

- stronger current benchmark evidence
- a newer generation signal
- better quantization quality
- full-GPU fit instead of partial offload
- higher estimated speed
- a more trustworthy source

That is intentional. whichllm ranks likely usable quality, not parameter count
alone.
