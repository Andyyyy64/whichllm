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
| Partial offload | `0.42`-`0.88`, based on spill ratio |
| CPU-only | `0.50` |

Light partial offload is penalized less than heavy offload. MoE models receive
a milder penalty when the active parameter working set can plausibly stay on
GPU while inactive experts spill to CPU RAM.

The final family selection key does not add a separate full-GPU bonus. Runtime
fit is already reflected in the quality score through the multiplier above and
the speed adjustment below. CPU-only results receive a small extra sort penalty
when mixed with GPU-backed candidates.

## Speed adjustment

Speed is treated as a usability gate. It is not the main quality signal.

Required speed depends on fit:

| Fit | Required speed |
| --- | ---: |
| Full GPU | `8 tok/s` |
| Partial offload | `4 tok/s` |
| CPU-only | `1.5 tok/s` |

Candidates below the required speed receive up to `-8` points. Candidates above
it receive up to `+8` points.

After ranking, if any candidate is at least `5 tok/s`, whichllm drops candidates
below `1.5 tok/s`. This avoids recommending models that technically fit but are
not practical to use.

The reported speed is a point estimate, not a live benchmark. Ranking also
exposes speed confidence:

| Confidence | Range factor | Typical cases |
| --- | ---: | --- |
| `medium` | `0.60x`-`1.60x` | Normal GPU estimates, synthetic GGUF estimates, AMD shared-memory APU MoE estimates |
| `low` | `0.35x`-`2.00x` | CPU-only, partial offload, unknown bandwidth, Apple Silicon MoE |
| `high` | `0.85x`-`1.20x` | Reserved for future measured-speed data |

With `--status`, speed cells use `~` for medium-confidence estimates and `?`
for low-confidence estimates. JSON exposes the same data as
`speed_confidence`, `speed_range_tok_per_sec`, and `speed_notes`.

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
