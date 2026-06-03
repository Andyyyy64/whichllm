# Troubleshooting

This page lists common issues and the first checks to make.

## No GPU detected

Run:

```bash
whichllm hardware
```

If an NVIDIA GPU is missing:

- check that the driver is installed
- check `nvidia-smi`
- check that `nvidia-ml-py` can load NVML

whichllm falls back to `nvidia-smi`, but it still needs the NVIDIA driver tools
to be working.

If an AMD GPU is missing:

- on Linux, check `rocm-smi`, `lspci`, and `/sys/class/drm`
- on Windows, check that PowerShell can read `Win32_VideoController`
- for Ryzen AI / Radeon integrated graphics, check whether `whichllm hardware`
  shows shared memory instead of a tiny 512 MB or 4 GB adapter

If an Intel iGPU is missing:

- Linux detection uses `lspci` or `/sys/class/drm`
- Windows detection uses `Win32_VideoController`
- many Intel iGPUs do not expose dedicated VRAM, so they may be shown as shared
  memory graphics

## Simulate hardware instead

If detection is unavailable or you are planning a purchase, use `--gpu`:

```bash
whichllm --gpu "RTX 4090"
whichllm hardware --gpu "Apple M3 Max"
whichllm --gpu "RTX 5060 Ti" --vram 16
```

Use `--vram` when the GPU name has multiple memory variants or is not in the
database.

## `--cpu-only` conflicts with `--gpu`

These flags are mutually exclusive:

```bash
whichllm --cpu-only --gpu "RTX 4090"
```

Choose one:

```bash
whichllm --cpu-only
whichllm --gpu "RTX 4090"
```

## `--vram` requires `--gpu`

`--vram` is an override for a simulated GPU. It does not change detected
hardware by itself.

Use:

```bash
whichllm --gpu "RTX 3060" --vram 12
```

## No compatible models found

Try:

```bash
whichllm --status
whichllm --cpu-only
whichllm --refresh
```

Common causes:

- the selected `--quant` is too restrictive
- `--min-speed` is too high
- `--evidence strict` filters out all candidates
- the requested context length is too large
- available RAM is too low after reserving space for the OS
- disk free space is too low for the model weights

For very small machines, remove optional filters first:

```bash
whichllm --top 20
```

## Results look stale

whichllm caches model data for 6 hours and benchmark data for 24 hours.

Force a refresh:

```bash
whichllm --refresh
whichllm plan "qwen 7b" --refresh
```

The caches live under:

```text
~/.cache/whichllm/
```

## The top pick has `~`, `!sr`, or `?`

These markers describe benchmark evidence:

| Marker | Meaning |
| --- | --- |
| `~` | Inherited or interpolated benchmark evidence |
| `!sr` | Uploader-reported benchmark only |
| `?` | No benchmark evidence |

Use stricter evidence when you want only independently matched benchmark data:

```bash
whichllm --evidence strict
whichllm --direct
```

Use `--evidence base` when base-model matches are acceptable but interpolation
and self-reported values are not.

## The largest model did not win

That is expected. whichllm scores:

- benchmark quality
- model size
- quantization loss
- full GPU vs partial offload vs CPU-only
- estimated speed
- evidence confidence
- source trust
- generation lineage

A smaller current-generation model with strong direct evidence can beat a
larger model that only barely fits or relies on stale benchmark data.

## Estimated speed differs from real speed

Speed is an estimate based on:

- model weight size
- MoE active parameters
- GPU memory bandwidth
- quantization efficiency
- backend factor
- partial-offload penalty

Real performance depends on the inference runtime, driver, prompt length,
batching, thermal limits, and background memory pressure.

Use `--status` to see the estimate and its confidence marker:

```bash
whichllm --status
```

Markers in the speed column:

- `~`: estimated speed range is available
- `?`: low-confidence estimate; runtime/backend differences can be large

JSON includes the same information as `speed_confidence`,
`speed_range_tok_per_sec`, and `speed_notes`.

## Apple Silicon partial offload looks different

Apple Silicon uses unified memory. Partial offload does not cross a discrete
PCIe boundary, so whichllm applies a milder speed penalty than it does for
discrete GPUs.

The same is true for recognized AMD shared-memory APUs such as Strix Halo,
Ryzen AI MAX, and Ryzen AI / Radeon 890M-class integrated graphics.
DGX Spark / NVIDIA GB10 is handled the same way when NVIDIA reports GPU memory
as unavailable.

On Windows, `Win32_VideoController.AdapterRAM` can cap around 4 GB. whichllm
uses the 64-bit registry memory value when it is available, and treats known
shared-memory APUs as unified-memory style devices instead of tiny discrete
GPUs.

## `run` says `uv is required`

Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then retry:

```bash
whichllm run
```

## `run` cannot download a model

Possible causes:

- the model is gated on HuggingFace
- local HuggingFace authentication is missing
- the selected GGUF filename no longer exists
- network access failed
- disk space is too low

Try a known public GGUF model first:

```bash
whichllm run "qwen 2.5 1.5b gguf"
```

## How much disk space does `run` need?

Normal ranking commands do not download model weights. They cache Hugging Face
model metadata and benchmark metadata under the whichllm cache.

`whichllm run` downloads the selected GGUF file through `huggingface_hub`. The
required disk space is roughly the selected GGUF file size plus normal Hugging
Face cache overhead.

By default, Hugging Face stores downloaded files under:

```text
~/.cache/huggingface/hub
```

You can move that cache by setting `HF_HOME` or `HF_HUB_CACHE`.

Cleanup is handled by the Hugging Face cache tools:

```bash
hf cache scan
hf cache delete
```

whichllm does not currently delete model files automatically after a run.

## Ollama names do not match HuggingFace IDs

JSON output returns HuggingFace repo IDs:

```bash
whichllm --top 1 --json | jq -r '.models[0].model_id'
```

Ollama model names often use a different naming scheme. Map the HuggingFace ID
to your local Ollama model name before calling `ollama run`.

## Debugging a specific model

Use `plan` to inspect memory requirements:

```bash
whichllm plan "Qwen2.5-72B" --quant Q4_K_M
whichllm plan "Qwen2.5-72B" --quant Q8_0 --context-length 32768
```

Use JSON output when filing issues:

```bash
whichllm --gpu "RTX 4090" --status --json
whichllm hardware
```

Include:

- OS
- GPU name and VRAM
- CPU and RAM
- command used
- expected result
- actual result
