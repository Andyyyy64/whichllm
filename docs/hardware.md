# Hardware detection and simulation

whichllm detects the current machine and can also simulate hardware for
purchase planning.

The source of truth is the `hardware/` package plus GPU constants in
`constants.py`.

## Detected data

The ranker receives a `HardwareInfo` object with:

- GPU list
- CPU name
- physical CPU cores
- AVX2 and AVX-512 support
- total RAM
- free disk space
- OS name

Each GPU is represented as `GPUInfo`:

- name
- vendor
- VRAM bytes
- NVIDIA compute capability, when known
- CUDA or ROCm version, when known
- memory bandwidth estimate
- whether the GPU uses shared memory

## NVIDIA

NVIDIA detection tries `nvidia-ml-py` first. If NVML is unavailable, fails to
initialize, or returns no devices, whichllm falls back to:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
```

For known cards, `constants.py` provides:

- memory bandwidth
- compute capability

DGX Spark / NVIDIA GB10 uses unified system memory. When the driver reports
`memory.total` as unavailable, whichllm treats GB10 as shared memory and uses
system RAM for fit checks.

Compute capability is used to warn when a card is below the minimum expected by
common local inference tools.

## AMD

On Linux, AMD detection tries `rocm-smi` first:

- product name
- VRAM
- ROCm driver version

If `rocm-smi` is unavailable, it falls back to `lspci` and then
`/sys/class/drm`.

On Windows, whichllm uses `Win32_VideoController` as a fallback for AMD GPUs.
When possible, it also reads the 64-bit dedicated-memory value from the
`Control\Video` registry path because `AdapterRAM` is a 32-bit field and can
cap larger cards around 4 GB.

AMD shared-memory APUs are treated differently from discrete GPUs. Names such
as Strix Halo, Ryzen AI MAX, Radeon 8050S, Radeon 8060S, Radeon 890M, and
Radeon 780M are modeled as shared-memory systems. If the reported VRAM is just
a small aperture, whichllm uses the system memory pool for fit checks instead
of treating it as a tiny discrete GPU.

## Intel

Intel integrated GPUs are detected on Linux through `lspci` or sysfs, and on
Windows through `Win32_VideoController`. They do not normally report dedicated
VRAM, so whichllm records them with `0` dedicated VRAM and labels them as
shared memory.

Discrete Intel Arc cards are kept as dedicated-memory GPUs when the device name
and memory report look like a discrete adapter.

The Intel backend factor is lower than NVIDIA, AMD, and Apple because local LLM
GPU inference support is less mature.

## Apple Silicon

On macOS, whichllm uses:

```bash
system_profiler SPHardwareDataType -json
```

Apple Silicon uses unified memory, so the detected chip memory is treated as
available GPU memory. Memory bandwidth is looked up by chip family when known.

Partial offload on Apple Silicon is not penalized like discrete PCIe offload.
Weights still live in unified memory, so the speed penalty is milder.

## CPU and memory

CPU detection reads:

- `/proc/cpuinfo` on Linux
- `sysctl` on macOS
- `wmic` on Windows

Physical core count comes from `psutil`, with a Linux `/proc/cpuinfo` fallback.

RAM comes from `psutil.virtual_memory()`. Disk free space is checked under the
user's home directory by default.

## GPU simulation

Use `--gpu` to simulate a GPU:

```bash
whichllm --gpu "RTX 4090"
whichllm hardware --gpu "Apple M3 Max"
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
```

### Multi-GPU simulation

Simulate multiple GPUs with comma-separated names or a count prefix:

```bash
whichllm --gpu "2x RTX 4090"
whichllm --gpu "4x H100"
whichllm --gpu "RTX 5080,RTX 5060 Ti"
whichllm --gpu "2x RTX 3090,RTX 4090"
```

VRAM is pooled across all GPUs for fit determination. Speed estimation uses a
tensor-parallel model where the slowest GPU is the bottleneck, with inter-GPU
communication overhead (PCIe or NVLink) factored in.

Simulation uses the `dbgpu` package for a TechPowerUp-backed GPU database.
whichllm adds extra handling for common aliases and Apple Silicon chips because
those are not covered by dbgpu.

Use `--vram` when a GPU name is ambiguous, unknown, or has multiple variants:

```bash
whichllm --gpu "RTX 5060 Ti" --vram 16
whichllm hardware --gpu "Unknown GPU" --vram 24
```

`--vram` requires `--gpu` and only works with single-GPU simulation.

## Fit types

Compatibility checks classify a candidate into one of three fit types:

| Fit | Meaning |
| --- | --- |
| `full_gpu` | Required memory fits in available GPU memory |
| `partial_offload` | GPU plus usable system RAM can hold the model |
| `cpu_only` | Usable system RAM can hold the model without GPU |

If neither GPU memory nor usable RAM can hold the model, the candidate is not
ranked.

whichllm reserves about 20% of system RAM for the OS and other processes.

## Multiple GPUs

For fit checks, whichllm sums available GPU memory. For speed estimates, it uses
the largest detected GPU as the representative device.

If a dedicated GPU is present, low-aperture shared-memory integrated GPUs are
not added to the fit pool. This avoids treating unrelated system RAM and
dedicated VRAM as one full-GPU target.

This is a practical approximation. It does not model every tensor-parallel or
pipeline-parallel runtime configuration.

## Disk checks

The compatibility check also compares estimated model weight size with free
disk space. If the model cannot be downloaded, it is marked unrunnable.

## Known limitations

- GPU bandwidth is a lookup or database estimate, not a live benchmark.
- Speed estimates are planning numbers. Use `--status` or JSON fields such as
  `speed_confidence` and `speed_range_tok_per_sec` to see uncertainty.
- Driver, runtime, batch size, prompt length, and thermal limits can change real
  performance.
- Multi-GPU runtime behavior depends on the inference backend and is only
  approximated.
- Apple and shared-memory APU behavior is modeled as unified-memory style, but
  real results still depend on OS pressure and memory bandwidth.
