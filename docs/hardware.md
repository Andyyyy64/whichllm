# Hardware detection and simulation

whichllm detects the current machine and can also simulate hardware for
purchase planning.

The source of truth is the `hardware/` package plus curated registry data in
`data/gpu.py`. `constants.py` remains as a compatibility export layer for older
imports.

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
- usable VRAM bytes, when a runtime headroom is active
- NVIDIA compute capability, when known
- CUDA or ROCm version, when known
- memory bandwidth estimate
- whether the GPU uses shared memory

## NVIDIA

NVIDIA detection tries `nvidia-ml-py` first. If NVML is unavailable, fails to
initialize, or returns no devices, whichllm falls back to:

```bash
nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv,noheader,nounits
```

If a driver rejects `clocks.max.memory`, whichllm retries the older
`name,memory.total` query.

For known cards, curated data and strict `dbgpu` lookups provide:

- memory bandwidth
- compute capability

The max memory clock is used when a marketing name covers multiple memory
types. For example, GTX 1650 GDDR5 and GDDR6 cards share the same broad driver
name, so whichllm uses the reported memory clock when available and falls back
to the conservative bandwidth when it is not.

DGX Spark / NVIDIA GB10 uses unified system memory. When the driver reports
`memory.total` as unavailable, whichllm treats GB10 as shared memory and uses
system RAM for fit checks.

Compute capability is used to warn when a card is below the minimum expected by
common local inference tools.

## NVIDIA Jetson Orin

Jetson (Tegra) boards are unified-memory SoCs, and neither signal desktop
detection relies on works there. The driver reports one generic name per SoC
generation — every Orin module is `Orin (nvgpu)`, from a 4 GB Orin Nano to a
64 GB AGX Orin — and the memory queries return `NVML_ERROR_NOT_SUPPORTED` or
`[N/A]`, so the memory-clock trick that separates same-name desktop variants is
unavailable too.

Only Orin is handled. Xavier and earlier stop at a JetPack that ships neither a
Tegra NVML nor a Tegra `nvidia-smi`, so no GPU name reaches this code there.

whichllm resolves the module from the kernel device tree instead, which needs
no root and no NVIDIA tooling:

```bash
cat /proc/device-tree/compatible | tr '\0' '\n'
# nvidia,p3768-0000+p3767-0000   carrier board + module
# nvidia,p3767-0000              module
# nvidia,tegra234                SoC generation
```

The module name replaces the generic one, so the curated tables resolve it like
any other GPU, and system RAM is used for fit checks as it is for DGX Spark. A
module that is not recognised keeps its generic name and resolves no bandwidth,
rather than a product being guessed from the SoC generation — `tegra210` alone
covers both Jetson Nano and TX1. Detection still requires the driver to
enumerate the GPU, so a board with a broken or absent CUDA stack reports none.

Bandwidth values are data-sheet peaks. Orin NX 16GB is the one verified against
measured throughput; the other Orin modules come from the module data sheets.
JetPack 6.2 "Super Mode" raises Orin Nano bandwidth (8 GB to 102, 4 GB to 51)
under a higher nvpmodel profile with no hardware change — the base value is
kept, and `--ram-bandwidth` covers that case. Orin NX is unchanged at 102 GB/s
in both modes.

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
Radeon 780M are modeled as shared-memory systems. On Linux the GPU's memory is
its BIOS carve-out plus the GTT, the system RAM the amdgpu driver lets it map
(`mem_info_gtt_total` in sysfs; half of RAM by default, raised with
`amdgpu.gttsize` or `ttm.pages_limit`). Without a GTT reading, a small
aperture means whichllm uses the system memory pool for fit checks instead of
treating it as a tiny discrete GPU.

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

Apple Silicon uses unified memory, but by default Metal lets the GPU use only
part of it: its recommended working set, which llama.cpp logs at start-up as
`recommendedMaxWorkingSetSize` and uses as its GPU budget. whichllm assumes
about two thirds of the memory up to 32 GB and three quarters above (a 16 GB
Mac gives the GPU about 10.7 GB), and the hardware panel says so. A limit
raised with `sudo sysctl iogpu.wired_limit_mb=<MB>` is detected and used
instead. On a detected Mac, `--vram` states the GPU's limit directly; with
`--gpu "Apple M4"`, `--vram` is the Mac's memory size and the working set is
taken from it. Memory bandwidth is looked up by chip family when known.

A model larger than the working set can still run: llama.cpp keeps the
remaining layers on the CPU, in the same unified memory, as long as the total
fits in RAM. That partial offload is not penalized like discrete PCIe offload,
because the weights stay in unified memory.

## CPU and memory

CPU detection reads:

- `/proc/cpuinfo` on Linux
- `sysctl` on macOS
- `wmic` on Windows, then PowerShell / CIM when `wmic` is unavailable or only
  returns a header

Physical core count comes from `psutil`, with a Linux `/proc/cpuinfo` fallback.

RAM comes from `psutil.virtual_memory()`. Disk free space is checked under the
user's home directory by default.

## GPU simulation

Use `--gpu` to simulate a GPU:

```bash
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090 Laptop"
whichllm hardware --gpu "Apple M3 Max"
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
```

Simulation uses the `dbgpu` package for a TechPowerUp-backed GPU database.
whichllm adds extra handling for common aliases and Apple Silicon chips because
those are not covered by dbgpu.

`Laptop`, `Laptop GPU`, `Mobile` and `Max-Q` select the laptop card, which often
has half the bandwidth of the desktop one and a different amount of memory. The
name a driver reports works too, such as `NVIDIA GeForce RTX 5090 Laptop GPU`.
A laptop name with no matching card needs `--vram` and never falls back to the
desktop card.

Use `--vram` when a GPU name is ambiguous, unknown, or has multiple variants:

```bash
whichllm --gpu "RTX 5060 Ti" --vram 16
whichllm hardware --gpu "Unknown GPU" --vram 24
```

For detected integrated or unified-memory GPUs, use `--vram` and
`--ram-bandwidth` to override the automatically detected usable capacity and
memory bandwidth:

```bash
whichllm --vram 8 --ram-bandwidth 68
whichllm hardware --vram 8 --bandwidth 68
```

`--bandwidth` sets the selected GPU's memory bandwidth. `--ram-bandwidth` sets
system RAM bandwidth, which bounds everything the CPU reads: CPU-only runs,
MoE experts kept in RAM and layers offloaded from a discrete GPU. An iGPU or
unified-memory GPU reads from the same RAM, so `--ram-bandwidth` sets its
bandwidth too unless `--bandwidth` is also given.

If multiple GPUs are detected, pass `--gpu-index` to choose the GPU shown by
`whichllm hardware`:

```bash
whichllm --gpu-index 1 --vram 8 --bandwidth 68
```

By default, whichllm applies a small automatic VRAM headroom before fit checks.
This avoids recommending models that only fit on paper but overflow in runtimes
that need extra graph buffers or loader overhead. Tune it with:

```bash
whichllm --vram-headroom 1.5GB
whichllm --vram-headroom 10%
whichllm --vram-headroom none
```

Multi-GPU simulation accepts repeated flags, comma-separated values, and count
shorthand:

```bash
whichllm --gpu "2x RTX 4090"
whichllm --gpu "RTX 4090" --gpu "RTX 3090"
whichllm --gpu "RTX 4090, RTX 3090"
```

`--vram` is only supported for a single simulated GPU. For multi-GPU
simulation, use known GPU names so whichllm can resolve each card's VRAM from
the GPU database.

## System RAM bandwidth

`whichllm hardware` shows the RAM bandwidth it assumed and where it came from:

| Source | How |
| --- | --- |
| detected | Windows: memory modules' transfer rate times bus width (`Win32_PhysicalMemory`), capped at the CPU family's channel count |
| unified | Apple Silicon: the chip's unified-memory bandwidth |
| estimated | The CPU family's typical memory, e.g. DDR5-5600 dual channel for Core Ultra 200S or AM5 Ryzen |
| assumed | Nothing known: dual-channel DDR4-3200, 51.2 GB/s |

On Linux the modules come from systemd's udev, which exports the DMI memory
table without root (`/run/udev/data/+dmi:id`, systemd 248 and later). WSL has
no DMI table, so whichllm asks Windows through `powershell.exe`. Windows and
WSL cache the module list for a week, since reading it starts PowerShell.
Elsewhere, or in a container, the CPU-family estimate applies. Pass the real
figure with `--ram-bandwidth` (transfer rate in MT/s x 8 bytes x channels /
1000, e.g. DDR5-6000 dual channel = 96).

## Fit types

Compatibility checks classify a candidate into one of three fit types:

| Fit | Meaning |
| --- | --- |
| `full_gpu` | Required memory fits in available GPU memory |
| `partial_offload` | GPU plus usable system RAM can hold the model |
| `cpu_only` | Usable system RAM can hold the model without GPU |

If neither GPU memory nor usable RAM can hold the model, the candidate is not
ranked.

A partial offload on a discrete GPU is placed one of two ways, shown in the
JSON `placement` field:

- **Expert offload** (`expert_offload`): for a mixture-of-experts model whose
  attention, shared weights and KV cache fit in VRAM, the routed experts of
  the first N layers stay in RAM, as llama.cpp's `--n-cpu-moe N` does. Each
  token reads only its few active experts from RAM, so gpt-oss-120b runs at
  interactive speed on a 16 GB card with 64 GB of RAM. The table shows these
  as `GPU+RAM`, and the warning names the `--n-cpu-moe` value.
- **Layer offload** (`layer_offload`): whole layers run on the CPU and read
  every weight from RAM, so speed falls to roughly what RAM bandwidth allows.

Unified-memory systems have no separate RAM pool to move experts to and keep
the old shared-memory estimate.

whichllm keeps a bounded system-RAM reserve for the OS and other processes.
Use `--ram-budget available` to cap partial-offload planning to the current
available RAM reported by the OS, or pass a fixed budget such as
`--ram-budget 8GB`.

## Multiple GPUs

For fit checks, whichllm plans a layer split instead of pretending all VRAM is
one perfect device. Each device first pays its own runtime buffer (driver
context, graph buffers, allocator slack). The budget left on each device is what
the plan can use, and weights and KV cache are assigned in proportion to it.
The runtime-only part of the estimate — activation buffers and framework
overhead — cannot be split by a layer split, so it is reserved on the device
with the most capacity. A utilization factor is applied last: homogeneous sets
receive a less severe reduction than heterogeneous sets.

A device that cannot pay its own runtime buffer stays out of the pool before any
of that is decided: two identical cards beside a card with 256 MiB free keep the
homogeneous 0.95 budget, the speed estimate ignores the card that holds nothing,
and one remaining card is treated as a plain single-GPU fit at its full
available memory. Because the shares are proportional to what each device has
left, the only way a split fails is the runtime-only part: when that fits no
single device, the placement is rejected even though the pooled budget would
have covered the estimate, and the pooled budget then only carries the partial
offload. The plan also describes the GPU-resident part of an expert or layer
offload, which is what the devices hold once the rest is in system RAM.

If a dedicated GPU is present, low-aperture shared-memory integrated GPUs are
not added to the fit pool. This avoids treating unrelated system RAM and
dedicated VRAM as one full-GPU target.

A model that fits on the largest card alone is treated as a single-GPU run:
no split budget, no split overhead and no multi-GPU speed discount.

When a model does need the split, the speed estimate follows how llama.cpp
splits layers: in proportion to each card's free memory, with each token
reading every card's share in turn. The decode time is the sum of share /
bandwidth over the cards, plus a 5% allowance for handing activations between
them. Two identical cards run about as fast as one would if the model fit;
a slower second card slows the split in proportion to its share. When a card's
bandwidth is unknown, a flat 0.7x factor applies. Multi-GPU speed stays
low-confidence, because real performance also depends on the split mode,
PCIe/NVLink, and the backend.

This is a practical fit approximation. It does not model every tensor-parallel
or pipeline-parallel runtime configuration.

The estimated per-GPU memory use can still differ from a real run: LM Studio and
llama.cpp choose their own layer split. To compare a real run, record the model
and quantization, context length, runtime version, GPU layer split, per-GPU
memory use, KV-cache placement, and generation speed. See
[#104](https://github.com/Andyyyy64/whichllm/issues/104) and
[#189](https://github.com/Andyyyy64/whichllm/issues/189) for the reported
multi-GPU cases.

## Disk checks

The compatibility check also compares estimated model weight size with free
disk space. If the model cannot be downloaded, it is marked unrunnable.

## Known limitations

- GPU bandwidth is a lookup or database estimate, not a live benchmark.
- Speed estimates are planning numbers. The default table and JSON fields such
  as `speed_confidence` and `speed_range_tok_per_sec` show uncertainty.
- Driver, runtime, batch size, prompt length, and thermal limits can change real
  performance.
- Multi-GPU runtime behavior depends on the inference backend and is only
  approximated.
- Apple and shared-memory APU behavior is modeled as unified-memory style, but
  real results still depend on OS pressure and memory bandwidth.
