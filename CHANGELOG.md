# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `whichllm bench` measures a model's real decode speed, prints it next to the
  estimate and saves the result in `~/.cache/whichllm/reports/`. It runs on
  `llama-server`, Ollama, LM Studio or any OpenAI-compatible server through
  `--backend`, with the placement the estimate assumed. It asks before a
  download, stops what it started even after Ctrl+C, and saves nothing that
  identifies you.
- `whichllm report` shares a measurement. It prints the whole report, asks, and
  opens a prefilled GitHub issue that you submit yourself; whichllm sends nothing.
  `--manual` covers runtimes `bench` cannot drive, and `--print` opens nothing.
- `whichllm run` opens a verified local LM Studio GGUF artifact instead of
  downloading a second copy, with the repeatable `--lm-studio-path` option. A
  match needs the resolved repository, the exact artifact path, and every part
  of a split GGUF beside the entry file in the same library. With llama-cli
  installed the recipe's `-hf` source becomes `-m <entry part>`, and the printed
  offload hint shows the same command; otherwise the generated script opens the
  local files and keeps `huggingface-hub` only as the fallback. The files are
  checked again when they are opened — right before llama-cli starts, and inside
  the generated script after the isolated build — so a set that disappeared is
  reported and downloaded instead of crashing. A set that was already incomplete
  names the missing part, including a set whose first part is the one that is
  gone. (#20, #186)
- The whichllm index: a daily prebuilt file with the model catalog,
  `config.json` shapes for every family, published dates and merged
  benchmark scores, published on GitHub Pages. Recommendations, `plan`,
  `upgrade`, `run` and `snippet` read it instead of making hundreds of
  Hugging Face requests and scraping leaderboards, so a cold start takes
  about a second and is unaffected by rate limits. The build refuses to
  publish when quality gates fail. `--live` keeps the direct fetch, and
  `WHICHLLM_INDEX_URL` points at a mirror or turns the index off.
  GGUF repositories in the index carry their real file sizes instead of
  estimates from the parameter count (unsloth's Qwen3.5-2B Q4_K_M is
  1.28 GB, not the estimated 1.06), reused while a repository's commit is
  unchanged and fetched for up to 300 repositories a day.
- `HF_TOKEN` is sent to huggingface.co when set. A mirror in `HF_ENDPOINT`
  (hf-mirror.com and the like) gets it only over https and only with
  `WHICHLLM_HF_TOKEN_TO_ENDPOINT=1`, so a token is not handed to a third
  party by default.
- MoE expert offload. When a mixture-of-experts model does not fit in VRAM
  but its attention, shared weights and KV cache do, whichllm keeps the
  routed experts of the first N layers in system RAM, as llama.cpp's
  `--n-cpu-moe N` does, and estimates speed from what each token actually
  reads: its few active experts from RAM, the rest from VRAM. On a 16 GB
  card with 64 GB of DDR5, gpt-oss-120b now ranks #5 at an estimated
  18 tok/s (it was #21). The CPU side is calibrated against llama.cpp on an
  RTX 5080 with DDR5-5600: Qwen3.6-35B-A3B at `--n-cpu-moe` 12-40 measured
  44-83 tok/s, and the estimates land within 25% of the medians. The table
  shows these fits as `GPU+RAM`, the warning names the `--n-cpu-moe` value,
  and JSON rows carry a `placement` object.
- Run recipes. Below the ranking, whichllm prints the `llama-server`
  command and LM Studio settings that run the top pick the way it was
  estimated (`--n-cpu-moe N` when experts stay in RAM, `-ngl N` for layer
  offload). `--markdown` prints it as a code block and `--json` adds a
  `recipe` object to every row. `run` and `snippet` use the ranking's layer
  split instead of always asking for every layer on the GPU.
  The LM Studio settings match LM Studio 0.4.26's model loader, and the
  Ollama note says only that Ollama picks its own split: Ollama 0.40 keeps
  MoE experts in RAM by itself and ran Qwen3.6-35B-A3B on an RTX 5080 at
  about 100 tok/s.
- `run` chats through llama.cpp's `llama-cli` when it is installed, with
  the same flags as the printed recipe, so it uses the GPU and keeps MoE
  experts in RAM. A named model is left to llama.cpp's `--fit`. Without
  llama-cli, `run` keeps using llama-cpp-python and first says that PyPI
  only ships its source: it compiles for 20 minutes or more and runs on the
  CPU only (on an RTX 5080, 22.7 minutes to the first reply).
- System RAM bandwidth is detected (Windows memory modules), estimated from
  the CPU family, or taken from Apple's unified memory, and shown by
  `whichllm hardware`.
- The index records where its scores came from: each benchmark source with
  the version or snapshot it read (live or the committed snapshot) and how
  many entries it gave. It also lists popular releases from the last 45 days
  (5,000+ downloads) that have no benchmark score: original releases by id,
  and finetunes by their base when nothing up the lineage has a score. A gap
  in the sources then shows up in the daily build log instead of in a user's
  ranking. `--json` output gains a
  `data` object with the index build time and these sources, or
  `"source": "live"`.

### Changed

- Multi-GPU fit now plans a layer split per device instead of comparing one
  combined budget. Each device pays its own runtime buffer first, the weights
  and KV cache that stay on the GPUs are assigned in proportion to what is
  left, and the part a split cannot divide (activation buffers and framework
  overhead) is reserved on the device with the most capacity. A device that
  cannot pay its own buffer stays out of the pool, so it no longer changes the
  homogeneous budget or the split speed, and one remaining card is a plain
  single-GPU fit at its full memory. Because the shares are proportional, the
  only failing case is the runtime-only part: when it does not fit any device
  after that device's runtime buffer, the split is rejected even when the pooled
  budget would have covered the estimate, the GPU part runs on the largest card
  alone, and an oversized model keeps its pooled partial offload.
  `--json` reports the per-device share under `placement.devices`. This is the
  estimator part of #189; the per-device comparison on real hardware is still
  open. (#104, #189)
- **Breaking:** `--ram-bandwidth` is now system RAM bandwidth, no longer an
  alias of `--bandwidth`. It bounds CPU-only runs, experts in RAM and
  offloaded layers; for an iGPU or unified memory it still sets the GPU's
  bandwidth, so `--vram 8 --ram-bandwidth 68` keeps working. Use
  `--bandwidth` to override a discrete GPU's VRAM bandwidth. With a
  discrete GPU present, the hardware panel says so when the flag is used.
- CPU-only speed and layers offloaded from a discrete GPU are estimated from
  RAM bandwidth instead of a size heuristic and a fixed 0.45x factor. Dense
  models spilling layers on a fast GPU are slower than before (Qwen3.6-27B
  at 128K context on an RTX 4090: 5 tok/s, not 16), CPU-only MoE models are
  faster, and CPU-only candidates must reach 4 tok/s to score as usable.

- `--profile coding` is no longer limited to repositories with "coder" in
  their names: strong general models such as gpt-oss, Devstral and
  Qwen3.6-27B are eligible, and a code-tuned model gets a 2-point edge over
  an otherwise equal general one. `--profile vision` includes natively
  multimodal models (Gemma 3/4, Qwen3.5/3.6) through the Hub's
  `image-text-to-text` tag, not only names with "VL".
- `--profile coding`, `math` and `vision` rank on benchmarks for that task
  instead of the general score: Artificial Analysis's SciCode,
  Terminal-Bench, LiveCodeBench and FrontierCode for coding, AIME 2025 for
  math, and MMMU-Pro plus the curated vision index for vision. Each result is
  mapped onto the general 0-100 scale through the models it shares with the
  general index, so a model without task results keeps its general score.
  On an RTX 3060, gpt-oss-20b and Qwen3-Coder-Next now rank above the dense
  Qwen3.8-27B for coding, and the name-based 2-point edge only applies to
  code-tuned models with no coding results. JSON rows gain
  `benchmark_basis`, and the index gains an optional `task_scores` map.
- Every placement is held to one usable speed, 8 tok/s (layer offload and
  CPU-only used 4). Below it a model's quality is scaled by
  `(speed / 8) ** 0.5`, where it used to lose at most 8 points, which a
  benchmark lead easily outweighed. On a 16 GB M4 the #1 is now
  gemma-4-26B-A4B at 25 tok/s instead of Qwen3.8-27B at 4, and a 3 tok/s
  Qwen3.8-27B leaves the RTX 3060 top 5. The golden constraint for a #1 that
  needs offload is replaced by one for the whole top 3.
- `--speed usable` hides models under 8 tok/s, the same speed the ranking
  holds every placement to (it was 10), and the speed column turns green
  from 8 tok/s.
- A model that fits on the largest GPU alone is no longer treated as a
  multi-GPU run, and a split's speed follows each card's bandwidth instead
  of a flat 0.7x. On two RTX 3090s, Qwen3.8-27B at Q5_K_M fits one card and
  is estimated at 25.5 tok/s (it was split at Q6_K and estimated at 14.5);
  a 4090 + 3060 split runs at about 0.6x of the 4090 alone.
- When a newer generation of a model line ranks above an older one of the
  same size, the older one is left out and named under the table
  (`older_generations` in JSON). The 4090 top 3 was Qwen3.8-27B, Qwen3.6-27B
  and Qwen3.5-27B; it is now Qwen3.8-27B, gemma-4-26B-A4B and
  Qwen3.6-35B-A3B. Gemma ids such as `diffusiongemma-26B` are no longer read
  as Gemma 2.
- New releases that neither LiveBench nor the pinned AA index covers get a
  general score from at least three of AA's broad benchmarks, mapped the same
  way, so models such as GLM-5.3-Flash, Qwen3.8-Flash-Next and Gemma 4 12B
  are no longer unscored. On models that have both, the two agree within
  about 4 points on average.
- Aider polyglot is no longer used. Its leaderboard stopped at 2025-10 and its
  display names no longer matched any model, so it contributed nothing. The
  curated vision index no longer raises general scores (it made
  Qwen3-VL-235B's general score its vision value).
- Pretrain-only checkpoints are never recommended: `-Base`/`-pt` names,
  checkpoints whose publisher ships an instruct version derived from them
  (`Qwen2.5-7B`, `Qwen2.5-Coder-7B`, `Llama-3.1-8B`) and their quantized
  uploads. The coding top 10 used to include the base `Qwen2.5-Coder-7B`.

- The KV cache is computed per layer from each model's `config.json` instead
  of from its parameter count. whichllm reads the config for the families
  near the top of a ranking (and for the model `plan` shows), caches it for a
  week, and ranks again. GQA models were undercounted by 4-5x at long context
  (Qwen3-8B at 32K: 4.5 GiB, not 0.9 GiB), so at `-c 32k` and above some
  models no longer claim a full-GPU fit. Hybrid models such as Qwen3.6, whose
  linear-attention layers keep no K/V, now get their smaller cache. Sliding
  windows, multi-head latent attention and recurrent state are handled; an
  unrecognised layout keeps the old estimate.
- The curated benchmark snapshots (the pinned AA index, AA's name map,
  LiveBench and the vision index) moved out of Python into JSON files in
  `src/whichllm/data/`. Each records the date it was read, and every AA
  entry says whether it came from an AA model page or is a hand estimate.
  `scripts/import_livebench_csv.py` now writes the LiveBench file itself.
  Scores are unchanged.

### Fixed

- `--gpu "RTX 5090 Laptop"` works, and so does the name a driver reports.
  `Laptop` was unknown, `GeForce RTX 5090 Laptop GPU` silently simulated the
  desktop card at twice the bandwidth, and `RTX 5090 Max-Q` became an RTX A5000.
- Speed estimates for dense GGUF models on a GPU are fitted to 379 published
  llama.cpp results on 194 GPUs, from 2 to 131 GiB. They ran about 30% low. With
  each GPU held out of the fit, the median error is 9% and 90% of runs land
  within a third, against 33% and 50% before. Decode time is the file read at
  96% of memory bandwidth plus 2.4 ms per token, with a factor per vendor, lower
  factors for older designs and Apple's Ultra chips, and a 1,100 GB/s ceiling on
  HBM cards. The quant makes no measurable difference. MoE and CPU estimates
  keep their own calibration.
- The 36 GB M4 Max and M5 Max and the 36 and 96 GB M3 Max use the bandwidth of
  their cut-down memory bus. They were listed a third too high.
- Speed confidence follows the published runs instead of being `medium` almost
  everywhere. A dense GGUF with a known file size, all on an NVIDIA GPU from
  Pascal on, a desktop RDNA Radeon or Apple Silicon is `high`, with a range of
  0.75x to 1.30x. On Intel, GCN Radeon, NVIDIA before Pascal, HBM and laptop GPUs
  it is `low`. A dense GGUF split across identical cards moves from `low` to
  `medium`. At least 90% of the runs at each level fall inside its range.
- CPU speed estimates for dense GGUF models are fitted to 255 LocalScore results
  on 138 CPUs. They ran at about half the real speed. On desktop CPUs the median
  error is now 12%, with 97% of runs inside the `medium` range. The CPU share of
  a layer split uses the same rate, which a published `-ngl` sweep backs. Without
  a GPU, Xeon, EPYC and Threadripper are `low` confidence. MoE estimates on the
  CPU keep their own calibration.
- An older generation keeps its own row only when it outscores the newer one by
  more than 2 points. Any lead at all used to keep it listed.
- A derivative that claims under a tenth of its base model's parameters no longer
  ranks. No quant shrinks a model that far, so these are heads, broken counts or
  misread names, and one listed at 14,000 tok/s on an 8 GB Mac.
- GGUF files too small for the quant on their label no longer count as the model.
  A 4.7 GB LoRA named `F16` passed for a 27B model at 7 tok/s on the CPU. Files
  with `MTP` in the name stay, since most are whole models.
- `8bit` and `2.9bpw` in a repo name no longer read as a size of 8B and 2.9B.
- Recipes for unsloth's dynamic quants and Bartowski's XL files work. A label
  like `UD-Q4_K_XL` lost its suffix, and the Hub answers 400 to
  `llama-server -hf repo:Q4_K`.
- Quant rates come from 5,000 real GGUF files instead of nominal bits. A repo
  the index has not sized yet was estimated too small: Q4_K_M by 9%, Q3_K_M by
  13%, Q2_K by 22%.
- Labels with no rate no longer rate as a 4-bit file. IQ2_XS, Q2_K_L and the
  1-bit IQ1_M_XL lost 5% of their quality like Q4_K_M, which let a much
  smaller file win. A suffix an uploader adds now rates as its format.
- Real models with "tiny" in their name are no longer hidden: Ling-3.0-tiny,
  Granite 4.0 H-Tiny and DeepSeek-VL2-tiny. The rule was meant for CI fixtures,
  which `tiny-random` and the `-internal-testing` orgs still exclude.
- `--gpu "DGX Spark"`, `--gpu "Jetson AGX Orin"` and other simulated NVIDIA
  unified-memory GPUs no longer count as dedicated VRAM with the same amount of
  system RAM next to it.
- Pretrain-only checkpoints that the naming rules missed are no longer
  recommended: Olmo-3-1025-7B, Mistral-7B-v0.1, Gemma 2B and 7B, Gemma 2 2B,
  Llama-2-7b-hf and PaliGemma's `pt` release. Their chat release has another
  name or is not in the catalog. They ranked from #13 down on the machines
  in the golden set.
- `run`, `snippet` and `upgrade` rank the way `whichllm` does. `run` and
  `snippet` skipped the VRAM headroom, the size floor and the rule that
  prefers models with direct benchmarks, so they could pick another model
  than the one at the top of the table. `upgrade` skipped the headroom and
  ranked simulated GPUs with the default RAM bandwidth.
- An official repo is only ranked at quants its GGUF uploads carry. It could
  be recommended at a quant nobody published: Qwen3.8-Flash-Next at Q3_K_M
  was #2 on an RTX 5090 and on a 128 GB Mac, with no file to download and no
  recipe. A repo with no upload in the catalog keeps its estimates.
- Pre-quantized checkpoints are sized from their real files. NVFP4 repos
  report only part of their parameters, so they were counted at under half
  their size: Muse-Glimmer-NVFP4 is 19.7 GB and showed as a 9.5 GB full-GPU
  fit on a 16 GB card. The index now carries the real size of NVFP4, FP8,
  AWQ, GPTQ and BNB repos, and the fallback rates for `--live` and repos it
  has not reached yet are corrected.
- Fit and speed of an official repo use the real size of the GGUF file its
  recipe downloads. On a 16 GB RTX 5080, Qwen3.6-35B-A3B at Q4_K_M was
  sized at 20.2 GB and given `--n-cpu-moe 12`, but unsloth's file is
  22.1 GB; it now gets `--n-cpu-moe 15`.
- The runtime memory beyond the weights is closer to what llama-server
  allocates. The compute buffer was counted at 0.08 byte per parameter
  (2.6 GB for a 27B); llama-server used 146 MiB. With the real 13.4 GB
  file, that would have split Qwen3.8-27B at Q3_K_M across a 16 GB card,
  where it runs fully on the GPU at 55 tok/s using 12.8 GiB. Recurrent
  state (Qwen3.5 and later, Qwen3-Next, Granite 4.0, LFM2) is counted for
  llama-server's 4 default slots. Dense models now fit one quant higher
  in a few places (Qwen3.8-27B at Q6_K on a 24 GB card).
- GGUF variants: `run` downloads part 1 of a split file (it could pick a
  later part), and vision projectors (`mmproj`) and draft models are no
  longer listed as a model's F16, BF16 or Q8_0 variant. With real sizes, a
  variant counts only its own file or split parts: an MTP build next to the
  plain one, or Q4_0's ARM repacks, would otherwise add up (a 27B Q4_K_M at
  121 GB).
- Apple Silicon is budgeted at Metal's default GPU working set, about two
  thirds of unified memory up to 32 GB and three quarters above, not all of
  it. On a 16 GB Mac the GPU got 15.2 GB, so the #1 was a 15 GB gemma-4-26B
  build that llama.cpp would not keep on the GPU; it is now gpt-oss-20b. A
  model past the working set can spill layers to the CPU within unified
  memory, and a raised `iogpu.wired_limit_mb` is detected.
- RAM bandwidth is detected on Linux from systemd's udev DMI properties,
  which need no root, and on WSL by asking Windows through
  `powershell.exe`; both fell back to a CPU-name estimate. Windows and WSL
  cache the module list for a week instead of starting PowerShell on every
  run.
- AMD shared-memory APUs on Linux (Strix Halo and others) get their BIOS
  carve-out plus the GTT the driver lets the GPU map, instead of all of
  system RAM.
- `--gpu "RTX 3060"` means the 12 GB card (360 GB/s), and `--vram` picks the
  memory variant it names. The 8 GB card's 240 GB/s was used for both, so
  the RTX 3060 archetype was a third slower than the real card.
- `run` without a model name ranks with the index's benchmark scores, as
  `snippet` does. It read only the `--live` benchmark cache, which the index
  path never writes, so its pick was ranked without any benchmarks.
- Speculative-decoding draft heads (DFlash, `-draft`) and diffusion LMs are
  never ranked. A draft's size is read from its name, so it passed for the
  full model; `google/diffusiongemma-26B-A4B-it` was recommended at Q3_K_M
  on CPU-only machines because official repos get assumed GGUF builds,
  although llama.cpp does not run it.
- NVFP4 and FP8 checkpoints need an Ampere or newer NVIDIA GPU. vLLM serves
  them through Marlin kernels there, so a Turing card no longer gets them;
  an NVFP4 pick before Blackwell notes that it runs weight-only.
- Models whose config.json keeps the expert counts under `text_config`
  are recognised as mixture-of-experts. The Hub parser only read the top
  level, so Qwen3.8-Flash-Next (180B, 10 of 512 experts per token) was
  estimated as a dense 180B at 2.4 tok/s on an M4 Max and never got its
  experts placed in RAM. Active parameters come from the config shape.
- `run`, `snippet` and `plan` resolve a model name to the model you meant.
  Among repos with the same size label the one whose parameter count was
  closest to the query used to win, and that count is often parsed from the
  label itself, so obscure uploads came first: `qwen3 8b` picked a DFlash
  draft head, `qwen 7b` CodeQwen1.5 and `gemma 3 12b` a Gemma 4 fine-tune.
  Ties now go to the closest name (a VL, Coder or `-FP8` the query does not
  mention counts against a repo), then the original over a quantized
  upload, then downloads. Draft heads and pretrain-only checkpoints come
  last unless the query names them.
- LiveBench and Artificial Analysis are combined on one scale. They used to
  be merged by taking the higher value, but LiveBench's normalization runs
  about 17 points above AA for the same models, so any model with a
  LiveBench entry outranked models only AA measures (Qwen3.6-27B scored
  83.5 from LiveBench against 61.0 on AA, while Qwen3.8-27B kept its AA
  77.8). LiveBench is now fitted onto AA's scale, the two are averaged, and
  hand-estimated entries only count for models neither source measured.
  Qwen3.8-27B now leads on 16 GB+ GPUs at default context, and
  Qwen3.6-35B-A3B with its experts in RAM is the top pick on 6-12 GB GPUs.
- Artificial Analysis scores are on one scale again. The snapshot mixed
  values from before and after AA rescaled its Intelligence Index, and
  models AA has deprecated (Kimi K2, GLM-5, ...) kept their old, roughly
  a third higher values even when the live fetch worked, so they outranked
  newer models. Every entry is re-read from AA's index v4.3.2 with new
  normalization bounds, DeepSeek V4's dated checkpoints map to their own
  repositories, and while AA reports another index version (v5.0 today)
  whichllm uses the v4.3.2 snapshot instead of mixing scales. Thanks to
  devangpratap (#195).

## [0.5.23] - 2026-10-29

### Fixed

- Trending models are fetched again. The Hugging Face API rejects
  `sort=trending` with HTTP 400, so the two trending lists were always
  skipped and newly popular repositories only appeared once they ranked by
  downloads. They now use `sort=trendingScore`. Since 0.5.22 the failing
  lists also marked every model list as incomplete, which cached it for 30
  minutes instead of 6 hours.

## [0.5.22] - 2026-10-28

### Added

- A golden ranking test set ranks a frozen catalog for 15 hardware
  archetypes and checks the results against constraints, with a
  reviewable top-pick snapshot. See CONTRIBUTING.md for updating it.

### Changed

- Ranking is about three times faster. Matching a community repo to a
  benchmark entry with the same model name used to scan every benchmark id
  for each candidate model; a name index built once per ranking replaces the
  scan. Matches and their order are unchanged.

### Fixed

- Curated recent models that still fail to load after retries (rate limits,
  server or network errors) are now reported on stderr with a count, and the
  partial model list is cached for 30 minutes instead of 6 hours. Missing or
  gated curated repositories do not shorten the cache.
- `--profile vision` and `--profile any` no longer reuse a cached model list
  that was fetched without vision candidates. Backfilling publish dates no
  longer extends the age of the model cache.
- The ranking footer shows the current benchmark snapshot month (2026-09)
  instead of 2026-05. A test keeps it equal to the newest curated fallback.
- `whichllm snippet` without a model name now prints code for the best ranked
  model that has a runnable GGUF build, as the help text promised, instead of
  the most-downloaded GGUF repository.
- `--bandwidth` / `--ram-bandwidth` help and docs now say that both set the
  selected GPU's memory bandwidth. On an iGPU or unified memory that is system
  RAM bandwidth; on a discrete GPU it does not change CPU offload estimates.
- `CONTRIBUTING.md` points at `src/whichllm/data/gpu.py` for GPU entries, the
  pre-commit Ruff hook matches the locked Ruff version, and the README
  development setup uses `uv sync --locked --dev`.

## [0.5.21] - 2026-10-19

### Fixed

- Artificial Analysis maps Kimi K2 only to Kimi-K2-Instruct and the April
  Command R+ row only to c4ai-command-r-plus. Kimi-K2-Base and the August
  Command R+ release no longer inherit their sibling's live AA score.
  Fallback snapshot values are unchanged. (#184, #188)

## [0.5.20] - 2026-10-03

### Added

- Recommendations now detect exact GGUF artifacts in the current LM Studio
  library, with repeatable `--lm-studio-path` overrides for legacy or custom
  locations. Local libraries remain read-only. (#20)

### Fixed

- Live Artificial Analysis scores replace stored AA scores even when lower or
  zero. Models absent from the live response retain their stored AA scores.
  Use `--refresh` to bypass an existing benchmark cache. (#101)
- DeepSeek V3 and V3-0324, and GLM-4.5 and GLM-4.5-Air, have separate
  Artificial Analysis mappings. (#181)
- Archived leaderboard requests honor valid `Retry-After` values with bounded
  retries. A later page's exhausted 429 retains earlier pages and warns on
  stderr. Partial results use the existing 24-hour cache; `--refresh` retries
  collection. A first-page failure still fails the source. (#180, #182)
- AMD PCI ID `1002:1586` maps to the SKU-neutral Strix Halo family name in
  sysfs and lspci fallbacks, with shared-memory classification. (#36, #154)
- Xavier names from NVML and nvidia-smi use the NVIDIA shared-memory path.
  This does not resolve a failure to enumerate any GPU. (#146)

### Changed

- Rich ranking tables show known GGUF file sizes and mark estimates.
  Estimates retain their origin through cache serialization. Missing size
  metadata for a split part makes the whole variant an estimate. (#168)
- Model cache schema 4 invalidates older caches that lack size provenance.
- CLI execution and helpers move out of the Typer facade while retaining
  command options, exact repository lookup, and help behavior. (#41, #162)

## [0.5.19] - 2026-09-20

### Added

- `-v` as a short alias for `--version`. (#159)

## [0.5.18] - 2026-09-19

### Fixed

- `plan owner/repo` fetches exact Hugging Face repository IDs outside the cached
  catalog, with errors for inaccessible repositories and insufficient metadata.
  (#163, #164)
- Model grouping preserves organization namespaces, minor versions, checkpoint
  dates, and instruction/chat variants. Only explicit quantizations follow a referenced base model; other
  derivatives keep their own identity. Ranking and JSON `family_id` values now
  distinguish these checkpoints, including when loading an existing cache. (#171)
- AMD shared-memory detection uses a recognized ROCm product SKU even when
  `Card Series` contains a generic name, preserving the reported display name
  and avoiding a 512 MB aperture being treated as dedicated VRAM. (#36)


### Changed

- Document the limits of multi-GPU weight and KV-cache placement estimates.
  (#104, #175)

## [0.5.17] - 2026-09-19

### Added

- Fetch the Qwen3.5 / Qwen3.6 / Qwen3.8 releases and the current coder line
  (`Qwen3-Coder-Next`, `Qwen3-Coder-480B-A35B-Instruct`), map them to their
  Artificial Analysis leaderboard names, and add curated index entries so they
  rank with real benchmark evidence instead of dropping out. (#24)
### Changed

- `--profile math` now keeps general reasoning models alongside math-specialized
  repos, instead of only repos with "math" in the name. (#24)
- Split ranking orchestration, scoring, filters, and variant selection into
  focused modules. (#41, #145)
- Keep cache-directory tests portable across platforms. (#149)

### Fixed

- `Qwen/Qwen3-Coder-Next` no longer reports ~33B active parameters: its 512
  expert / top-10 config fell through to the generic expert-fraction estimate,
  so VRAM and speed were computed for a much denser model than the 3B-active
  one the model card describes. (#24)
- Qwen3.8 repositories are no longer demoted as an older generation: they had no
  lineage entry, so they fell through to the generic `qwen3` pattern and scored
  a smaller generation bonus than Qwen3.5 and Qwen3.6. (#24)
- Replaced the hand-estimated curated index values for `Qwen/Qwen3.6-27B` and
  `Qwen/Qwen3-Coder-30B-A3B-Instruct` with their measured ones. Both were off by
  enough to reorder neighbouring models offline. (#24)
- Non-UTF-8 model and benchmark caches are treated as cache misses instead of
  crashing during decoding. (#160, #161)
- Apple Silicon GPU memory budgets respect the configured
  `iogpu.wired_limit_mb` value. (#152, #153)
- RTX Ada Generation laptop workstation GPUs resolve bandwidth from the GPU
  catalog without confusing mobile and desktop variants. (#144)
- Restore the README star-history chart. (#166)

## [0.5.16] - 2026-08-14

### Changed

- Split benchmark caching, fetching, indexing, lineage, lookup, and shared types
  into focused modules while preserving the existing
  `whichllm.models.benchmark` import surface. (#41, #143)
- Locked the uv and Ruff versions used by contributors and required CI checks,
  so unchanged commits are validated with the same development tools. (#150,
  #151)

### Fixed

- Generated `whichllm run` and `whichllm snippet` scripts now encode model IDs,
  GGUF filenames, and quantization types as Python literals, preventing crafted
  Hugging Face metadata from changing the generated code structure. (#147)
- Synthetic GGUF recommendations now resolve only to direct quantizations of
  the selected checkpoint, rejecting fine-tunes, merges, conflicting lineage,
  and artifacts without explicit provenance. (#155, #156)

## [0.5.15] - 2026-07-03

### Changed

- Split the oversized Hugging Face model fetcher into focused modules while
  keeping the existing `whichllm.models.fetcher` import surface. (#41, #140)

### Fixed

- Output now resolves ranked GGUF recommendations to the actual downloadable
  artifact repo and filename when the ranked base model and runnable GGUF live
  in different Hugging Face repos. (#137, #138)
- AA index scoring now uses retuned normalization bounds and a refreshed
  fallback snapshot for the reworked Artificial Analysis scale. (#101, #139)
- Invalid ranking filters such as `--top 0`, negative `--min-speed`, and
  negative `--min-params` now fail with clear errors instead of silently
  producing wrong output. (#142)

## [0.5.14] - 2026-06-29

### Added

- Added sliding-window attention metadata to model fetching and KV cache
  estimation, improving VRAM estimates for models that use SWA. (#124)
- Added curated Intel Arc Pro B70 / Battlemage G31 detection and simulation,
  including the `0xe223` PCI device ID and 32 GB VRAM / 608 GB/s bandwidth
  defaults. (#93, #136)

### Fixed

- Model and benchmark metadata fetches now request `gzip, deflate` instead of
  brotli, avoiding broken `br` responses from mirrors or intermediate servers.
  (#128, #136)

## [0.5.13] - 2026-06-25

### Added

- Added `HF_ENDPOINT` support for Hugging Face model metadata fetches, so users
  behind a mirror can point whichllm at a compatible Hub endpoint. (#128, #131)
- Added manual detected-GPU overrides for usable VRAM and bandwidth, which helps
  iGPU and unified-memory systems where automatic detection is too conservative.
  (#132, #133)
- README now points users toward safer first-run flags when they want
  full-GPU, usable-speed recommendations with extra VRAM headroom.

### Fixed

- Search terms such as `7B`, `0.5B`, and `500M` now match model parameter size
  instead of plain substrings, so `qwen 7b` no longer returns `1.7B` or
  `30B-A3B` models by accident. (#107, #126)
- GGUF sizing now treats FP16 and ternary `TQ1_0` / `TQ2_0` quant types
  correctly, avoiding underestimates for those files. (#125)

## [0.5.12] - 2026-06-18

### Added

- Default ranking tables now show memory required, estimated generation speed,
  fit type, and published date. Download counts are still available with
  `--details`.
- Added `--speed any|usable|fast` as named generation-speed filters while
  keeping `--min-speed` for exact tok/s thresholds.
- Added `--fit gpu` as a natural alias for full-GPU-only recommendations.
- Added `--markdown` / `-m` for pasteable GitHub-Flavored Markdown ranking
  tables. (#111)
- Added `--vram-headroom` and `--ram-budget` so users can avoid edge VRAM fits
  and cap partial-offload planning to available or fixed system RAM.

### Changed

- Speed color now reflects practical generation speed. `~` and `?` remain
  estimate-confidence markers instead of being the primary speed color.

## [0.5.11] - 2026-06-18

### Added

- Multi-GPU simulation for repeated `--gpu` flags, comma-separated GPU specs,
  and count shorthand such as `2x RTX 4090`. The fit model uses a conservative
  effective VRAM budget and keeps speed confidence low for split-device
  recommendations. (#113)
- `python -m whichllm` now runs the CLI entrypoint. (#116)
- `--gpu-only` and `--fit full-gpu` now filter recommendations to models that
  fit fully in GPU VRAM. `--fit any` keeps the existing behavior. (#119, #122)
- T5 lineage support so T5-family models get version-aware benchmark handling.

### Fixed

- Fixed UTF-8 decoding for cached model and benchmark data on systems whose
  default filesystem encoding is not UTF-8. (#121)
- GTX 1650 simulation now distinguishes GDDR5 and GDDR6 variants by memory
  clock instead of treating every card as the slower 128 GB/s model. (#115)
- RAM reserve logic now uses a bounded reserve formula instead of a fixed 80%
  usable-RAM cap, which avoids underestimating machines with more system
  memory. (#103)

## [0.5.10] - 2026-06-11

### Fixed

- Strong partial-offload candidates are no longer buried below weaker full-GPU
  models because the final ranking sort no longer counts full-GPU fit a second
  time after runtime-fit and speed penalties have already been applied. Light
  partial offload is penalized less aggressively, while heavy dense offload
  remains strongly discounted. (#105, #108)
- MoE partial-offload scoring now uses the active parameter working set when it
  can plausibly stay on GPU, so active-small MoE models are not penalized like
  dense models with the same total parameter count. (#105, #108)

## [0.5.9] - 2026-06-10

### Added

- MXFP4 and NVFP4 4-bit quantization support across ID/filename parsing, VRAM
  estimation, quality penalties, speed efficiency, and family grouping.
  Repos shipping these formats were previously labeled FP16 and their VRAM
  requirement overestimated about 3.5x. (#99)
- Apple M5-family entries for `--gpu` simulation. (#92)
- Kepler-era Quadro bandwidth and compute capability entries. (#75)

### Fixed

- AMD discrete GPU detection on Linux: rocm-smi names are read from the
  correct `Card Series` key, compound lspci names such as
  `Navi 22 [Radeon RX 6700/6700 XT/6750 XT ...]` resolve bandwidth, sysfs VRAM
  enriches the fallback path, and discrete cards are no longer mislabeled
  `shared memory`. Adds RX 6750 XT / RX 6700 / RX 6650 XT / RX 6600 series and
  Radeon AI PRO R9700 to the bandwidth catalog. (#61, #68)
- Community GGUF repos without `base_model` metadata (for example
  `unsloth/...-GGUF`) now inherit the official model's benchmark score by
  name matching instead of falling through to no evidence. (#94)
- GPU bandwidth detection no longer depends solely on the hand-curated
  catalog. When a detected card is missing from `GPU_BANDWIDTH`, bandwidth is
  now resolved from the bundled TechPowerUp database (dbgpu, 2824 GPUs) using
  strict name matching only: an exact normalized hit or a name plus VRAM-size
  bin, never fuzzy. Laptop / Mobile / Max-Q names can no longer inherit a
  desktop card's bandwidth, and VRAM bins written without a space
  (`RTX A2000 12GB`) are recognized. This fixes the cluster of reports where
  an uncatalogued GPU showed `BW: N/A`, was estimated at `0.0 tok/s`, and
  received oversized recommendations (#74, #98).
- Artificial Analysis Intelligence Index is fetched live again. The
  artificialanalysis.ai leaderboard migrated to the Next.js App Router and no
  longer ships a `__NEXT_DATA__` blob, so every run logged
  `AA Index fetch failed ... __NEXT_DATA__ payload not found` and silently used
  the frozen snapshot. The scraper now parses the App Router RSC stream
  (`self.__next_f.push(...)`), canonicalizes AA's variant-suffixed display
  names (`(Reasoning)`, `(high)`, ...) for mapping, and overlays live scores on
  top of the curated fallback so a successful fetch can only add coverage. The
  legacy `__NEXT_DATA__` path is kept as a secondary fallback.

## [0.5.8] - 2026-06-05

### Added

- `--context-length` now accepts shorthand values such as `64k` and `128k`.
- JSON ranking output now includes benchmark source and confidence metadata.
- Asahi Linux / Apple Silicon detection now recognizes Apple CPU and GPU names.
- Added GPU catalog coverage for `NVIDIA RTX A3000 Laptop GPU`, `RTX 3050`,
  `RTX 5060`, `RTX 5070 Ti`, `RX 9070`, and `RX 9070 XT`.

### Fixed

- A3000 Laptop 6GB systems no longer get `0.0 tok/s` / heavy partial-offload
  recommendations at the top just because bandwidth was missing.
- Windows CPU detection now falls back through PowerShell/CIM when `wmic` does
  not return a useful CPU name.
- Models that cannot hold the requested context are demoted instead of staying
  near the top of the ranking.
- Hugging Face and benchmark fetches now retry transient failures such as 429s
  before falling back or failing.
- `Error fetching models:` now includes useful detail even when the underlying
  network exception message is empty.
- Upgrade tables now show `0 GB` VRAM instead of treating zero as missing.

### Changed

- Curated registry data was split out of `constants.py` into
  `whichllm.data.*` modules.
- Troubleshooting and cache documentation now better explain disk-cache paths
  and stale fetch behavior.

## [0.5.7] - 2026-05-20

### Added

- LiveBench fallback data is now kept inline so benchmark scoring remains
  available without relying on a generated sidecar file.

### Fixed

- DGX Spark / NVIDIA GB10 is now detected as a shared-memory NVIDIA GPU when
  NVIDIA reports `memory.total` as unavailable.
- `whichllm run` now provides a Transformers `offload_folder`, avoiding crashes
  when large models need disk offload.
- Cache paths now respect `XDG_CACHE_HOME`, including ignoring relative values
  per the XDG base directory specification.
- Apple Silicon is now treated as shared memory in fit detection.
- Benchmark score fetching now runs concurrently.

## [0.5.6] - 2026-05-18

### Added

- Speed estimates now include confidence metadata and an estimated tok/s range
  in table and JSON output, so uncertain backend/model predictions are visible.
- Windows now has an AMD/Intel GPU detection fallback via
  `Win32_VideoController`, including 64-bit registry memory reads for GPUs
  where `AdapterRAM` is capped around 4 GB.

### Fixed

- MoE speed estimates now use active-parameter metadata and a
  bandwidth-scaled read floor, improving shared-memory APU estimates without
  over-promoting sparse models on high-bandwidth GPUs.
- Newer MoE model metadata now recognizes A3B-style active-parameter names.
- Ryzen AI / Radeon 890M-class Windows iGPUs are modeled as shared-memory AMD
  GPUs instead of CPU-only or tiny-VRAM discrete GPUs.
- Mixed dedicated-GPU plus shared-memory-iGPU systems no longer sum unrelated
  memory pools as one full-GPU target.
- Windows AMD GPUs no longer receive a misleading ROCm-only warning when
  Vulkan or DirectML backends may be valid.

## [0.5.5] - 2026-05-17

### Fixed

- `whichllm run` now resolves auto-picked GGUF recommendations to a real
  GGUF repository and file before launch, instead of falling back to the
  official Transformers repository. This fixes the accidental Transformers path
  for models such as `Qwen/Qwen3.6-27B`.

## [0.5.4] - 2026-05-17

### Fixed

- Strix Halo / Ryzen AI MAX systems are now modeled as AMD shared-memory APUs
  instead of tiny-VRAM discrete GPUs. `STRXLGEN`, `Radeon 8050S`,
  `Radeon 8060S`, and related names get a 256 GB/s bandwidth estimate and use
  the system shared-memory pool for fit checks, avoiding false CPU-only,
  99%-offload, and `0 tok/s` recommendations.

## [0.5.3] - 2026-05-17

### Added

- Linux Intel integrated GPU detection via `/sys/class/drm`, so Intel iGPU
  systems are no longer always treated as CPU-only.
- NVIDIA `nvidia-smi` fallback detection when pynvml is missing, NVML init
  fails, or NVML reports no devices.
- Apple-prefixed Apple Silicon simulator aliases, so `--gpu "Apple M3 Max"`
  resolves the same way as `--gpu "M3 Max"`.

### Fixed

- `whichllm run` transformers chat generation now passes tokenizer mappings
  into `model.generate(**inputs)`, fixing the `KeyError: 'shape'` crash path.
- RTX 5060 Ti bandwidth lookup now reports 448 GB/s instead of `N/A`.

### Changed

- README install guidance now prefers `uvx` / `uv tool install`.
- Removed the old marketing note from the repository and added sponsor metadata.

## [0.5.2] - 2026-05-15

### Added

- Curated vision-language benchmark source (`benchmark_sources/vision.py`):
  a 0-100 multimodal capability index (MMMU-Pro / MMBench / general
  multimodal, 2026-05) covering the Qwen3-VL / Qwen2.5-VL / Qwen2-VL /
  Llama-Vision / Phi-vision / Gemma-3 / Pixtral / InternVL3 lines.
- Benchmark snapshot date is now shown under every ranking so a stale
  recommendation is self-evident instead of silently trusted.
- Round 3 regression suite (`tests/test_r3_regressions.py`, 20 tests),
  each verified to fail when its fix is reverted.

### Fixed

- `--profile vision` generation inversion: text leaderboards do not
  score VLMs, so the only model with a direct hit was a
  two-generations-old Qwen2-VL-7B, which outranked the current
  Qwen3-VL-32B even on an 80 GB GPU. Vision models now score from the
  curated multimodal index (Qwen3-VL-32B leads at 73-76).
- Apple Silicon partial-offload speed was estimated ~3x too low: the
  flat 0.45x PCIe penalty was applied to unified memory, where spilled
  weights stay in the same high-bandwidth pool. DeepSeek-R1-class
  models on M2/M3 Ultra now report a realistic 4-15 t/s instead of
  ~1.7. Discrete GPUs keep the 0.45x penalty.
- Duplicate `Qwen/Qwen3-Coder-30B-A3B-Instruct` key in the LiveBench
  fallback (silently scored 62 instead of the intended 58, and broke
  CI lint via ruff F601).
- `ruff format` / `ruff check` are now clean across the codebase, so
  the Lint CI job passes (it was red for the entire 0.5.1 release).

### CI

- GitHub Actions updated to the Node 24 runtime (`checkout@v5`,
  `setup-python@v6`); the Node 20 actions are deprecated from 2026-06.

## [0.5.1] - 2026-05-14

### Added

- `whichllm upgrade` subcommand: side-by-side comparison of the current
  machine against potential GPU upgrades, with a verdict (worth it /
  meaningful / marginal / flat / downgrade).
- Apple Silicon support in `--gpu` flag (M1-M4 base / Pro / Max / Ultra)
  so simulator runs no longer fuzzy-match to ATI Rage Mobility-M1 and
  emit spurious AMD ROCm warnings.
- Curated LiveBench, Arena AA, and Aider benchmark source modules with
  frozen 2026-Q2 fallbacks for offline operation.
- Curated entries for reasoning / thinking lines: `Qwen/QwQ-32B`,
  `Qwen3-4B-Thinking-2507`, `DeepSeek-R1-Distill-Qwen-32B/14B` and
  `Llama-8B`.
- Frontier-model surfacing for 2026-Q2 releases that do not auto-surface
  via cardinality (Kimi-K2, MiMo, DeepSeek-V4, GLM-5, Qwen3.6/Next,
  gpt-oss, Llama-4, Mistral Small/Large, Devstral, Codestral, MiniMax,
  Granite 3.3/4.0, Olmo-3, Nemotron-3).
- VRAM-aware auto floor for `--profile general` so tiny GPUs surface
  full-GPU 3-4B picks instead of partial-offload-only 7B+.

### Changed

- VRAM estimation: KV cache scaled to 3.5 MB / billion-param / Kctx (was
  0.5 MB) so 128K contexts are realistic; MoE KV uses active*4 to model
  attention head sharing; activation overhead refined.
- Speed estimation: per-quant efficiency table, per-backend multiplier
  (CUDA 1.0, Apple 0.82, AMD 0.78, Intel 0.65), MoE active-ratio floor,
  partial-offload penalty.
- Ranking: composite family selection key replaces tier dominance;
  size_score cap 20 → 35; MoE size_score uses total params;
  `_knowledge_capacity_b` so `--min-params` no longer hides
  Qwen3-Next-80B-A3B on its 3B active.
- Benchmark merging splits frozen (OLLB v2, Arena ELO) from current
  (AA, LiveBench, Aider) with separate caps and lineage-aware recency
  demotion so stale 2024-era leaderboards stop over-rewarding older
  generations.
- httpx `AsyncClient` uses `follow_redirects=True` so case-mismatch HF
  URLs (307) no longer silently drop frontier IDs.

### Fixed

- Reject benchmark inheritance when actual params differ by more than
  2x from the family's dominant member, catching draft/MTP/abliterated
  forks that share a `family_id` with their much larger base
  (e.g. a 6.6B "imatrix-aligned" inheriting from a 158B base).
- Family grouping prefers the upstream-referenced model as the family
  base instead of the highest-downloads member, so a popular fork no
  longer overrides the official base for `family_id` assignment.
- MoE active-parameter registry corrected (gpt-oss-20b 3.6B,
  gpt-oss-120b 5.1B, MiniMax-M2 10B).
- Quality floor (≥ 20) and speed floor (≥ 1.5 t/s) drop junk Q1_0 /
  Bonsai-class attack vectors.
- 11 non-existent HF IDs removed from curated fallbacks (Kimi K2.5/K2.6,
  GLM-5-Turbo, OLMo-3-32B, Llama-3.2-8B, Codestral-25.08, Mistral-Large-3
  etc.).

## [0.4.0] - 2026-03-09

### Added

- `whichllm plan` subcommand — reverse lookup to find what GPU you need for a model
- Ollama integration examples and shell alias
- Homebrew formula for `brew install whichllm`
- VHS tape file for recording CLI demo GIF
- GitHub Actions CI/CD (tests, lint, PyPI publish)
- CONTRIBUTING.md, CODE_OF_CONDUCT.md
- Issue and PR templates
- PyPI metadata (classifiers, keywords, URLs)

## [0.3.0] - 2026-03-09

### Added

- Evidence filtering options (`--evidence`, `--direct`) in CLI and ranking logic
- A100/H100 80GB aliases to GPU simulator
- Eval benchmark integration with confidence-based score dampening
- BenchmarkEvidence with confidence-aware size interpolation
- HuggingFace evalResults as supplementary benchmark source

## [0.2.2]

### Added

- `--version` option to display package version

### Changed

- Updated demo image asset

## [0.2.1]

### Added

- Vision model support based on task profile (`--profile vision`)

## [0.2.0]

### Added

- `--status` flag to show Speed/Fit columns in output
- Published date and download count columns in display
- `published_at` backfill for ranking display
- GGUF-only backend filtering for model ranking
- Task profile support (`--profile`) for general, coding, vision, math
- GPU simulation (`--gpu`, `--vram`) for testing different hardware
- JSON output mode (`--json`)
- Rich table output with color-coded scores
- GPU detection for NVIDIA, AMD, and Apple Silicon
- HuggingFace API integration for model fetching
- Quantization-aware VRAM calculation
- Cache system with TTL (6h models, 24h benchmarks)

## [0.1.0]

### Added

- Initial release
- Basic hardware detection
- Simple model ranking with Typer CLI
