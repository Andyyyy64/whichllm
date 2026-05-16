# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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
