# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `whichllm plan` subcommand — reverse lookup to find what GPU you need for a model
- Homebrew formula for `brew install whichllm`
- VHS tape file for recording CLI demo GIF

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
