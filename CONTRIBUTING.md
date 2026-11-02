# Contributing to whichllm

Thanks for your interest in contributing! Here's how you can help.

## Development Setup

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --locked --dev
```

## Running Checks

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

## How to Contribute

### Bug Reports
Open an issue with:
- Your hardware (GPU model, VRAM, OS)
- Python version
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Open an issue describing the feature and why it would be useful.

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the checks above
5. Submit a PR

### AI-assisted Contributions
AI-assisted code is welcome. Use the tools that help you move faster. Working
code wins.

The bar is practical: make it work, make it fit the project, and make it
reviewable. If you wrote it or asked a tool to write it, you own it. Read it,
test the parts that matter, and be ready to explain or fix it.

### Adding GPU Support
To add a new GPU to the bandwidth database, edit `src/whichllm/data/gpu.py` and add the GPU specs.

### Ranking Changes
`tests/test_golden_rankings.py` ranks a frozen model catalog for fixed hardware
archetypes (`tests/golden/archetypes.toml`) and checks each result against
constraints. If your change moves recommendations, regenerate the snapshot and
include the diff of `tests/golden/top_picks.md` in your PR:

```bash
uv run --locked python scripts/update_golden.py snapshot
```

`report` prints every constraint result without writing files. When a fix makes
a known failure pass, the strict xfail turns into a failure; remove its `xfail`
entry from `archetypes.toml`.

The `measured-*` constraints hold the estimates to real runs. If a change moves
a speed or memory estimate past those bounds, measure again instead of widening
them. `tests/test_golden_checks.py` fails when a constraint's pattern no longer
matches anything in the catalog, so a stale pattern cannot pass by accident.

### Measurements
`whichllm bench` and `whichllm report` produce issues with the `measurement`
label. A maintainer reads each one, and if it looks real, adds
`measurement-verified`. Then:

```bash
uv run --locked python scripts/collect_measurements.py
```

That rewrites `src/whichllm/data/measurements.json`. Read the diff and open a PR
by hand. The script treats issue text as untrusted: it only parses JSON and checks
each field, and it prints every issue it left out and why.

## Code Style
- Follow existing code conventions
- Use type hints
- Add tests for new functionality

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
