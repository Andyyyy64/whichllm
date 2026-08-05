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
To add a new GPU to the bandwidth database, edit `src/whichllm/constants.py` and add the GPU specs.

## Code Style
- Follow existing code conventions
- Use type hints
- Add tests for new functionality

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
