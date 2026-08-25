"""Read-only discovery of GGUF models installed by LM Studio."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from whichllm.engine.types import CompatibilityResult
from whichllm.models.gguf import _extract_quant_type

_SKIPPED_DIRS = frozenset({".cache", ".internal", "blobs", "manifests"})


@dataclass(frozen=True)
class LocalGGUF:
    """A GGUF file found in an LM Studio model library."""

    path: Path
    repo_id: str | None
    quant_type: str | None


def default_lmstudio_paths(home: Path | None = None) -> tuple[Path, ...]:
    """Return current and legacy LM Studio model-library locations."""
    home = home or Path.home()
    return (
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
    )


def _repo_id_from_path(path: Path, root: Path) -> str | None:
    """Derive the Hugging Face repo ID from LM Studio's owner/repo layout."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) < 3:
        return None
    owner, repo = relative.parts[:2]
    if owner.startswith(".") or repo.startswith("."):
        return None
    return f"{owner}/{repo}"


def _iter_gguf_files(root: Path) -> Iterable[Path]:
    """Yield model GGUFs without following symlinked directories."""
    try:
        if not root.is_dir():
            return
    except OSError:
        return

    def ignore_error(_error: OSError) -> None:
        return None

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=ignore_error, followlinks=False
    ):
        dirnames[:] = [
            name
            for name in dirnames
            if name.casefold() not in _SKIPPED_DIRS
            and not (Path(directory) / name).is_symlink()
        ]
        for filename in filenames:
            lower = filename.casefold()
            if (
                not lower.endswith(".gguf")
                or filename.startswith(".")
                or lower.startswith("mmproj-")
            ):
                continue
            yield Path(directory) / filename


def discover_lmstudio_ggufs(
    custom_paths: Sequence[Path] = (),
    *,
    home: Path | None = None,
) -> list[LocalGGUF]:
    """Scan default and custom LM Studio libraries for local GGUF files."""
    roots = (*default_lmstudio_paths(home), *(Path(path) for path in custom_paths))
    seen_paths: set[str] = set()
    discovered: list[LocalGGUF] = []

    for root in roots:
        root = root.expanduser().absolute()
        for path in _iter_gguf_files(root):
            key = os.path.normcase(os.path.abspath(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            quant_type = _extract_quant_type(path.name)
            discovered.append(
                LocalGGUF(
                    path=path,
                    repo_id=_repo_id_from_path(path, root),
                    quant_type=quant_type if quant_type != "unknown" else None,
                )
            )

    return sorted(discovered, key=lambda item: os.path.normcase(str(item.path)))


def _find_local_match(
    result: CompatibilityResult, local_models: Sequence[LocalGGUF]
) -> LocalGGUF | None:
    model = result.artifact_model or result.model
    variant = result.artifact_variant or result.gguf_variant
    if variant is None:
        return None

    repo_id = model.id.casefold()
    filename = Path(variant.filename).name.casefold()
    quant_type = variant.quant_type.casefold()

    exact_repo = [
        local
        for local in local_models
        if local.repo_id is not None and local.repo_id.casefold() == repo_id
    ]
    for local in exact_repo:
        if local.path.name.casefold() == filename:
            return local
    for local in exact_repo:
        if local.quant_type is not None and local.quant_type.casefold() == quant_type:
            return local

    # A custom path may point directly at a repo directory, so it has no
    # owner/repo layout. In that case only an exact filename is safe enough.
    for local in local_models:
        if local.repo_id is None and local.path.name.casefold() == filename:
            return local
    return None


def attach_local_matches(
    results: list[CompatibilityResult], local_models: Sequence[LocalGGUF]
) -> None:
    """Annotate recommendations that have an exact local GGUF match."""
    for result in results:
        match = _find_local_match(result, local_models)
        result.local_path = str(match.path) if match else None
