"""Read-only discovery of GGUF models installed by LM Studio."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from whichllm.engine.types import CompatibilityResult
from whichllm.models.gguf import _GGUF_SPLIT_RE

_SKIPPED_DIRS = frozenset({".cache", ".internal", "blobs", "manifests"})


@dataclass(frozen=True)
class LocalGGUF:
    """A GGUF file found in an LM Studio model library."""

    path: Path
    repo_id: str | None
    artifact_path: str | None


class LMStudioPathError(ValueError):
    """An explicitly configured LM Studio library cannot be scanned."""


def default_lmstudio_paths(home: Path | None = None) -> tuple[Path, ...]:
    """Return the current LM Studio model-library location."""
    home = home or Path.home()
    return (home / ".lmstudio" / "models",)


def _repo_artifact_from_path(path: Path, root: Path) -> tuple[str | None, str | None]:
    """Derive the Hugging Face repo and artifact paths from an LM Studio path."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None, None
    if len(relative.parts) < 3:
        return None, None
    owner, repo = relative.parts[:2]
    if owner.startswith(".") or repo.startswith("."):
        return None, None
    return f"{owner}/{repo}", PurePosixPath(*relative.parts[2:]).as_posix()


def validate_lmstudio_paths(paths: Sequence[Path]) -> None:
    """Fail clearly when an explicitly configured library cannot be read."""
    for path in paths:
        root = Path(path).expanduser().absolute()
        try:
            root.stat()
            if not root.is_dir():
                raise LMStudioPathError(f"LM Studio path is not a directory: {root}")
            with os.scandir(root):
                pass
        except FileNotFoundError as error:
            raise LMStudioPathError(f"LM Studio path does not exist: {root}") from error
        except LMStudioPathError:
            raise
        except OSError as error:
            raise LMStudioPathError(
                f"LM Studio path cannot be read: {root}: {error}"
            ) from error


def _is_readable_file(path: Path) -> bool:
    """Return whether a GGUF path resolves to a readable regular file."""
    try:
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def _iter_gguf_files(root: Path, *, strict: bool = False) -> Iterable[Path]:
    """Yield model GGUFs without following symlinked directories."""
    try:
        if not root.is_dir():
            return
    except OSError as error:
        if strict:
            raise LMStudioPathError(
                f"LM Studio path cannot be read: {root}: {error}"
            ) from error
        return

    def handle_error(error: OSError) -> None:
        if strict:
            target = error.filename or str(root)
            raise LMStudioPathError(
                f"LM Studio path cannot be read: {target}: {error}"
            ) from error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=handle_error, followlinks=False
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
            path = Path(directory) / filename
            if _is_readable_file(path):
                yield path


def discover_lmstudio_ggufs(
    custom_paths: Sequence[Path] = (),
    *,
    home: Path | None = None,
) -> list[LocalGGUF]:
    """Scan default and custom LM Studio libraries for local GGUF files."""
    custom_roots = tuple(Path(path) for path in custom_paths)
    validate_lmstudio_paths(custom_roots)
    roots = (
        *((path, False) for path in default_lmstudio_paths(home)),
        *((path, True) for path in custom_roots),
    )
    seen_paths: set[str] = set()
    discovered: list[LocalGGUF] = []

    for root, strict in roots:
        root = root.expanduser().absolute()
        for path in _iter_gguf_files(root, strict=strict):
            key = os.path.normcase(os.path.abspath(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            repo_id, artifact_path = _repo_artifact_from_path(path, root)
            discovered.append(
                LocalGGUF(
                    path=path,
                    repo_id=repo_id,
                    artifact_path=artifact_path,
                )
            )

    return sorted(discovered, key=lambda item: os.path.normcase(str(item.path)))


def _has_all_split_parts(
    repo_id: str,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> bool:
    artifact = PurePosixPath(artifact_path)
    match = _GGUF_SPLIT_RE.search(artifact.name)
    if match is None:
        return True

    total = int(match.group(2))
    if total < 1:
        return False
    prefix = artifact.name[: match.start()]
    expected = {
        artifact.with_name(f"{prefix}-{part:05d}-of-{total:05d}.gguf")
        .as_posix()
        .casefold()
        for part in range(1, total + 1)
    }
    available = {
        local.artifact_path.casefold()
        for local in local_models
        if local.repo_id is not None
        and local.repo_id.casefold() == repo_id.casefold()
        and local.artifact_path is not None
        and _is_readable_file(local.path)
    }
    return expected <= available


def _find_local_match(
    result: CompatibilityResult, local_models: Sequence[LocalGGUF]
) -> LocalGGUF | None:
    model = result.artifact_model
    variant = result.artifact_variant
    if model is None or variant is None:
        return None

    repo_id = model.id.casefold()
    artifact_path = PurePosixPath(variant.filename).as_posix().casefold()
    for local in local_models:
        if (
            local.repo_id is not None
            and local.repo_id.casefold() == repo_id
            and local.artifact_path is not None
            and local.artifact_path.casefold() == artifact_path
            and _is_readable_file(local.path)
            and _has_all_split_parts(model.id, variant.filename, local_models)
        ):
            return local
    return None


def attach_local_matches(
    results: list[CompatibilityResult], local_models: Sequence[LocalGGUF]
) -> None:
    """Annotate recommendations that have an exact local GGUF match."""
    for result in results:
        match = _find_local_match(result, local_models)
        result.local_path = str(match.path) if match else None
