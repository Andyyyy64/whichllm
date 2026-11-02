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


def _split_parts(name: str) -> tuple[str, int] | None:
    """Return the filename prefix and part count of a split GGUF name."""
    match = _GGUF_SPLIT_RE.search(name)
    if match is None:
        return None
    return name[: match.start()], int(match.group(2))


def _split_entry_filename(name: str) -> str | None:
    """Return the first part filename of a split GGUF name, if it is one."""
    parts = _split_parts(name)
    if parts is None:
        return None
    prefix, total = parts
    if total < 1:
        return None
    return f"{prefix}-00001-of-{total:05d}.gguf"


def _missing_split_parts(
    local: LocalGGUF,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> tuple[list[int], int]:
    """Return the split parts missing beside this candidate's entry file.

    llama.cpp resolves the remaining parts relative to the file it opens, so
    parts that only exist in another library directory cannot complete the set.
    Only files in the candidate's own directory are counted. A plain artifact
    returns no missing parts and a total of zero.
    """
    artifact = PurePosixPath(artifact_path)
    parts = _split_parts(artifact.name)
    if parts is None:
        return [], 0

    prefix, total = parts
    if total < 1:
        return [1], 1
    directory = os.path.normcase(os.path.abspath(local.path.parent))
    available = {
        PurePosixPath(candidate.artifact_path).name
        for candidate in local_models
        if candidate.artifact_path is not None
        and os.path.normcase(os.path.abspath(candidate.path.parent)) == directory
        and _is_readable_file(candidate.path)
    }
    expected = {
        part: f"{prefix}-{part:05d}-of-{total:05d}.gguf" for part in range(1, total + 1)
    }
    missing = sorted(part for part, name in expected.items() if name not in available)
    return missing, total


def _has_all_split_parts(
    local: LocalGGUF,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> bool:
    """Return whether every split part sits beside this candidate's entry file."""
    if _split_parts(PurePosixPath(artifact_path).name) is None:
        return True
    missing, total = _missing_split_parts(local, artifact_path, local_models)
    return total >= 1 and not missing


def _expected_parts(artifact_path: str) -> dict[int, str] | None:
    """Part number and filename of every part, or None for a plain artifact."""
    parts = _split_parts(PurePosixPath(artifact_path).name)
    if parts is None:
        return None
    prefix, total = parts
    if total < 1:
        return None
    return {
        part: f"{prefix}-{part:05d}-of-{total:05d}.gguf" for part in range(1, total + 1)
    }


def _is_readable_entry(local: LocalGGUF, repo_id: str, artifact_path: str) -> bool:
    """Whether this file is the readable entry of the requested artifact.

    The one predicate the lookup, the re-check and the incomplete-set report
    share, so they cannot drift apart.
    """
    return (
        local.repo_id is not None
        and local.repo_id.casefold() == repo_id.casefold()
        and local.artifact_path is not None
        and local.artifact_path == PurePosixPath(artifact_path).as_posix()
        and _is_readable_file(local.path)
    )


def find_local_artifact(
    repo_id: str,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> LocalGGUF | None:
    """Return the readable local file that exactly matches a repository artifact."""
    for local in local_models:
        if _is_readable_entry(local, repo_id, artifact_path) and _has_all_split_parts(
            local, artifact_path, local_models
        ):
            return local
    return None


def _find_partial_entry_set(
    repo_id: str,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> tuple[LocalGGUF, list[int], int] | None:
    """Report a split set whose entry part is gone but other parts remain.

    The entry file is what a backend opens, so a library that only kept the
    later parts cannot be used either, and the download it falls back to
    deserves the same warning.
    """
    expected = _expected_parts(artifact_path)
    if expected is None:
        return None
    readable = {name for part, name in expected.items()}
    for local in local_models:
        if (
            local.repo_id is None
            or local.repo_id.casefold() != repo_id.casefold()
            or local.artifact_path is None
            or PurePosixPath(local.artifact_path).name not in readable
            or not _is_readable_file(local.path)
        ):
            continue
        missing, total = _missing_split_parts(local, artifact_path, local_models)
        if missing:
            return local, missing, total
    return None


def find_incomplete_local_artifact(
    repo_id: str,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> tuple[LocalGGUF, list[int], int] | None:
    """Return a local match that cannot load, with its missing part numbers.

    ``find_local_artifact`` skips such a candidate so the scan can keep looking
    in other libraries. This reports the first one it skipped, so the caller can
    say why the artifact is downloaded anyway. It covers both an entry file that
    lost later parts and a set whose entry file is gone.
    """
    for local in local_models:
        if not _is_readable_entry(local, repo_id, artifact_path):
            continue
        missing, total = _missing_split_parts(local, artifact_path, local_models)
        if missing:
            return local, missing, total
    return _find_partial_entry_set(repo_id, artifact_path, local_models)


def artifact_part_paths(entry: Path, artifact_path: str) -> tuple[Path, ...]:
    """Every file llama.cpp opens for this artifact, the entry part first."""
    expected = _expected_parts(artifact_path)
    if expected is None or len(expected) < 2:
        return (entry,)
    return tuple(entry.with_name(expected[part]) for part in sorted(expected))


def verify_local_artifact(
    local: LocalGGUF,
    repo_id: str,
    artifact_path: str,
    local_models: Sequence[LocalGGUF],
) -> Path | None:
    """Re-check a match before launch and return the GGUF path to load.

    A match can go stale between discovery and launch, so the same predicate as
    the lookup is applied again. The returned path is the entry part a backend
    expects to open.
    """
    if not _is_readable_entry(local, repo_id, artifact_path):
        return None
    if not _has_all_split_parts(local, artifact_path, local_models):
        return None

    entry = _split_entry_filename(PurePosixPath(artifact_path).name)
    if entry is None:
        return local.path
    entry_path = local.path.with_name(entry)
    return entry_path if _is_readable_file(entry_path) else None


def _find_local_match(
    result: CompatibilityResult, local_models: Sequence[LocalGGUF]
) -> LocalGGUF | None:
    model = result.artifact_model
    variant = result.artifact_variant
    if model is None or variant is None:
        return None

    return find_local_artifact(model.id, variant.filename, local_models)


def attach_local_matches(
    results: list[CompatibilityResult], local_models: Sequence[LocalGGUF]
) -> None:
    """Annotate recommendations that have an exact local GGUF match."""
    for result in results:
        match = _find_local_match(result, local_models)
        result.local_path = str(match.path) if match else None
