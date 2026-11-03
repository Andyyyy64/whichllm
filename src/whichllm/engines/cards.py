"""Engine cards: where each inference engine runs, what it reads and how it starts.

One JSON file per engine lives in ``data/engines``. Every value comes from the
engine's own repository or docs, listed under ``sources`` with the date it was
checked, so a card can be re-checked against its sources rather than trusted
from memory. A value the sources do not give stays out of the card.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from whichllm.data import DATA_DIR

ENGINE_DIR = DATA_DIR / "engines"

# "runner" reads any architecture its parser knows; "fixed_model" runs a short
# list of models from its own packs; "distributed" spans several machines.
KINDS = frozenset({"runner", "fixed_model", "distributed"})
# How much stands behind the card's speed: whichllm's own verified runs, public
# benchmarks with a source, or the engine's docs alone. Strongest first.
EVIDENCE = ("verified", "community", "documented")
STATUSES = frozenset({"active", "slowing", "dormant", "archived"})
OSES = frozenset({"linux", "windows", "darwin"})
ACCELS = frozenset({"cuda", "rocm", "metal", "sycl", "vulkan", "cpu"})
MEMORY = frozenset({"gpu", "layer_offload", "expert_offload", "cpu"})

# Unknown keys are refused, so a typo cannot drop a value without a word.
_CARD_KEYS = frozenset(
    {
        "id", "name", "kind", "url", "evidence", "upstream", "checked_at",
        "sources", "platforms", "gguf", "memory", "serve", "port", "api",
        "install", "detect_binaries", "detect_paths", "probe", "version_args",
        "help_marker", "note",
    }
)  # fmt: skip
_UPSTREAM_KEYS = frozenset({"github", "version", "released_at", "status"})
_PLATFORM_KEYS = frozenset(
    {"os", "accel", "min_compute_capability", "requires_avx2", "note"}
)
_GGUF_KEYS = frozenset({"extra", "missing"})
_PROBE_KEYS = frozenset({"path", "contains"})

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CC_RE = re.compile(r"(\d+)\.(\d+)")


class CardError(ValueError):
    """A card that does not follow the schema."""


@dataclass(frozen=True)
class Platform:
    """One way the engine runs: an OS set and an accelerator."""

    os: tuple[str, ...]
    accel: str
    min_compute_capability: tuple[int, int] | None = None
    requires_avx2: bool = False
    note: str | None = None


@dataclass(frozen=True)
class GgufSupport:
    """GGUF quant types beyond, or missing from, mainline llama.cpp."""

    extra: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class EngineCard:
    id: str
    name: str
    kind: str
    url: str
    evidence: str
    status: str
    version: str | None
    released_at: str | None
    checked_at: str
    sources: tuple[str, ...]
    platforms: tuple[Platform, ...]
    gguf: GgufSupport | None
    memory: tuple[str, ...]
    serve: str | None
    port: int | None
    api: tuple[str, ...]
    install: tuple[tuple[str, str], ...]
    detect_binaries: tuple[str, ...]
    # (os or "all", path) pairs where an installer puts the engine.
    detect_paths: tuple[tuple[str, str], ...] = ()
    # GET this path on the default port; the reply must contain the text.
    probe_path: str | None = None
    probe_contains: str | None = None
    # Arguments that print the version, for binaries where that is safe.
    version_args: tuple[str, ...] = ()
    # Text only this engine's --help prints, for a binary name it shares.
    help_marker: str | None = None
    note: str | None = None


def _fail(path: Path, message: str) -> CardError:
    return CardError(f"{path.name}: {message}")


def _known_keys(raw: dict, allowed: frozenset[str], where: str, path: Path) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(path, f"unknown key(s) in {where}: {', '.join(unknown)}")


def _text(data: dict, key: str, path: Path, *, optional: bool = False) -> str | None:
    value = data.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, f"{key!r} must be a non-empty string")
    return value


def _choice(data: dict, key: str, allowed, path: Path) -> str:
    value = _text(data, key, path)
    if value not in allowed:
        raise _fail(path, f"{key!r} is {value!r}, not one of {sorted(allowed)}")
    return value


def _strings(data: dict, key: str, path: Path) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise _fail(path, f"{key!r} must be a list of strings")
    return tuple(value)


def _date(data: dict, key: str, path: Path, *, optional: bool = False) -> str | None:
    value = _text(data, key, path, optional=optional)
    if value is not None and not _DATE_RE.fullmatch(value):
        raise _fail(path, f"{key!r} must be YYYY-MM-DD, not {value!r}")
    return value


def _platform(raw: dict, path: Path) -> Platform:
    if not isinstance(raw, dict):
        raise _fail(path, "each platform must be an object")
    _known_keys(raw, _PLATFORM_KEYS, "a platform", path)
    oses = _strings(raw, "os", path)
    if not oses or not set(oses) <= OSES:
        raise _fail(path, f"platform os {oses} must be a subset of {sorted(OSES)}")
    accel = _choice(raw, "accel", ACCELS, path)
    capability = None
    if raw.get("min_compute_capability") is not None:
        match = _CC_RE.fullmatch(str(raw["min_compute_capability"]))
        if accel != "cuda" or match is None:
            raise _fail(path, "min_compute_capability is a CUDA 'major.minor'")
        capability = (int(match.group(1)), int(match.group(2)))
    return Platform(
        os=oses,
        accel=accel,
        min_compute_capability=capability,
        requires_avx2=bool(raw.get("requires_avx2", False)),
        note=_text(raw, "note", path, optional=True),
    )


def _gguf(raw, path: Path) -> GgufSupport | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _fail(path, "'gguf' must be an object")
    _known_keys(raw, _GGUF_KEYS, "'gguf'", path)
    return GgufSupport(
        extra=_strings(raw, "extra", path), missing=_strings(raw, "missing", path)
    )


def _detect_paths(raw, path: Path) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise _fail(path, "'detect_paths' maps an OS or 'all' to a list of paths")
    pairs = []
    for os_key, paths in raw.items():
        if os_key not in OSES | {"all"}:
            raise _fail(path, f"'detect_paths' key {os_key!r} is not an OS or 'all'")
        for entry in _strings(raw, os_key, path):
            pairs.append((os_key, entry))
    return tuple(pairs)


def _probe(raw, path: Path) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise _fail(path, "'probe' must be an object")
    _known_keys(raw, _PROBE_KEYS, "'probe'", path)
    probe_path = _text(raw, "path", path)
    if not probe_path.startswith("/"):
        raise _fail(path, "'probe' path starts with '/'")
    return probe_path, _text(raw, "contains", path)


def parse_card(data: dict, path: Path) -> EngineCard:
    """Validate one card. Raises CardError naming the file and the field."""
    if not isinstance(data, dict):
        raise _fail(path, "a card is a JSON object")
    _known_keys(data, _CARD_KEYS, "the card", path)
    upstream = data.get("upstream")
    if not isinstance(upstream, dict):
        raise _fail(path, "'upstream' must be an object")
    _known_keys(upstream, _UPSTREAM_KEYS, "'upstream'", path)
    sources = _strings(data, "sources", path)
    if not sources or not all(s.startswith("https://") for s in sources):
        raise _fail(path, "'sources' needs at least one https URL")
    platforms = data.get("platforms")
    if not isinstance(platforms, list) or not platforms:
        raise _fail(path, "'platforms' needs at least one entry")
    memory = _strings(data, "memory", path)
    if not set(memory) <= MEMORY:
        raise _fail(path, f"'memory' {memory} must be a subset of {sorted(MEMORY)}")
    install = data.get("install", {})
    if not isinstance(install, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in install.items()
    ):
        raise _fail(path, "'install' maps a platform or 'docs' to a string")
    port = data.get("port")
    if port is not None and not (isinstance(port, int) and 0 < port < 65536):
        raise _fail(path, f"'port' {port!r} is not a TCP port")
    probe_path, probe_contains = _probe(data.get("probe"), path)
    if probe_path is not None and port is None:
        raise _fail(path, "'probe' needs the card's 'port'")
    card = EngineCard(
        id=_text(data, "id", path),
        name=_text(data, "name", path),
        kind=_choice(data, "kind", KINDS, path),
        url=_text(data, "url", path),
        evidence=_choice(data, "evidence", EVIDENCE, path),
        status=_choice(upstream, "status", STATUSES, path),
        version=_text(upstream, "version", path, optional=True),
        released_at=_date(upstream, "released_at", path, optional=True),
        checked_at=_date(data, "checked_at", path),
        sources=sources,
        platforms=tuple(_platform(p, path) for p in platforms),
        gguf=_gguf(data.get("gguf"), path),
        memory=memory,
        serve=_text(data, "serve", path, optional=True),
        port=port,
        api=_strings(data, "api", path),
        install=tuple(sorted(install.items())),
        detect_binaries=_strings(data, "detect_binaries", path),
        detect_paths=_detect_paths(data.get("detect_paths"), path),
        probe_path=probe_path,
        probe_contains=probe_contains,
        version_args=_strings(data, "version_args", path),
        help_marker=_text(data, "help_marker", path, optional=True),
        note=_text(data, "note", path, optional=True),
    )
    if path.stem != card.id:
        raise _fail(path, f"the file is named after the id, {card.id!r}")
    return card


@lru_cache(maxsize=1)
def load_cards() -> tuple[EngineCard, ...]:
    """Every engine card, sorted by id."""
    cards = []
    for path in sorted(ENGINE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise _fail(path, f"not valid JSON: {error}") from error
        cards.append(parse_card(data, path))
    return tuple(cards)


def card_by_id(engine_id: str) -> EngineCard | None:
    return next((card for card in load_cards() if card.id == engine_id), None)


def install_hint(card: EngineCard, os_name: str) -> str | None:
    """The install command for this OS, else the install docs."""
    commands = dict(card.install)
    if os_name in commands:
        return f"install with {commands[os_name]}"
    if "docs" in commands:
        return f"install from {commands['docs']}"
    return None


__all__ = [
    "ACCELS",
    "CardError",
    "EVIDENCE",
    "EngineCard",
    "GgufSupport",
    "Platform",
    "card_by_id",
    "install_hint",
    "load_cards",
    "parse_card",
]
