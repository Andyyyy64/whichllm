"""Which engines are installed here, and which already serve on their port.

Detection looks for an engine's binaries on PATH and at the places its
installer uses, and asks its default port whether that engine answers. Ports
are shared between engines (8080 and 8000 especially), so a port only counts
when the reply carries the engine's own fingerprint.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from whichllm.engines.cards import EngineCard

_PROBE_TIMEOUT_SECONDS = 0.5
_COMMAND_TIMEOUT_SECONDS = 5
# "0.40.14" or "v1.124.2"; llama.cpp prints its build as "b12069-<hash>" or
# "version: 12069 (<hash>)".
_DOTTED_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)*)\b")
_BUILD_RE = re.compile(r"\b(b\d{4,})\b|version:\s*(\d{4,})\b")


@dataclass(frozen=True)
class Installed:
    path: str | None  # the binary or app found, if any
    version: str | None
    running: bool  # a server with the engine's fingerprint answered


def _candidates(card: EngineCard, os_name: str) -> list[str]:
    paths = [name for name in card.detect_binaries]
    paths += [p for os_key, p in card.detect_paths if os_key in (os_name, "all")]
    return paths


def _find(candidate: str) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(candidate))
    if os.sep in expanded or "/" in expanded:
        return expanded if Path(expanded).exists() else None
    return shutil.which(expanded)


def _run(path: str, args: tuple[str, ...]) -> str:
    try:
        done = subprocess.run(
            [path, *args],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"{done.stdout}\n{done.stderr}"


def _version(text: str) -> str | None:
    build = _BUILD_RE.search(text)
    if build:
        return build.group(1) or f"b{build.group(2)}"
    dotted = _DOTTED_RE.search(text)
    return dotted.group(1) if dotted else None


def _probe(card: EngineCard, client: httpx.Client) -> str | None:
    """The reply body when the engine answers on its default port, else None."""
    if card.port is None or card.probe_path is None:
        return None
    try:
        reply = client.get(f"http://127.0.0.1:{card.port}{card.probe_path}")
    except httpx.HTTPError:
        return None
    if reply.status_code != 200 or card.probe_contains not in reply.text:
        return None
    return reply.text


def detect_installed(
    cards: tuple[EngineCard, ...],
    os_name: str,
    *,
    probe: bool = True,
    client: httpx.Client | None = None,
) -> dict[str, Installed]:
    """Installed engines by card id; engines not found are left out.

    ``client`` replaces the one that asks each engine's port, for tests.
    """
    found: dict[str, Installed] = {}
    help_cache: dict[str, str] = {}
    # Engines that share a binary name (llama.cpp and ik_llama.cpp both ship
    # llama-server) are told apart by a marker in --help.
    markers = {
        name: card.help_marker
        for card in cards
        if card.help_marker
        for name in card.detect_binaries
    }
    owned = client is None and probe
    if owned:
        client = httpx.Client(timeout=_PROBE_TIMEOUT_SECONDS)
    if not probe:
        client = None
    try:
        for card in cards:
            path = None
            for candidate in _candidates(card, os_name):
                hit = _find(candidate)
                if hit is None:
                    continue
                name = Path(candidate).name
                if name in markers:
                    if hit not in help_cache:
                        help_cache[hit] = _run(hit, ("--help",))
                    owns = card.help_marker is not None
                    if (markers[name] in help_cache[hit]) != owns:
                        continue
                path = hit
                break
            reply = _probe(card, client) if client else None
            if path is None and reply is None:
                continue
            version = None
            if path and card.version_args:
                version = _version(_run(path, card.version_args))
            if version is None and reply:
                version = _version(reply)
            found[card.id] = Installed(
                path=path, version=version, running=reply is not None
            )
    finally:
        if owned:
            client.close()
    return found


__all__ = ["Installed", "detect_installed"]
