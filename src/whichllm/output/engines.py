"""The `whichllm engines` table and its JSON."""

from __future__ import annotations

import json

from rich.markup import escape
from rich.table import Table

from whichllm.engines import Availability
from whichllm.engines.cards import EngineCard
from whichllm.engines.detect import Installed
from whichllm.hardware.types import HardwareInfo
from whichllm.output import _console
from whichllm.output.json_output import hardware_json

_ACCEL_NAMES = {
    "cuda": "CUDA",
    "rocm": "ROCm",
    "metal": "Metal",
    "sycl": "SYCL",
    "vulkan": "Vulkan",
    "cpu": "CPU",
}
_EVIDENCE_STYLES = {"verified": "green", "community": "yellow", "documented": "dim"}
_OS_NAMES = {"linux": "Linux", "windows": "Windows", "darwin": "macOS"}


def _machine(hardware: HardwareInfo) -> str:
    if not hardware.gpus:
        return f"{_OS_NAMES[hardware.os]}, CPU only"
    gpu = max(hardware.gpus, key=lambda g: g.vram_bytes)
    capability = (
        f" (CC {gpu.compute_capability[0]}.{gpu.compute_capability[1]})"
        if gpu.compute_capability
        else ""
    )
    return f"{_OS_NAMES[hardware.os]}, {gpu.name}{capability}"


def _upstream(card: EngineCard) -> str:
    version = card.version or "rolling"
    return f"{version}, {card.released_at}" if card.released_at else version


def _installed(found: Installed | None) -> str:
    if found is None:
        return "[dim]-[/dim]"
    parts = [found.version] if found.version else []
    if found.running:
        parts.append("[green]running[/]")
    return ", ".join(["yes", *parts])


def display_engines(
    rows: list[Availability],
    hardware: HardwareInfo,
    installed: dict[str, Installed] | None = None,
    *,
    wsl: bool = False,
) -> None:
    table = Table(title=f"Inference engines for {escape(_machine(hardware))}")
    table.add_column("Engine", style="cyan", no_wrap=True)
    table.add_column("Runs here", min_width=10)
    if installed is not None:
        table.add_column("Installed", no_wrap=True)
    table.add_column("Evidence", no_wrap=True)
    table.add_column("Upstream", no_wrap=True)
    for row in rows:
        if row.runs:
            runs = f"[green]yes[/], {_ACCEL_NAMES[row.platform.accel]}"
        else:
            runs = "[red]no[/]"
        if row.reason:
            runs += f"\n[dim]{escape(row.reason)}[/dim]"
        style = _EVIDENCE_STYLES[row.card.evidence]
        cells = [row.card.name, runs]
        if installed is not None:
            cells.append(_installed(installed.get(row.card.id)))
        cells += [f"[{style}]{row.card.evidence}[/{style}]", _upstream(row.card)]
        table.add_row(*cells)
    _console.console.print(table)
    _console.console.print(
        "  [dim]Evidence:[/dim]  verified = whichllm bench runs,  "
        "community = public benchmarks,  documented = docs only"
    )
    if wsl:
        _console.console.print(
            "  [dim]WSL: engines installed on the Windows side are not looked for here."
            "[/dim]"
        )


def engine_json(row: Availability, found: Installed | None = None) -> dict:
    card = row.card
    return {
        "id": card.id,
        "name": card.name,
        "kind": card.kind,
        "runs": row.runs,
        "accel": row.platform.accel if row.platform else None,
        "reason": row.reason,
        "installed": found is not None,
        "installed_path": found.path if found else None,
        "installed_version": found.version if found else None,
        "running": found.running if found else False,
        "evidence": card.evidence,
        "status": card.status,
        "version": card.version,
        "released_at": card.released_at,
        "checked_at": card.checked_at,
        "url": card.url,
        "serve": card.serve,
        "port": card.port,
        "api": list(card.api),
        "install": dict(card.install),
        "sources": list(card.sources),
        "note": card.note,
    }


def display_engines_json(
    rows: list[Availability],
    hardware: HardwareInfo,
    installed: dict[str, Installed] | None = None,
) -> None:
    found = installed or {}
    output = {
        "hardware": hardware_json(hardware),
        "engines": [engine_json(row, found.get(row.card.id)) for row in rows],
    }
    _console.console.print_json(json.dumps(output, ensure_ascii=False))


__all__ = ["display_engines", "display_engines_json", "engine_json"]
