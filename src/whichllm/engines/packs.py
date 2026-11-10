"""Whether a fixed-model engine's packs fit this machine.

A fixed-model engine splits one large MoE model over three tiers. The GPU
holds attention and as many experts as it has room for, RAM holds the other
experts, and the SSD holds what neither does. A pack either fits in memory,
runs while reading some experts from the SSD (much slower), or does not run
here. The numbers come from the engine's docs, on its card.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass

from whichllm.engines.availability import Availability, availability
from whichllm.engines.cards import EngineCard, FixedModel, Pack, Platform, Requires
from whichllm.hardware.types import GPUInfo, HardwareInfo

_GiB = 1024**3
# Cards give sizes in GB the way the engines' docs do, and the docs name the
# size a PC is sold as. A PC reports a little less: a "64 GB" one has about
# 62.7 GiB of RAM, a "32 GB" one 31.2, a "16 GB" one 15.5, and an RTX 5070
# "12 GB" has 11.9 GiB of VRAM. An amount within 4% of the size a pack names
# counts as that size.
_NOMINAL = 0.96


@dataclass(frozen=True)
class PackFit:
    tier: str | None  # "memory", "ssd", or None when the pack does not run
    reason: str | None = None  # why not, or what reading from the SSD costs

    @property
    def runs(self) -> bool:
        return self.tier is not None


@dataclass(frozen=True)
class PackOption:
    """One pack of one engine, and how it does here."""

    card: EngineCard
    model: FixedModel
    pack: Pack
    platform: Platform | None
    fit: PackFit


def _unified(gpu: GPUInfo) -> bool:
    return gpu.vendor == "apple" or gpu.shared_memory


def _ram_bytes(hardware: HardwareInfo) -> int:
    if hardware.ram_budget_bytes is not None:
        return min(hardware.ram_bytes, hardware.ram_budget_bytes)
    return hardware.ram_bytes


def _gb(value: float) -> str:
    return f"{value:g} GB"


def _have(value_bytes: int) -> str:
    return f"{value_bytes / _GiB:.0f} GB"


def _holds(have_bytes: int, need_gb: float) -> bool:
    return have_bytes >= need_gb * _GiB * _NOMINAL


def _short_of(need: Requires, ram: int) -> str:
    """What memory the pack is missing, for a machine where it does not run."""
    if need.ssd_ram_gb is not None:
        if not _holds(ram, need.ssd_ram_gb):
            return f"needs {_gb(need.ssd_ram_gb)} of RAM, this machine has {_have(ram)}"
        # Enough RAM for the SSD tier, but not the GPU it was documented with.
        if need.ram_gb is None and need.memory_gb is None:
            return (
                f"needs {_gb(need.ssd_ram_gb)} of RAM with a "
                f"{_gb(need.ssd_vram_gb or 0)} GPU"
            )
        if need.ram_gb is None:
            return (
                f"needs {_gb(need.memory_gb or 0)} of RAM and VRAM together, or "
                f"{_gb(need.ssd_ram_gb)} of RAM with a "
                f"{_gb(need.ssd_vram_gb or 0)} GPU"
            )
        return (
            f"needs {_gb(need.ram_gb)} of RAM, or "
            f"{_gb(need.ssd_ram_gb)} with a {_gb(need.ssd_vram_gb or 0)} GPU"
        )
    if need.ram_gb is not None and not _holds(ram, need.ram_gb):
        return f"needs {_gb(need.ram_gb)} of RAM, this machine has {_have(ram)}"
    return f"needs {_gb(need.memory_gb or 0)} of RAM and VRAM together"


def pack_fit(pack: Pack, hardware: HardwareInfo) -> PackFit:
    """Which tier the pack runs from here, or why it does not run."""
    need = pack.requires
    if hardware.disk_free_bytes > 0 and need.disk_gb * _GiB > hardware.disk_free_bytes:
        return PackFit(
            None,
            f"needs {_gb(need.disk_gb)} of disk, {_have(hardware.disk_free_bytes)} free",
        )
    if need.unified_memory and not any(_unified(g) for g in hardware.gpus):
        return PackFit(
            None, "needs a GPU that shares system RAM (a Mac, DGX Spark or Strix Halo)"
        )
    largest = max((gpu.vram_bytes for gpu in hardware.gpus), default=0)
    if need.vram_gb is not None and not _holds(largest, need.vram_gb):
        where = f"the largest here has {_have(largest)}" if largest else "none here"
        return PackFit(None, f"needs a GPU with {_gb(need.vram_gb)}, {where}")
    ram = _ram_bytes(hardware)
    # A GPU that shares the system RAM adds nothing to the pool.
    pool = ram + sum(g.vram_bytes for g in hardware.gpus if not _unified(g))
    holds = []
    if need.ram_gb is not None:
        holds.append(_holds(ram, need.ram_gb))
    if need.memory_gb is not None:
        holds.append(_holds(pool, need.memory_gb))
    if holds and all(holds):
        return PackFit("memory")
    if (
        need.ssd_ram_gb is not None
        and _holds(ram, need.ssd_ram_gb)
        and _holds(largest, need.ssd_vram_gb or 0)
    ):
        reason = "reads the experts that do not fit from the SSD, much slower"
        if need.nvme:
            reason += " (an NVMe drive)"
        return PackFit("ssd", reason)
    return PackFit(None, _short_of(need, ram))


def _platforms(card: EngineCard, model: FixedModel, pack: Pack) -> tuple:
    return pack.platforms or model.platforms or card.platforms


def pack_options(
    cards: tuple[EngineCard, ...], hardware: HardwareInfo, *, wsl: bool = False
) -> list[PackOption]:
    """Every pack of every fixed-model engine, with how it does here."""
    options = []
    for card in cards:
        for model in card.models:
            for pack in model.packs:
                narrowed = dataclasses.replace(
                    card, platforms=_platforms(card, model, pack)
                )
                where = availability(narrowed, hardware, wsl=wsl)
                if where.runs:
                    fit = pack_fit(pack, hardware)
                else:
                    fit = PackFit(None, where.reason)
                options.append(PackOption(card, model, pack, where.platform, fit))
    return options


def with_packs(
    row: Availability, hardware: HardwareInfo, *, wsl: bool = False
) -> Availability:
    """A fixed-model engine runs here only when one of its packs does.

    The engine may build for this GPU while every pack it offers needs more
    memory or a GPU that shares system RAM. Then it does not run, and the
    reason most of its packs give says why. When some fit, the reason says how
    many.
    """
    if row.card.kind != "fixed_model" or not row.runs:
        return row
    options = pack_options((row.card,), hardware, wsl=wsl)
    fitting = sum(option.fit.runs for option in options)
    if not fitting:
        reasons = Counter(option.fit.reason for option in options)
        return Availability(
            row.card, None, f"no pack fits, {reasons.most_common(1)[0][0]}"
        )
    count = f"{fitting} of {len(options)} packs fit"
    return Availability(
        row.card, row.platform, f"{row.reason}; {count}" if row.reason else count
    )


__all__ = ["PackFit", "PackOption", "pack_fit", "pack_options", "with_packs"]
