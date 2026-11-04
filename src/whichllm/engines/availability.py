"""Which engines run on this machine, and through which accelerator."""

from __future__ import annotations

from dataclasses import dataclass

from whichllm.engines.cards import EngineCard, Platform
from whichllm.hardware.types import GPUInfo, HardwareInfo

_OS_NAMES = {"linux": "Linux", "windows": "Windows", "darwin": "macOS"}

# Accelerators a GPU vendor can use, best first. Vulkan is the cross-vendor
# fallback; the CPU always works.
_VENDOR_ACCELS = {
    "nvidia": ("cuda", "vulkan"),
    "amd": ("rocm", "vulkan"),
    "intel": ("sycl", "vulkan"),
    "apple": ("metal",),
}


@dataclass(frozen=True)
class Availability:
    card: EngineCard
    platform: Platform | None  # the best way it runs here; None when it does not
    reason: str | None = None  # why it does not run, or why it fell back

    @property
    def runs(self) -> bool:
        return self.platform is not None


def _main_gpu(hardware: HardwareInfo) -> GPUInfo | None:
    if not hardware.gpus:
        return None
    return max(hardware.gpus, key=lambda gpu: gpu.vram_bytes)


def _capability(capability: tuple[int, int]) -> str:
    return f"{capability[0]}.{capability[1]}"


def _blocker(
    platform: Platform, hardware: HardwareInfo, gpu: GPUInfo | None
) -> str | None:
    """Why this platform entry does not fit the machine, or None when it does."""
    if platform.requires_avx2 and not hardware.has_avx2:
        return "needs a CPU with AVX2"
    needed = platform.min_compute_capability
    if needed is not None and gpu is not None:
        if gpu.compute_capability is None:
            return f"needs CUDA compute capability {_capability(needed)}, unknown here"
        if gpu.compute_capability < needed:
            return (
                f"needs CUDA compute capability {_capability(needed)}, this GPU "
                f"is {_capability(gpu.compute_capability)}"
            )
    return None


def availability(card: EngineCard, hardware: HardwareInfo) -> Availability:
    """The best accelerator the engine can use here, or why it cannot run."""
    on_os = [p for p in card.platforms if hardware.os in p.os]
    if not on_os:
        return Availability(card, None, f"no {_OS_NAMES[hardware.os]} build")
    gpu = _main_gpu(hardware)
    wanted = list(_VENDOR_ACCELS.get(gpu.vendor, ())) if gpu else []
    best = wanted[0] if wanted else None
    blocked: list[str] = []
    for accel in [*wanted, "cpu"]:
        for platform in on_os:
            if platform.accel != accel:
                continue
            problem = _blocker(platform, hardware, gpu if accel == "cuda" else None)
            if problem is None:
                # Falling back from the GPU's own accelerator is worth saying,
                # with what kept it out.
                reason = None
                if best is not None and accel != best:
                    reason = blocked[0] if blocked else f"no {best} build"
                return Availability(card, platform, reason)
            blocked.append(problem)
    reason = blocked[0] if blocked else "no build for this hardware"
    return Availability(card, None, reason)


__all__ = ["Availability", "availability"]
