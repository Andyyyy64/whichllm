"""Create synthetic GPUInfo for GPU simulation (--gpu flag).

Uses the dbgpu package (2000+ GPU database from TechPowerUp) for dynamic
lookup of VRAM, memory bandwidth, and compute capability.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbgpu import GPUSpecification

from whichllm.constants import AMD_SHARED_MEMORY_APU_MARKERS, GPU_BANDWIDTH, _GiB
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_MANUFACTURER_TO_VENDOR: dict[str, str] = {
    "NVIDIA": "nvidia",
    "AMD": "amd",
    "ATI": "amd",
    "Intel": "intel",
    "Apple": "apple",
}

_MANUFACTURER_PREFIXES = ["GeForce ", "Radeon ", "Arc ", "NVIDIA ", "AMD "]
_COMMON_GPU_ALIASES: dict[str, list[str]] = {
    "a10080gb": [
        "NVIDIA A100 PCIe 80 GB",
        "NVIDIA A100 SXM4 80 GB",
    ],
    "h10080gb": [
        "NVIDIA H100 PCIe 80 GB",
        "NVIDIA H100 SXM5 80 GB",
    ],
}

# Apple Silicon chips. dbgpu does not include these (it tracks discrete GPUs
# via TechPowerUp), but users routinely simulate with --gpu "M2 Max" etc.
# Without this short-circuit, "M1" fuzzy-matches the 1997 ATI Rage Mobility-M1
# and "M3 Max" falls through to vendor=nvidia default.
# Format: chip_name -> (canonical_name, default_unified_memory_gb).
# --vram override is still respected; the default is the most common SKU.
_APPLE_SILICON_CHIPS: dict[str, tuple[str, float]] = {
    "M1": ("Apple M1", 8.0),
    "M1 Pro": ("Apple M1 Pro", 16.0),
    "M1 Max": ("Apple M1 Max", 32.0),
    "M1 Ultra": ("Apple M1 Ultra", 64.0),
    "M2": ("Apple M2", 16.0),
    "M2 Pro": ("Apple M2 Pro", 16.0),
    "M2 Max": ("Apple M2 Max", 32.0),
    "M2 Ultra": ("Apple M2 Ultra", 64.0),
    "M3": ("Apple M3", 16.0),
    "M3 Pro": ("Apple M3 Pro", 18.0),
    "M3 Max": ("Apple M3 Max", 36.0),
    "M3 Ultra": ("Apple M3 Ultra", 96.0),
    "M4": ("Apple M4", 16.0),
    "M4 Pro": ("Apple M4 Pro", 24.0),
    "M4 Max": ("Apple M4 Max", 36.0),
    "M4 Ultra": ("Apple M4 Ultra", 64.0),
    "M5": ("Apple M5", 16.0),
    "M5 Pro": ("Apple M5 Pro", 24.0),
    "M5 Max": ("Apple M5 Max", 36.0),
}


def _lookup_apple_silicon(
    name: str,
) -> tuple[str, str, float, float] | None:
    """Match Apple Silicon chip names. Returns (canonical_name, vendor,
    default_vram_gb, bandwidth_gbps) or None.

    Matches are case-insensitive and accept "M2 Max", "m2max", and
    display-name forms such as "Apple M2 Max". Longest match wins so
    "M2 Ultra" does not get caught by the "M2" entry.
    """
    compact = re.sub(r"\s+", "", name).lower()
    if compact.startswith("apple"):
        compact = compact.removeprefix("apple")

    # Sort keys by length descending so "M2 Ultra" wins over "M2".
    for key in sorted(_APPLE_SILICON_CHIPS, key=len, reverse=True):
        key_compact = re.sub(r"\s+", "", key).lower()
        if compact == key_compact:
            canonical, default_vram = _APPLE_SILICON_CHIPS[key]
            bandwidth = GPU_BANDWIDTH.get(key, 100.0)
            return canonical, "apple", default_vram, bandwidth
    return None


def _is_amd_shared_memory_apu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in AMD_SHARED_MEMORY_APU_MARKERS)


def _lookup_static_bandwidth(name: str) -> float | None:
    name_upper = name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in name_upper:
            return GPU_BANDWIDTH[key]
    return None


def _normalize_gpu_name(name: str) -> str:
    """Normalize user input: 'GTX1080' → 'GTX 1080', 'RX7900XTX' → 'RX 7900 XTX'."""
    # Insert space between letters and digits
    name = re.sub(r"([A-Za-z])(\d)", r"\1 \2", name)
    # Insert space between digits and letters
    name = re.sub(r"(\d)([A-Za-z])", r"\1 \2", name)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", name).strip()


def _substring_search(db, name: str):
    """Substring match with word-boundary filtering.

    e.g. "RTX 3060" should match "GeForce RTX 3060 12 GB" but NOT "GeForce RTX 3060 Ti".
    """
    name_upper = name.upper()
    candidates = []
    for db_name in db.names:
        idx = db_name.upper().find(name_upper)
        if idx < 0:
            continue
        after = db_name[idx + len(name) :]
        # Accept if nothing follows, or what follows is VRAM/form-factor spec
        # Reject if a variant suffix follows (Ti, SUPER, Mobile, Max-Q, etc.)
        if not after or re.match(r"^(\s+(\d|GA\d|PCIe|SXM|NVL|CNX))", after):
            candidates.append(db_name)
    if candidates:
        candidates.sort(key=len)
        return db[candidates[0]]
    return None


def _lookup_dbgpu(name: str) -> GPUSpecification | None:
    """Look up GPU spec from dbgpu database. Returns GPUSpecification or None."""
    from dbgpu import GPUDatabase

    db = GPUDatabase.default()

    # Normalize input: "GTX1080" → "GTX 1080"
    normalized = _normalize_gpu_name(name)
    compact = re.sub(r"\s+", "", normalized.lower())
    names_to_try = [name] if normalized == name else [name, normalized]
    alias_hits = _COMMON_GPU_ALIASES.get(compact)
    if alias_hits:
        names_to_try.extend(alias_hits)

    for n in names_to_try:
        # 1) Exact key lookup
        try:
            return db[n]
        except KeyError:
            pass

        # 2) Try with common manufacturer prefixes
        for prefix in _MANUFACTURER_PREFIXES:
            try:
                return db[prefix + n]
            except KeyError:
                pass

        # 3) Substring match with word-boundary filtering
        result = _substring_search(db, n)
        if result is not None:
            return result

    # 4) Fuzzy search as last resort (use normalized name + token_set_ratio)
    try:
        from thefuzz import fuzz, process

        results = process.extract(
            normalized, db.names, limit=3, scorer=fuzz.token_set_ratio
        )
        if results and results[0][1] >= 90:
            return db[results[0][0]]
        # Store top suggestions for error messages
        if results:
            _last_suggestions[:] = [(n, s) for n, s in results if s >= 70]
    except ImportError:
        pass
    return None


# Mutable list to pass suggestions from lookup to error message
_last_suggestions: list[tuple[str, int]] = []


def parse_synthetic_gpu_specs(values: Sequence[str] | str) -> list[str]:
    """Expand CLI GPU simulation values into individual GPU names.

    Accepts repeated options, comma-separated names, and count shorthand such
    as ``2x RTX 4090``. The returned names are still looked up by
    ``create_synthetic_gpu`` so existing fuzzy matching and aliases stay in
    one place.
    """
    raw_values = [values] if isinstance(values, str) else list(values)
    gpu_names: list[str] = []

    for raw in raw_values:
        for part in raw.split(","):
            spec = part.strip()
            if not spec:
                raise ValueError("Empty GPU entry in --gpu.")

            count_match = re.match(r"^(\d+)\s*x\s+(.+)$", spec, re.IGNORECASE)
            if count_match:
                count = int(count_match.group(1))
                name = count_match.group(2).strip()
                if count < 1:
                    raise ValueError("GPU count must be at least 1.")
                if not name:
                    raise ValueError("GPU count shorthand requires a GPU name.")
                gpu_names.extend([name] * count)
            else:
                gpu_names.append(spec)

    if not gpu_names:
        raise ValueError("At least one GPU must be specified.")
    return gpu_names


def create_synthetic_gpus(
    values: Sequence[str] | str,
    vram_override_gb: float | None = None,
) -> list[GPUInfo]:
    """Create one or more synthetic GPUs from CLI-style values."""
    names = parse_synthetic_gpu_specs(values)
    if vram_override_gb is not None and len(names) != 1:
        raise ValueError(
            "--vram currently supports exactly one simulated GPU. "
            "For multi-GPU simulation, specify known GPU names and omit --vram."
        )
    return [create_synthetic_gpu(name, vram_override_gb) for name in names]


def create_synthetic_gpu(name: str, vram_override_gb: float | None = None) -> GPUInfo:
    """Create a synthetic GPUInfo from a GPU name.

    Looks up specs from the dbgpu database (2000+ GPUs).

    Args:
        name: GPU name (e.g. "RTX 4090", "RX 7900 XTX").
        vram_override_gb: Override VRAM in GB. Required if GPU not in database.

    Returns:
        GPUInfo with ``(simulated)`` suffix in the name.

    Raises:
        ValueError: If GPU is not found and no vram_override_gb given.
    """
    _last_suggestions.clear()

    amd_shared_memory_apu = _is_amd_shared_memory_apu(name)

    # Apple Silicon short-circuit: dbgpu has no Apple entries, so we check
    # first to avoid fuzzy-matching "M1" against "Rage Mobility-M1".
    apple_hit = _lookup_apple_silicon(name)
    if apple_hit is not None:
        canonical, vendor, default_vram_gb, bandwidth = apple_hit
        vram_gb = vram_override_gb if vram_override_gb is not None else default_vram_gb
        return GPUInfo(
            name=f"{canonical} (simulated)",
            vendor=vendor,
            vram_bytes=int(vram_gb * _GiB),
            memory_bandwidth_gbps=bandwidth,
            shared_memory=True,
            vram_overridden=vram_override_gb is not None,
        )

    spec = _lookup_dbgpu(name)

    # VRAM
    if vram_override_gb is not None:
        vram_bytes = int(vram_override_gb * _GiB)
    elif spec is not None and spec.memory_size_gb:
        vram_bytes = int(spec.memory_size_gb * _GiB)
    else:
        msg = f"Unknown GPU '{name}'."
        if _last_suggestions:
            candidates = ", ".join(n for n, _ in _last_suggestions)
            msg += f" Did you mean: {candidates}?"
        msg += " Use --vram to specify VRAM in GB."
        raise ValueError(msg)

    # Bandwidth
    bandwidth: float | None = None
    if spec is not None and spec.memory_bandwidth_gb_s:
        bandwidth = spec.memory_bandwidth_gb_s
    if bandwidth is None:
        bandwidth = _lookup_static_bandwidth(name)

    # Compute capability (CUDA version in dbgpu = compute capability)
    compute_cap: tuple[int, int] | None = None
    if spec is not None and spec.cuda_major_version is not None:
        compute_cap = (spec.cuda_major_version, spec.cuda_minor_version or 0)

    # Vendor
    vendor = "nvidia"
    if spec is not None:
        vendor = _MANUFACTURER_TO_VENDOR.get(spec.manufacturer, "nvidia")
    elif amd_shared_memory_apu:
        vendor = "amd"

    display_name = spec.name if spec is not None else name

    return GPUInfo(
        name=f"{display_name} (simulated)",
        vendor=vendor,
        vram_bytes=vram_bytes,
        compute_capability=compute_cap,
        memory_bandwidth_gbps=bandwidth,
        shared_memory=amd_shared_memory_apu,
        vram_overridden=vram_override_gb is not None,
    )
