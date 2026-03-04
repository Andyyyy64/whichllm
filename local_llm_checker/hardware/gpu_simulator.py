"""Create synthetic GPUInfo for GPU simulation (--gpu flag).

Uses the dbgpu package (2000+ GPU database from TechPowerUp) for dynamic
lookup of VRAM, memory bandwidth, and compute capability.
"""

from __future__ import annotations

import logging
import re

from local_llm_checker.constants import _GiB
from local_llm_checker.hardware.types import GPUInfo

logger = logging.getLogger(__name__)

_MANUFACTURER_TO_VENDOR: dict[str, str] = {
    "NVIDIA": "nvidia",
    "AMD": "amd",
    "ATI": "amd",
    "Intel": "intel",
    "Apple": "apple",
}

_MANUFACTURER_PREFIXES = ["GeForce ", "Radeon ", "Arc ", "NVIDIA ", "AMD "]


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
        after = db_name[idx + len(name):]
        # Accept if nothing follows, or what follows is VRAM/form-factor spec
        # Reject if a variant suffix follows (Ti, SUPER, Mobile, Max-Q, etc.)
        if not after or re.match(r"^(\s+(\d|GA\d|PCIe|SXM|NVL|CNX))", after):
            candidates.append(db_name)
    if candidates:
        candidates.sort(key=len)
        return db[candidates[0]]
    return None


def _lookup_dbgpu(name: str):
    """Look up GPU spec from dbgpu database. Returns GPUSpecification or None."""
    from dbgpu import GPUDatabase

    db = GPUDatabase.default()

    # Normalize input: "GTX1080" → "GTX 1080"
    normalized = _normalize_gpu_name(name)
    names_to_try = [name] if normalized == name else [name, normalized]

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

        results = process.extract(normalized, db.names, limit=3, scorer=fuzz.token_set_ratio)
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

    # Compute capability (CUDA version in dbgpu = compute capability)
    compute_cap: tuple[int, int] | None = None
    if spec is not None and spec.cuda_major_version is not None:
        compute_cap = (spec.cuda_major_version, spec.cuda_minor_version or 0)

    # Vendor
    vendor = "nvidia"
    if spec is not None:
        vendor = _MANUFACTURER_TO_VENDOR.get(spec.manufacturer, "nvidia")

    display_name = spec.name if spec is not None else name

    return GPUInfo(
        name=f"{display_name} (simulated)",
        vendor=vendor,
        vram_bytes=vram_bytes,
        compute_capability=compute_cap,
        memory_bandwidth_gbps=bandwidth,
    )
