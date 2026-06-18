"""Resolve GPU memory bandwidth for *detected* hardware.

Detection passes the raw driver name (e.g. ``"NVIDIA GeForce RTX 5090 Laptop
GPU"``). Unlike ``--gpu`` simulation, where the user typed the name and a fuzzy
guess plus a ``(simulated)`` label is acceptable, a wrong match on real
hardware is worse than no data: giving a laptop card its desktop bandwidth
produces confidently wrong speed estimates and oversized recommendations
(issues #74, #61, #93).

So this resolver is deliberately strict:

* The hand-curated ``GPU_BANDWIDTH`` table stays authoritative and is tried
  first, so existing behaviour for known cards is unchanged.
* The curated lookup is mobile-aware: a laptop/Max-Q driver name will not match
  a desktop key (``"RTX 5090"`` no longer swallows ``"RTX 5090 Laptop GPU"``).
* dbgpu is only consulted to fill gaps, and only via an exact normalised-name
  hit or a name plus a VRAM-size suffix (``"RTX 4060 Ti 16 GB"``). It never
  falls back to fuzzy matching, so a variant qualifier (Ti / SUPER / Mobile /
  Max-Q / XT) can never be silently dropped onto the wrong card.
* ``"Laptop GPU"`` in a driver name is normalised to dbgpu's ``"Mobile"``.

The change is purely additive: cards already covered by ``GPU_BANDWIDTH`` keep
their exact value, and cards that previously resolved to ``None`` now get a
correct bandwidth whenever dbgpu can identify them safely.
"""

from __future__ import annotations

import functools
import logging
import re

from whichllm.constants import GPU_BANDWIDTH, GPU_MEMORY_CLOCK_VARIANTS, _GiB

logger = logging.getLogger(__name__)

_TRADEMARK_RE = re.compile(r"\((?:tm|r)\)", re.IGNORECASE)
_VENDOR_WORD_RE = re.compile(r"\b(?:nvidia|amd|ati|intel|corporation)\b", re.IGNORECASE)
_LAPTOP_GPU_RE = re.compile(r"\blaptop gpu\b", re.IGNORECASE)
_TRAILING_GRAPHICS_RE = re.compile(r"\bgraphics\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_MOBILE_MARKER_RE = re.compile(r"\b(?:laptop|mobile|max-?q)\b", re.IGNORECASE)
# Driver names write VRAM bins without a space ("RTX A2000 12GB"); dbgpu
# writes "RTX A2000 12 GB" (#98).
_VRAM_NOSPACE_RE = re.compile(r"\b(\d+)GB\b", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[(.+)]")
# A dbgpu name may extend a matched query with a VRAM-size bin only
# ("RTX 4060 Ti 16 GB"). A variant word (Mobile, Ti, D, ...) is never benign.
_VRAM_SUFFIX_RE = re.compile(r"^\s+\d+\s*gb\b", re.IGNORECASE)
_VRAM_GB_RE = re.compile(r"(\d+)\s*gb", re.IGNORECASE)


def _normalize_detected_name(name: str) -> str:
    """Reduce a raw driver name toward dbgpu's naming convention."""
    text = _TRADEMARK_RE.sub("", name)
    text = _VENDOR_WORD_RE.sub("", text)
    text = _LAPTOP_GPU_RE.sub("Mobile", text)
    text = _TRAILING_GRAPHICS_RE.sub("", text)
    text = _VRAM_NOSPACE_RE.sub(r"\1 GB", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


_SORTED_BW_KEYS = sorted(GPU_BANDWIDTH, key=len, reverse=True)


def _substring_bandwidth(name: str) -> float | None:
    """Curated ``GPU_BANDWIDTH`` lookup (longest key first), mobile-aware.

    When the detected name is a laptop/Max-Q card, a desktop key is not allowed
    to match it via substring, since the two have very different bandwidth.
    """
    if not name:
        return None
    name_upper = name.upper()
    name_is_mobile = bool(_MOBILE_MARKER_RE.search(name))
    for key in _SORTED_BW_KEYS:
        if key.upper() in name_upper:
            if name_is_mobile and not _MOBILE_MARKER_RE.search(key):
                continue
            return GPU_BANDWIDTH[key]
    return None


def _static_bandwidth(name: str) -> float | None:
    """Curated lookup that also handles compound lspci names.

    Compound names like ``"Navi 22 [Radeon RX 6700/6700 XT/6750 XT /
    6800M/6850M XT]"`` list several variants; the first segment that resolves
    wins, with a ``"RX "`` prefix retry for bare segments like ``"6750 XT"``.
    dbgpu is never consulted for these: the name does not identify a single
    card, so the curated value for the listed family is the safest answer.
    """
    if not name:
        return None
    if "/" not in name:
        return _substring_bandwidth(name)
    bracket = _BRACKET_RE.search(name)
    raw = bracket.group(1) if bracket else name
    for seg in raw.split("/"):
        seg = seg.strip()
        if not seg:
            continue
        bandwidth = _substring_bandwidth(seg) or _substring_bandwidth(f"RX {seg}")
        if bandwidth is not None:
            return bandwidth
    return None


def _vram_gb(canonical_name: str) -> int | None:
    match = _VRAM_GB_RE.search(canonical_name)
    return int(match.group(1)) if match else None


@functools.lru_cache(maxsize=1)
def _dbgpu_index() -> tuple[object | None, dict[str, str] | None]:
    """Build ``{normalized_name: canonical_dbgpu_name}``.

    Returns ``(db, index)`` or ``(None, None)`` if dbgpu is unavailable, so the
    resolver degrades to static-only instead of raising.
    """
    try:
        from dbgpu import GPUDatabase

        db = GPUDatabase.default()
    except Exception as exc:  # pragma: no cover - dbgpu is a hard dependency
        logger.debug("dbgpu unavailable, using static bandwidth only: %s", exc)
        return None, None
    index: dict[str, str] = {}
    for canonical in db.names:
        index.setdefault(_normalize_detected_name(canonical).lower(), canonical)
    return db, index


def _dbgpu_bandwidth(name: str, vram_bytes: int | None) -> float | None:
    """Strict dbgpu bandwidth lookup. Never fuzzy-matches a variant away."""
    db, index = _dbgpu_index()
    if db is None or index is None:
        return None
    query = _normalize_detected_name(name).lower()
    if not query:
        return None

    if query in index:
        candidates = [index[query]]
    else:
        candidates = [
            original
            for normalized, original in index.items()
            if normalized.startswith(query + " ")
            and _VRAM_SUFFIX_RE.match(normalized[len(query) :])
        ]
        if not candidates:
            return None
        if vram_bytes and len(candidates) > 1:
            target_gb = round(vram_bytes / _GiB)
            same_vram = [c for c in candidates if _vram_gb(c) == target_gb]
            if same_vram:
                candidates = same_vram

    # If several VRAM bins remain (VRAM unknown or no bin matched it), take the
    # lowest bandwidth among them: an ambiguous match must never over-promise
    # speed. Bins of the same card usually share one value anyway.
    bandwidths: list[float] = []
    for canonical in candidates:
        try:
            spec = db[canonical]
        except KeyError:  # pragma: no cover - canonical comes from the index
            continue
        bandwidth = getattr(spec, "memory_bandwidth_gb_s", None)
        if bandwidth:
            bandwidths.append(float(bandwidth))
    return min(bandwidths) if bandwidths else None


def _memory_clock_variant_bandwidth(
    name: str, mem_clock_mhz: float | None
) -> float | None:
    """Disambiguate cards sold in multiple memory types by max memory clock.

    Some GPUs (e.g. GTX 1650 GDDR5 vs GDDR6) share a marketing name and PCI
    device id, so only the memory clock tells them apart. Returns the matching
    variant bandwidth when the name is a known dual-memory card and the clock is
    known, else ``None`` (so the caller falls back to the curated default).
    """
    if not name or not mem_clock_mhz or mem_clock_mhz <= 0:
        return None
    name_upper = name.upper()
    for key in sorted(GPU_MEMORY_CLOCK_VARIANTS, key=len, reverse=True):
        if key.upper() in name_upper:
            for min_clock, bandwidth in GPU_MEMORY_CLOCK_VARIANTS[key]:
                if mem_clock_mhz >= min_clock:
                    return bandwidth
    return None


def resolve_detected_bandwidth(
    name: str,
    vram_bytes: int | None = None,
    mem_clock_mhz: float | None = None,
) -> float | None:
    """Best memory bandwidth (GB/s) for a detected GPU, or ``None`` if unknown.

    Memory-clock variant disambiguation wins first (for cards that share a name
    across memory types, e.g. GTX 1650 GDDR5/GDDR6); then curated
    ``GPU_BANDWIDTH``; then dbgpu fills the gaps. ``vram_bytes`` (when known)
    disambiguates same-name cards that ship in multiple VRAM bins;
    ``mem_clock_mhz`` (max memory clock, when known) disambiguates memory-type
    variants. With both unset, behaviour is identical to before.
    """
    if not name:
        return None
    variant = _memory_clock_variant_bandwidth(name, mem_clock_mhz)
    if variant is not None:
        return variant
    return _static_bandwidth(name) or _dbgpu_bandwidth(name, vram_bytes)
