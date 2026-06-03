"""Shared low-level helpers: byte/param/date formatters and color blending."""

from __future__ import annotations

from datetime import datetime
from math import log10

from whichllm.engine.types import CompatibilityResult


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


def _format_params(count: int) -> str:
    """Format parameter count."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return str(count)


def _format_downloads(downloads: int) -> str:
    """Format download count for compact table display."""
    if downloads >= 1_000_000:
        return f"{downloads / 1_000_000:.1f}M"
    if downloads >= 1_000:
        return f"{downloads / 1_000:.1f}K"
    return str(downloads)


def _format_published_at(value: str | None) -> str:
    """Format published datetime into YYYY-MM-DD."""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return value[:10] if len(value) >= 10 else value


def _format_speed(result: CompatibilityResult) -> str:
    speed = result.estimated_tok_per_sec
    if speed is None:
        return "N/A"
    base = f"{speed:.1f} tok/s"
    if result.speed_confidence == "low":
        return f"[red]{base} ?[/red]"
    if result.speed_confidence == "medium":
        return f"[yellow]{base} ~[/yellow]"
    return base


def _parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _blend_hex(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = _lerp_channel(a[0], b[0], t)
    g = _lerp_channel(a[1], b[1], t)
    bch = _lerp_channel(a[2], b[2], t)
    return f"#{r:02x}{g:02x}{bch:02x}"


def _downloads_style(downloads: int, min_log: float, max_log: float) -> str:
    if downloads <= 0:
        return "grey50"
    dlog = log10(max(downloads, 1))
    span = max(max_log - min_log, 1e-6)
    t = (dlog - min_log) / span
    return _blend_hex((145, 80, 80), (55, 190, 120), t)


def _published_style(
    published: datetime | None,
    oldest_ts: float | None,
    newest_ts: float | None,
) -> str:
    if published is None or oldest_ts is None or newest_ts is None:
        return "grey50"
    pts = published.timestamp()
    span = max(newest_ts - oldest_ts, 1e-6)
    t = (pts - oldest_ts) / span
    return _blend_hex((190, 85, 85), (80, 190, 110), t)
