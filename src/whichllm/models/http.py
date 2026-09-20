from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_ACCEPT_ENCODING = "gzip, deflate"


def _retry_after_delay(response: httpx.Response) -> float | None:
    """Return the server-requested retry delay, if it is valid."""
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()

    return max(0.0, delay)


async def get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 2.0,
    jitter: float = 0.1,
    retry_status_codes: set[int] | None = None,
    **kwargs,
) -> httpx.Response:
    """GET with bounded retry/backoff for transient HTTP failures."""
    retry_codes = retry_status_codes or RETRYABLE_STATUS_CODES
    last_attempt = max(1, attempts) - 1

    for attempt in range(last_attempt + 1):
        retry_after = None
        try:
            response = await client.get(url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt >= last_attempt:
                raise
        else:
            if response.status_code not in retry_codes or attempt >= last_attempt:
                return response
            if response.status_code == 429:
                retry_after = _retry_after_delay(response)

        if retry_after is not None:
            delay = min(max_delay, retry_after)
        else:
            delay = min(max_delay, base_delay * (2**attempt))
        if jitter > 0 and retry_after is None:
            delay += random.uniform(0, jitter)
        if delay > 0:
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable retry state")
