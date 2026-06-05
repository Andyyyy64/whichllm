from __future__ import annotations

import asyncio
import random

import httpx

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
        try:
            response = await client.get(url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt >= last_attempt:
                raise
        else:
            if response.status_code not in retry_codes or attempt >= last_attempt:
                return response

        delay = min(max_delay, base_delay * (2**attempt))
        if jitter > 0:
            delay += random.uniform(0, jitter)
        if delay > 0:
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable retry state")
