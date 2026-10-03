import asyncio
from datetime import datetime, timezone

import httpx
import pytest

import whichllm.models.http as http_helpers
from whichllm.models.benchmark_sources.chatbot_arena import fetch_arena_scores
from whichllm.models.http import get_with_retries


def test_get_with_retries_retries_429_then_returns_response(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def run() -> httpx.Response:
        monkeypatch.setattr("whichllm.models.http.asyncio.sleep", fake_sleep)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await get_with_retries(
                client,
                "https://example.test/models",
                base_delay=0.01,
                jitter=0,
            )

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == 3
    assert sleeps == [0.01, 0.02]


def test_get_with_retries_honors_retry_after(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                request=request,
            )
        return httpx.Response(200, request=request)

    async def run() -> httpx.Response:
        monkeypatch.setattr("whichllm.models.http.asyncio.sleep", fake_sleep)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await get_with_retries(
                client,
                "https://example.test/models",
                max_delay=10.0,
                jitter=0,
            )

    response = asyncio.run(run())

    assert response.status_code == 200
    assert calls == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    "header,expected_delay",
    [
        ("Thu, 01 Jan 2026 00:00:03 GMT", 3.0),
        ("Thu, 01 Jan 2026 00:01:00 GMT", 10.0),
        ("Wed, 31 Dec 2025 23:59:59 GMT", 0.0),
        ("120", 10.0),
        ("invalid", 1.0),
        ("NaN", 1.0),
        ("Infinity", 1.0),
        ("-Infinity", 1.0),
    ],
)
def test_retry_after_date_invalid_values_and_cap(monkeypatch, header, expected_delay):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    sleeps = []
    calls = 0

    async def fake_sleep(delay):
        sleeps.append(delay)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": header}, request=request)
        return httpx.Response(200, request=request)

    async def run():
        monkeypatch.setattr(http_helpers, "datetime", FixedDatetime)
        monkeypatch.setattr(http_helpers.asyncio, "sleep", fake_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_with_retries(
                client,
                "https://example.test/models",
                attempts=2,
                base_delay=1.0,
                max_delay=10.0,
                jitter=0,
            )

    assert asyncio.run(run()).status_code == 200
    assert calls == 2
    assert sleeps == ([expected_delay] if expected_delay else [])


def test_benchmark_source_retries_429_before_final_failure(monkeypatch):
    calls = 0

    async def fake_sleep(delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, request=request)

    async def run() -> None:
        monkeypatch.setattr("whichllm.models.http.asyncio.sleep", fake_sleep)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_arena_scores(client)

    asyncio.run(run())

    assert calls == 3
