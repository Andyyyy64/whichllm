import asyncio

import httpx
import pytest

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
