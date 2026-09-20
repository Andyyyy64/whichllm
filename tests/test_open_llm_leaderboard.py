import asyncio
import logging

import httpx
import pytest

from whichllm.models.benchmark_sources import open_llm_leaderboard
from whichllm.models.benchmark_sources.open_llm_leaderboard import (
    _fetch_leaderboard_api,
    fetch_leaderboard_with_fallback,
)


def _rows(offset: int, total: int) -> list[dict[str, dict[str, object]]]:
    return [
        {
            "row": {
                "fullname": f"test/model-{index}",
                "Average ⬆️": 26.0,
            }
        }
        for index in range(offset, min(offset + 100, total))
    ]


def test_leaderboard_api_fetches_all_46_pages():
    offsets: list[int] = []
    total = 4576

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(
            200,
            json={"rows": _rows(offset, total), "num_rows_total": total},
            request=request,
        )

    async def run() -> dict[str, float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _fetch_leaderboard_api(client)

    scores = asyncio.run(run())

    assert len(scores) == total
    assert offsets == list(range(0, 4600, 100))


def test_leaderboard_api_returns_partial_scores_after_exhausted_429(
    monkeypatch, caplog
):
    offsets: list[int] = []

    async def fake_sleep(delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        if offset == 3600:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            json={"rows": _rows(offset, 4576), "num_rows_total": 4576},
            request=request,
        )

    async def run() -> dict[str, float]:
        monkeypatch.setattr("whichllm.models.http.asyncio.sleep", fake_sleep)
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _fetch_leaderboard_api(client)

    with caplog.at_level(logging.WARNING):
        scores = asyncio.run(run())

    assert len(scores) == 3600
    assert offsets[-5:] == [3600] * 5
    assert "rate-limited at offset 3600" in caplog.text
    assert "using 3600 scores fetched so far" in caplog.text


def test_leaderboard_api_first_page_429_still_fails(monkeypatch):
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
                await _fetch_leaderboard_api(client)

    asyncio.run(run())

    assert calls == 5


def test_leaderboard_falls_back_to_rows_when_pyarrow_is_unavailable(monkeypatch):
    expected = {"test/model": 42.0}

    async def unavailable(client: httpx.AsyncClient) -> dict[str, float]:
        raise ImportError("pyarrow is unavailable")

    async def rows(client: httpx.AsyncClient) -> dict[str, float]:
        return expected

    monkeypatch.setattr(open_llm_leaderboard, "_fetch_leaderboard_parquet", unavailable)
    monkeypatch.setattr(open_llm_leaderboard, "_fetch_leaderboard_api", rows)

    async def run() -> dict[str, float]:
        async with httpx.AsyncClient() as client:
            return await fetch_leaderboard_with_fallback(client)

    assert asyncio.run(run()) == expected
