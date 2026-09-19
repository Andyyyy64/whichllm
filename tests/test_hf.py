"""Tests for the HuggingFace API client in ``whichllm.models.hf``."""

import asyncio

import httpx

import whichllm.models.hf as hf


def test_fetch_model_by_id_fetches_and_parses_exact_repo(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    captured: dict[str, object] = {}

    async def fake_get_with_retries(client, url: str, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        captured["encoding"] = client.headers["accept-encoding"]
        return httpx.Response(
            200,
            json={
                "id": "ilsp/Llama-Krikri-8B-Instruct",
                "config": {"architectures": ["LlamaForCausalLM"]},
                "safetensors": {"total": 8_202_227_712},
                "cardData": {"license": "llama3.1"},
                "downloads": 3055,
                "likes": 35,
            },
            request=httpx.Request(
                "GET",
                "https://hf-mirror.example/api/models/ilsp/Llama-Krikri-8B-Instruct",
            ),
        )

    monkeypatch.setattr(hf, "get_with_retries", fake_get_with_retries)

    model = asyncio.run(hf.fetch_model_by_id("ilsp/Llama-Krikri-8B-Instruct"))

    assert model is not None
    assert model.id == "ilsp/Llama-Krikri-8B-Instruct"
    assert model.parameter_count == 8_202_227_712
    assert model.architecture == "llama"
    assert captured["url"] == (
        "https://hf-mirror.example/api/models/ilsp/Llama-Krikri-8B-Instruct"
    )
    assert captured["encoding"] == "gzip, deflate"
    assert "config" in captured["params"]["expand[]"]
