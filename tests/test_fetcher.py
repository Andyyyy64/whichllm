"""Tests for model metadata normalization in fetcher."""

import asyncio

import httpx

import whichllm.models.fetcher as fetcher
from whichllm.models.fetcher import (
    _extract_hf_eval_score,
    _extract_published_at,
    _hf_api_url,
    _normalize_param_count,
    _parse_model,
    dicts_to_models,
    models_to_dicts,
)
from whichllm.models.types import ModelInfo


def test_hf_api_url_uses_default_endpoint(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    assert _hf_api_url("models") == "https://huggingface.co/api/models"


def test_hf_api_url_respects_hf_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")

    assert _hf_api_url("models/Qwen/Qwen3-8B") == (
        "https://hf-mirror.example/api/models/Qwen/Qwen3-8B"
    )


def test_hf_api_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")

    assert _hf_api_url("/models") == "https://hf-mirror.example/api/models"


def test_hf_api_url_rejects_empty_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", " ")

    try:
        _hf_api_url("models")
    except ValueError as exc:
        assert "HF_ENDPOINT must not be empty" in str(exc)
    else:
        raise AssertionError("expected empty HF_ENDPOINT to raise")


def test_hf_api_url_rejects_endpoint_without_scheme(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "hf-mirror.example")

    try:
        _hf_api_url("models")
    except ValueError as exc:
        assert "HF_ENDPOINT must start with http:// or https://" in str(exc)
    else:
        raise AssertionError("expected invalid HF_ENDPOINT to raise")


def test_fetch_models_respects_hf_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    urls: list[str] = []
    encodings: list[str] = []

    async def fake_get_with_retries(client, url: str, **kwargs):
        urls.append(url)
        encodings.append(client.headers["accept-encoding"])
        request = httpx.Request("GET", url)
        if "/models/" in url:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=[], request=request)

    monkeypatch.setattr(fetcher, "get_with_retries", fake_get_with_retries)

    models = asyncio.run(fetcher.fetch_models(limit=1, include_vision=False))

    assert models == []
    assert urls
    assert all(url.startswith("https://hf-mirror.example/api/") for url in urls)
    assert "https://hf-mirror.example/api/models" in urls
    assert not any(url.startswith("https://huggingface.co/api/") for url in urls)
    assert set(encodings) == {"gzip, deflate"}


def test_fetch_model_published_at_respects_hf_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    urls: list[str] = []
    encodings: list[str] = []

    async def fake_get(self, url: str, **kwargs):
        urls.append(url)
        encodings.append(self.headers["accept-encoding"])
        return httpx.Response(
            200,
            json={"createdAt": "2026-06-22T00:00:00.000Z"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = asyncio.run(fetcher.fetch_model_published_at(["Qwen/Qwen3-8B"]))

    assert result == {"Qwen/Qwen3-8B": "2026-06-22T00:00:00.000Z"}
    assert urls == ["https://hf-mirror.example/api/models/Qwen/Qwen3-8B"]
    assert encodings == ["gzip, deflate"]


def test_normalize_param_count_for_quantized_repo_uses_size_hint():
    corrected = _normalize_param_count(
        extracted=5_233_828_308,
        model_id="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        base_model="google/gemma-3-27b-it",
    )
    assert corrected == 27_000_000_000


def test_normalize_param_count_keeps_reasonable_value():
    kept = _normalize_param_count(
        extracted=11_765_788_416,
        model_id="MaziyarPanahi/gemma-3-12b-it-GGUF",
        base_model="google/gemma-3-12b-it",
    )
    assert kept == 11_765_788_416


def test_normalize_param_count_with_no_hint_keeps_original():
    kept = _normalize_param_count(
        extracted=3_820_000_000,
        model_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        base_model=None,
    )
    assert kept == 3_820_000_000


def test_dicts_to_models_normalizes_cached_parameter_count():
    models = dicts_to_models(
        [
            {
                "id": "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
                "family_id": "gemma-3-27b",
                "name": "gemma-3-27b-it-GPTQ-4b-128g",
                "parameter_count": 5_233_828_308,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
                "base_model": "google/gemma-3-27b-it",
            }
        ]
    )
    assert len(models) == 1
    assert models[0].parameter_count == 27_000_000_000


def test_dicts_to_models_refreshes_cached_deepseek_v4_flash_counts():
    models = dicts_to_models(
        [
            {
                "id": "deepseek-ai/DeepSeek-V4-Flash",
                "family_id": "deepseek-v4-flash",
                "name": "DeepSeek-V4-Flash",
                "parameter_count": 158_069_433_298,
                "parameter_count_active": 10_000_000_000,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 284_000_000_000
    assert models[0].parameter_count_active == 13_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_uses_case_insensitive_curated_active_params():
    models = dicts_to_models(
        [
            {
                "id": "google/gemma-4-26B-A4B-it",
                "family_id": "gemma-4-26b-a4b",
                "name": "gemma-4-26B-A4B-it",
                "parameter_count": 26_544_131_376,
                "parameter_count_active": None,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_800_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_a3b_active_params_from_cached_qwen_model():
    models = dicts_to_models(
        [
            {
                "id": "Qwen/Qwen3.6-35B-A3B",
                "family_id": "qwen3.6-35b-a3b",
                "name": "Qwen3.6-35B-A3B",
                "parameter_count": 35_951_822_704,
                "parameter_count_active": None,
                "architecture": "qwen3_5moe",
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_a3b_active_params_from_base_model():
    models = dicts_to_models(
        [
            {
                "id": "local/Qwen36-GGUF",
                "family_id": "qwen36-gguf",
                "name": "Qwen36-GGUF",
                "parameter_count": 34_660_610_688,
                "parameter_count_active": None,
                "architecture": "qwen35moe",
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
                "base_model": "Qwen/Qwen3.6-35B-A3B",
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_refreshes_stale_xiaomi_moe_cache_counts():
    models = dicts_to_models(
        [
            {
                "id": "XiaomiMiMo/MiMo-V2.5-Pro",
                "family_id": "mimo-v2.5-pro",
                "name": "MiMo-V2.5-Pro",
                "parameter_count": 58_000_000_000,
                "parameter_count_active": 11_000_000_000,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 1_020_000_000_000
    assert models[0].parameter_count_active == 42_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_missing_known_parameter_count():
    models = dicts_to_models(
        [
            {
                "id": "zai-org/GLM-5",
                "family_id": "glm-5",
                "name": "GLM-5",
                "parameter_count": 0,
                "parameter_count_active": None,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 744_000_000_000
    assert models[0].parameter_count_active == 40_000_000_000
    assert models[0].is_moe is True


def test_parse_model_uses_current_glm5_and_xiaomi_active_counts():
    glm = _parse_model(
        {
            "id": "zai-org/GLM-5",
            "config": {"architectures": ["GlmForCausalLM"]},
            "safetensors": {"total": 753_864_139_008},
            "siblings": [],
            "cardData": {},
        }
    )
    mimo = _parse_model(
        {
            "id": "XiaomiMiMo/MiMo-V2-Flash",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "safetensors": {"total": 309_785_318_400},
            "siblings": [],
            "cardData": {},
        }
    )

    assert glm is not None
    assert glm.parameter_count_active == 40_000_000_000
    assert mimo is not None
    assert mimo.parameter_count_active == 15_000_000_000


def test_parse_model_recovers_qwen36_a3b_active_params_from_name():
    parsed = _parse_model(
        {
            "id": "Qwen/Qwen3.6-35B-A3B",
            "config": {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
            },
            "safetensors": {"total": 35_951_822_704},
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.parameter_count == 35_951_822_704
    assert parsed.parameter_count_active == 3_000_000_000
    assert parsed.is_moe is True


def test_models_cache_roundtrip_keeps_published_at():
    models = [
        ModelInfo(
            id="Qwen/Qwen3-8B-AWQ",
            family_id="qwen3-8b",
            name="Qwen3-8B-AWQ",
            parameter_count=8_000_000_000,
            published_at="2025-09-17T12:34:56.000Z",
            downloads=123_456,
            likes=789,
        )
    ]
    cached = models_to_dicts(models)
    restored = dicts_to_models(cached)
    assert len(restored) == 1
    assert restored[0].published_at == "2025-09-17T12:34:56.000Z"
    assert restored[0].downloads == 123_456


def test_extract_published_at_prefers_created_at():
    value = _extract_published_at(
        {
            "createdAt": "2025-01-01T00:00:00.000Z",
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2025-01-01T00:00:00.000Z"


def test_extract_published_at_falls_back_to_last_modified():
    value = _extract_published_at(
        {
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2026-01-01T00:00:00.000Z"


def test_parse_model_keeps_split_gguf_as_single_variant():
    parsed = _parse_model(
        {
            "id": "org/Test-8B-GGUF",
            "config": {
                "architectures": ["LlamaForCausalLM"],
            },
            "safetensors": {"total": 8_000_000_000},
            "siblings": [
                {
                    "rfilename": "model-Q4_K_M-00001-of-00002.gguf",
                    "size": 2_000_000_000,
                },
                {
                    "rfilename": "model-Q4_K_M-00002-of-00002.gguf",
                    "size": 2_500_000_000,
                },
                {"rfilename": "model-Q8_0.gguf", "size": 8_000_000_000},
            ],
            "cardData": {},
        }
    )
    assert parsed is not None
    q4 = [v for v in parsed.gguf_variants if v.quant_type == "Q4_K_M"]
    q8 = [v for v in parsed.gguf_variants if v.quant_type == "Q8_0"]
    assert len(q4) == 1
    assert len(q8) == 1
    assert q4[0].file_size_bytes == 4_500_000_000


def test_extract_hf_eval_score_uses_general_datasets_and_median():
    score = _extract_hf_eval_score(
        {
            "evalResults": [
                {
                    "filename": ".eval_results/mmlu-pro.yaml",
                    "data": {"dataset": {"id": "TIGER-Lab/MMLU-Pro"}, "value": 48.3},
                },
                {
                    "filename": ".eval_results/gsm8k.yaml",
                    "data": {"dataset": {"id": "openai/gsm8k"}, "value": 84.5},
                },
                {
                    "filename": ".eval_results/hle_medium_with_tools.yaml",
                    "data": {
                        "dataset": {"id": "cais/hle"},
                        "value": 99.0,
                        "notes": "Reasoning: medium, With tools",
                    },
                },
                {
                    "filename": ".eval_results/swe_bench.yaml",
                    "data": {
                        "dataset": {"id": "SWE-bench/SWE-bench_Verified"},
                        "value": 53.2,
                    },
                },
            ]
        }
    )
    # general対象(mmlu/gsm8k)のみ集計し、中央値を使う。
    assert score == 66.4


def test_parse_model_extracts_hf_eval_benchmark_score():
    parsed = _parse_model(
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "safetensors": {"total": 8_000_000_000},
            "siblings": [],
            "cardData": {},
            "evalResults": [
                {
                    "filename": ".eval_results/mmlu-pro.yaml",
                    "data": {"dataset": {"id": "TIGER-Lab/MMLU-Pro"}, "value": 48.3},
                },
                {
                    "filename": ".eval_results/gsm8k.yaml",
                    "data": {"dataset": {"id": "openai/gsm8k"}, "value": 84.5},
                },
            ],
        }
    )
    assert parsed is not None
    assert parsed.benchmark_scores.get("hf_eval") == 66.4


def test_deepseek_v4_flash_uses_model_card_counts_over_hf_tensor_metadata():
    """DeepSeek V4 Flash's mixed-precision HF tensor metadata reports a
    smaller stored tensor count than the model-card total. Ranking and GGUF
    synthesis must use the published model capacity instead."""

    parsed = _parse_model(
        {
            "id": "deepseek-ai/DeepSeek-V4-Flash",
            "config": {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
            "safetensors": {
                "total": 158_069_433_298,
                "parameters": {
                    "BF16": 1_415_259_264,
                    "F8_E8M0": 8_858_737_664,
                    "F8_E4M3": 6_023_020_544,
                    "I8": 141_733_920_768,
                },
            },
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.parameter_count == 284_000_000_000
    assert parsed.parameter_count_active == 13_000_000_000


def test_parse_model_resolves_gemma3_sliding_window():
    parsed = _parse_model(
        {
            "id": "google/gemma-3-27b-it",
            "config": {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "max_position_embeddings": 131072,
            },
            "safetensors": {"total": 27_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window == 1024
    assert parsed.sliding_window_global_ratio == 1.0 / 6.0


def test_parse_model_resolves_gpt_oss_sliding_window():
    parsed = _parse_model(
        {
            "id": "openai/gpt-oss-20b",
            "config": {
                "architectures": ["GptOssForCausalLM"],
                "model_type": "gpt_oss",
                "sliding_window": 128,
                "max_position_embeddings": 131072,
            },
            "safetensors": {"total": 20_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window == 128
    assert parsed.sliding_window_global_ratio == 0.5


def test_parse_model_does_not_honor_mistral_sliding_window():
    """Mistral declares sliding_window in config but runtimes ignore it."""
    parsed = _parse_model(
        {
            "id": "mistralai/Mistral-7B-v0.1",
            "config": {
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "sliding_window": 4096,
                "max_position_embeddings": 32768,
            },
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window is None
    assert parsed.sliding_window_global_ratio is None


def test_parse_model_respects_disabled_sliding_window():
    parsed = _parse_model(
        {
            "id": "google/gemma-3-4b-it",
            "config": {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "sliding_window": 1024,
                "use_sliding_window": False,
            },
            "safetensors": {"total": 4_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window is None
    assert parsed.sliding_window_global_ratio is None


def test_parse_model_does_not_honor_gemma1_window():
    """Gemma-1 has no sliding window; the gemma2/gemma3 keys must not match it."""
    parsed = _parse_model(
        {
            "id": "google/gemma-7b",
            "config": {
                "architectures": ["GemmaForCausalLM"],
                "model_type": "gemma",
            },
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window is None


def test_parse_model_sliding_window_from_gguf_metadata():
    """GGUF-only repos: trust the authoritative GGUF architecture metadata."""
    parsed = _parse_model(
        {
            "id": "unsloth/gemma-3-12b-it-GGUF",
            "config": {},
            "gguf": {"architecture": "gemma3"},
            "safetensors": {"total": 12_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window == 1024
    assert parsed.sliding_window_global_ratio == 1.0 / 6.0


def test_parse_model_sliding_window_unset_without_metadata():
    """With neither HF config nor GGUF arch metadata, SWA stays unhonored.

    Detection is limited to authoritative metadata; a GGUF-only repo whose id
    merely names an SWA family keeps the conservative full-context estimate
    rather than being guessed at from the id.
    """
    parsed = _parse_model(
        {
            "id": "TheBloke/gemma-2-9b-it-GGUF",
            "config": {},
            "gguf": {},
            "safetensors": {"total": 9_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )
    assert parsed is not None
    assert parsed.sliding_window is None
    assert parsed.sliding_window_global_ratio is None


def test_models_cache_roundtrip_keeps_sliding_window():
    models = [
        ModelInfo(
            id="google/gemma-3-27b-it",
            family_id="gemma-3-27b",
            name="gemma-3-27b-it",
            parameter_count=27_000_000_000,
            sliding_window=1024,
            sliding_window_global_ratio=1.0 / 6.0,
        )
    ]
    restored = dicts_to_models(models_to_dicts(models))
    assert restored[0].sliding_window == 1024
    assert restored[0].sliding_window_global_ratio == 1.0 / 6.0
