"""Real file sizes in the index: fetched, reused by commit, capped."""

from __future__ import annotations

import asyncio

import httpx

import whichllm.models.file_sizes as file_sizes
from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.models.serialization import dicts_to_models, models_to_dicts
from whichllm.models.types import GGUFVariant, ModelInfo


def _repo(model_id: str, sha: str, downloads: int = 1000) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=2_000_000_000,
        downloads=downloads,
        sha=sha,
        gguf_variants=[
            GGUFVariant("m-Q4_K_M.gguf", "Q4_K_M", 1_060_000_000, is_estimated=True),
            GGUFVariant("m-Q8_0.gguf", "Q8_0", 2_100_000_000, is_estimated=True),
        ],
    )


def _detail(sha: str, sizes: dict[str, int]) -> dict:
    return {
        "sha": sha,
        "siblings": [{"rfilename": f, "size": s} for f, s in sizes.items()],
    }


def _serve(monkeypatch, answers: dict[str, dict | int]) -> list[str]:
    """Answer detail requests per repo id; record which repos were asked."""
    asked = []

    async def fake_get(client, url, **kwargs):
        model_id = url.split("/api/models/", 1)[1]
        asked.append(model_id)
        answer = answers[model_id]
        request = httpx.Request("GET", url)
        if isinstance(answer, int):
            return httpx.Response(answer, request=request)
        return httpx.Response(200, json=answer, request=request)

    monkeypatch.setattr(file_sizes, "get_with_retries", fake_get)
    return asked


_REAL = {"m-Q4_K_M.gguf": 1_280_835_840, "m-Q8_0.gguf": 2_010_000_000}


def test_estimated_sizes_are_replaced_by_the_real_ones(monkeypatch):
    model = _repo("org/Small-2B-GGUF", "abc")
    _serve(monkeypatch, {"org/Small-2B-GGUF": _detail("abc", _REAL)})

    report = asyncio.run(file_sizes.fill_file_sizes([model]))

    assert report.fetched == 1
    sizes = {
        v.quant_type: (v.file_size_bytes, v.is_estimated) for v in model.gguf_variants
    }
    assert sizes == {"Q4_K_M": (1_280_835_840, False), "Q8_0": (2_010_000_000, False)}


def test_previous_sizes_are_reused_while_the_commit_is_unchanged(monkeypatch):
    done = _repo("org/Small-2B-GGUF", "abc")
    for variant in done.gguf_variants:
        variant.file_size_bytes = _REAL[variant.filename]
        variant.is_estimated = False
    previous = {"models": models_to_dicts([done])}
    same = _repo("org/Small-2B-GGUF", "abc")
    changed = _repo("org/Other-2B-GGUF", "new")
    asked = _serve(monkeypatch, {"org/Other-2B-GGUF": _detail("new", _REAL)})

    report = asyncio.run(file_sizes.fill_file_sizes([same, changed], previous))

    assert asked == ["org/Other-2B-GGUF"]
    assert (report.reused, report.fetched) == (1, 1)
    assert not any(v.is_estimated for v in same.gguf_variants)


def test_a_new_commit_is_fetched_again(monkeypatch):
    done = _repo("org/Small-2B-GGUF", "old")
    for variant in done.gguf_variants:
        variant.is_estimated = False
    previous = {"models": models_to_dicts([done])}
    asked = _serve(monkeypatch, {"org/Small-2B-GGUF": _detail("new", _REAL)})

    model = _repo("org/Small-2B-GGUF", "new")
    asyncio.run(file_sizes.fill_file_sizes([model], previous))

    assert asked == ["org/Small-2B-GGUF"]


def test_requests_are_capped_most_downloaded_first(monkeypatch):
    models = [_repo(f"org/M{i}-GGUF", "s", downloads=i) for i in range(5)]
    asked = _serve(monkeypatch, {m.id: _detail("s", _REAL) for m in models})

    report = asyncio.run(file_sizes.fill_file_sizes(models, max_requests=2))

    assert sorted(asked) == ["org/M3-GGUF", "org/M4-GGUF"]
    assert (report.fetched, report.pending) == (2, 3)
    assert models[0].gguf_variants[0].is_estimated


def test_a_rate_limit_stops_the_remaining_requests(monkeypatch):
    models = [_repo(f"org/M{i}-GGUF", "s", downloads=10 - i) for i in range(6)]
    monkeypatch.setattr(file_sizes, "_CONCURRENCY", 1)
    asked = _serve(monkeypatch, {m.id: 429 for m in models})

    report = asyncio.run(file_sizes.fill_file_sizes(models))

    assert asked == ["org/M0-GGUF"]
    assert (report.fetched, report.failed, report.pending) == (0, 0, 6)


def test_a_failed_request_keeps_the_estimates(monkeypatch):
    model = _repo("org/Gone-GGUF", "s")
    _serve(monkeypatch, {"org/Gone-GGUF": 404})

    report = asyncio.run(file_sizes.fill_file_sizes([model]))

    assert report.failed == 1
    assert all(v.is_estimated for v in model.gguf_variants)


def test_the_commit_round_trips_for_repos_with_sizes_only():
    gguf = _repo("org/Small-2B-GGUF", "abc")
    plain = ModelInfo(
        id="org/Plain-2B", family_id="p", name="Plain-2B", parameter_count=2, sha="def"
    )

    dicts = models_to_dicts([gguf, plain])

    assert dicts[0]["sha"] == "abc"
    assert "sha" not in dicts[1]
    assert dicts_to_models(dicts)[0].sha == "abc"


def _checkpoint(model_id: str, sha: str = "abc", downloads: int = 1000) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.rsplit("/", 1)[-1],
        parameter_count=18_700_000_000,
        downloads=downloads,
        sha=sha,
    )


def test_quantized_checkpoints_get_their_real_size(monkeypatch):
    model = _checkpoint("nvidia/Test-35B-A3B-NVFP4")
    siblings = {
        "model-00001-of-00002.safetensors": 10**10,
        "model-00002-of-00002.safetensors": 13 * 10**9,
    }
    _serve(monkeypatch, {model.id: _detail("abc", {**siblings, "README.md": 900})})

    report = asyncio.run(file_sizes.fill_file_sizes([model]))

    assert report.fetched == 1
    assert model.weight_bytes == 23 * 10**9
    assert estimate_weight_bytes(model, None) == 23 * 10**9


def test_previous_checkpoint_size_is_reused_while_the_commit_holds(monkeypatch):
    done = _checkpoint("nvidia/Test-35B-A3B-NVFP4")
    done.weight_bytes = 23 * 10**9
    previous = {"models": models_to_dicts([done])}
    asked = _serve(monkeypatch, {})

    same = _checkpoint("nvidia/Test-35B-A3B-NVFP4")
    report = asyncio.run(file_sizes.fill_file_sizes([same], previous))

    assert asked == []
    assert report.reused == 1
    assert same.weight_bytes == 23 * 10**9


def test_unquantized_checkpoints_are_left_to_the_parameter_count(monkeypatch):
    model = _checkpoint("Qwen/Qwen3-8B")
    asked = _serve(monkeypatch, {})
    asyncio.run(file_sizes.fill_file_sizes([model]))
    assert asked == []
    assert model.weight_bytes is None


def test_shards_win_over_a_consolidated_copy():
    data = _detail(
        "abc",
        {
            "consolidated.safetensors": 5 * 10**9,
            "model-00001-of-00001.safetensors": 5 * 10**9,
            "onnx/model.safetensors": 9 * 10**9,
        },
    )
    assert file_sizes._checkpoint_bytes(data) == 5 * 10**9


def test_pickle_shards_count_when_there_are_no_safetensors():
    data = _detail(
        "abc", {"pytorch_model-00001.bin": 3 * 10**9, "training_args.bin": 4000}
    )
    assert file_sizes._checkpoint_bytes(data) == 3 * 10**9
