"""Uncensored / abliterated model detection and the --uncensored modes."""

import pytest
import typer

from whichllm.cli_options import _resolve_uncensored_mode
from whichllm.engine.ranker import rank_models
from whichllm.engine.ranking_filters import _derivative_name_penalty
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.models.uncensored import (
    UNCENSORED_INCLUDE,
    UNCENSORED_OFF,
    UNCENSORED_ONLY,
    is_uncensored,
    model_is_uncensored,
    uncensored_mode,
)


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="Test GPU",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=900.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


def _model(model_id: str, downloads: int = 1000, **kwargs) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id.lower(),
        name=model_id.split("/", 1)[1],
        parameter_count=8_000_000_000,
        downloads=downloads,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="m-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_700_000_000,
            )
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "huihui-ai/Qwen3-8B-abliterated",
        "mlabonne/Llama-3.1-8B-Instruct-abliterated",
        "p-e-w/gemma-3-12b-it-heretic",
        "Orenguteng/Llama-3-8B-Lexi-Uncensored",
        "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        "someone/Mistral-7B-Unfiltered",
        "Goekdeniz-Guelmez/Josiefied-Qwen3-8B",
    ],
)
def test_detects_uncensored_names(model_id):
    assert is_uncensored(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-3-12b-it",
        # The org name does not count, only the repository name.
        "uncensored-labs/Plain-8B",
        "someone/flexible-8B",
        "",
    ],
)
def test_ignores_ordinary_names(model_id):
    assert not is_uncensored(model_id)


def test_tag_and_base_model_count():
    assert is_uncensored("org/Clean-8B", tags=("text-generation", "uncensored"))
    assert is_uncensored("org/Clean-8B", tags=("abliterated",))
    # A converter's clean name still quantizes an abliterated repo.
    assert is_uncensored("bartowski/Clean-8B-GGUF", base_model="x/Clean-8B-abliterated")
    assert not is_uncensored("org/Clean-8B", tags=("text-generation",))


def test_model_is_uncensored_reads_catalog_fields():
    model = _model(
        "mradermacher/Thing-8B-GGUF",
        base_model="huihui-ai/Thing-8B-abliterated",
        base_model_relation="quantized",
    )
    assert model_is_uncensored(model)
    assert not model_is_uncensored(_model("Qwen/Qwen3-8B"))


def test_mode_from_flags():
    assert uncensored_mode(False, False) == UNCENSORED_OFF
    assert uncensored_mode(True, False) == UNCENSORED_INCLUDE
    assert uncensored_mode(False, True) == UNCENSORED_ONLY
    assert uncensored_mode(True, True) == UNCENSORED_ONLY


def test_cli_flags_are_mutually_exclusive():
    with pytest.raises(typer.Exit):
        _resolve_uncensored_mode(True, True)
    assert _resolve_uncensored_mode(False, True) == UNCENSORED_ONLY
    assert _resolve_uncensored_mode(True, False) == UNCENSORED_INCLUDE
    assert _resolve_uncensored_mode(False, False) == UNCENSORED_OFF


def test_penalty_default_and_exemption():
    abliterated = "huihui-ai/Qwen3-8B-abliterated"
    assert _derivative_name_penalty(abliterated) == -10.0
    assert _derivative_name_penalty(abliterated, exempt_uncensored=True) == 0.0
    # Roleplay and nsfw derivatives are not about refusals: still penalized.
    assert _derivative_name_penalty("x/Foo-8B-RP", exempt_uncensored=True) == -10.0
    assert _derivative_name_penalty("x/Foo-8B-nsfw", exempt_uncensored=True) == -10.0
    # Mixed: uncensored AND roleplay keeps the roleplay penalty.
    assert (
        _derivative_name_penalty("x/Foo-8B-uncensored-rp", exempt_uncensored=True)
        == -10.0
    )


def _catalog() -> tuple[list[ModelInfo], dict[str, float]]:
    stock = _model("Qwen/Qwen3-8B", downloads=5_000_000)
    abliterated = _model(
        "huihui-ai/Qwen3-8B-abliterated",
        downloads=50_000,
        base_model="Qwen/Qwen3-8B",
        base_model_relation="finetune",
    )
    scores = {"Qwen/Qwen3-8B": 70.0, "huihui-ai/Qwen3-8B-abliterated": 70.0}
    return [stock, abliterated], scores


def _score(results, model_id):
    return next(r for r in results if r.model.id == model_id).quality_score


def test_off_keeps_the_penalty_and_marks_rows():
    models, scores = _catalog()
    results = rank_models(models, _hardware(), top_n=10, benchmark_scores=scores)
    ids = [r.model.id for r in results]
    assert "huihui-ai/Qwen3-8B-abliterated" in ids
    assert "Qwen/Qwen3-8B" in ids
    flags = {r.model.id: r.uncensored for r in results}
    assert flags["huihui-ai/Qwen3-8B-abliterated"] is True
    assert flags["Qwen/Qwen3-8B"] is False


def test_include_removes_the_penalty():
    models, scores = _catalog()
    off = rank_models(
        models, _hardware(), top_n=10, benchmark_scores=scores, uncensored="off"
    )
    inc = rank_models(
        models, _hardware(), top_n=10, benchmark_scores=scores, uncensored="include"
    )
    abl = "huihui-ai/Qwen3-8B-abliterated"
    assert _score(inc, abl) == pytest.approx(_score(off, abl) + 10.0)
    # The stock model is untouched.
    assert _score(inc, "Qwen/Qwen3-8B") == pytest.approx(_score(off, "Qwen/Qwen3-8B"))


def test_only_lists_nothing_but_uncensored():
    models, scores = _catalog()
    results = rank_models(
        models, _hardware(), top_n=10, benchmark_scores=scores, uncensored="only"
    )
    assert [r.model.id for r in results] == ["huihui-ai/Qwen3-8B-abliterated"]
    assert all(r.uncensored for r in results)


def test_only_keeps_quantized_uploads_of_uncensored_models():
    models, scores = _catalog()
    quant = _model(
        "mradermacher/Qwen3-8B-i1-GGUF",
        base_model="huihui-ai/Qwen3-8B-abliterated",
        base_model_relation="quantized",
    )
    results = rank_models(
        [*models, quant],
        _hardware(),
        top_n=10,
        benchmark_scores=scores,
        uncensored="only",
    )
    assert results
    assert all(r.uncensored for r in results)
    assert not any(r.model.id == "Qwen/Qwen3-8B" for r in results)


def test_only_with_no_uncensored_models_is_empty():
    models, scores = _catalog()
    results = rank_models(
        models[:1], _hardware(), top_n=10, benchmark_scores=scores, uncensored="only"
    )
    assert results == []


def _invoke_main(monkeypatch, args, fetched=()):
    """Run the main command with the ranking stubbed; return what it received."""
    from typer.testing import CliRunner

    from whichllm.cli import app

    captured: dict = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["models"] = list(models)
        captured["uncensored"] = kwargs.get("uncensored")
        return []

    async def fake_fetch_uncensored(*_args, **_kwargs):
        return list(fetched)

    monkeypatch.setattr("whichllm.hardware.detector.detect_hardware", _hardware)
    monkeypatch.setattr("whichllm.models.index_client.load_index", lambda **_: None)
    monkeypatch.setattr("whichllm.models.cache.load_cache", lambda **_kwargs: [])
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.models.hf.fetch_uncensored_models", fake_fetch_uncensored
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichllm.output.display.display_ranking", lambda results, **kwargs: None
    )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    return captured


def test_main_default_is_off_and_does_not_search(monkeypatch):
    extra = _model("huihui-ai/Qwen3-8B-abliterated")
    captured = _invoke_main(monkeypatch, [], fetched=[extra])
    assert captured["uncensored"] == UNCENSORED_OFF
    assert all(m.id != extra.id for m in captured["models"])


def test_main_uncensored_flag_passes_only_and_adds_searched_models(monkeypatch):
    extra = _model("huihui-ai/Qwen3-8B-abliterated")
    captured = _invoke_main(monkeypatch, ["--uncensored"], fetched=[extra])
    assert captured["uncensored"] == UNCENSORED_ONLY
    assert extra.id in {m.id for m in captured["models"]}


def test_main_include_uncensored_flag_passes_include(monkeypatch):
    extra = _model("p-e-w/gemma-3-12b-it-heretic")
    captured = _invoke_main(monkeypatch, ["--include-uncensored"], fetched=[extra])
    assert captured["uncensored"] == UNCENSORED_INCLUDE
    assert extra.id in {m.id for m in captured["models"]}


def test_search_does_not_duplicate_models_the_catalog_has(monkeypatch):
    from whichllm.commands.recommend import _with_uncensored_models

    have = _model("huihui-ai/Qwen3-8B-abliterated")

    async def fake_fetch(*_args, **_kwargs):
        return [_model("huihui-ai/Qwen3-8B-abliterated"), _model("x/New-8B-heretic")]

    monkeypatch.setattr("whichllm.models.hf.fetch_uncensored_models", fake_fetch)
    merged = _with_uncensored_models([have])
    assert [m.id for m in merged] == [
        "huihui-ai/Qwen3-8B-abliterated",
        "x/New-8B-heretic",
    ]


def test_failed_search_keeps_the_catalog(monkeypatch):
    from whichllm.commands.recommend import _with_uncensored_models

    async def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("whichllm.models.hf.fetch_uncensored_models", boom)
    catalog = [_model("Qwen/Qwen3-8B")]
    assert _with_uncensored_models(catalog) == catalog


def test_hf_search_asks_for_every_marker_with_and_without_gguf(monkeypatch):
    import asyncio

    from whichllm.models import hf

    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    async def fake_get(client, url, params=None, **_kwargs):
        calls.append(dict(params or {}))
        return FakeResponse()

    monkeypatch.setattr(hf, "get_with_retries", fake_get)
    assert asyncio.run(hf.fetch_uncensored_models()) == []

    searched = {(c["search"], c.get("filter")) for c in calls}
    for term in ("uncensored", "abliterated", "heretic", "dolphin"):
        assert (term, None) in searched
        assert (term, "gguf") in searched
    assert all(c["pipeline_tag"] == "text-generation" for c in calls)
    assert all(c["sort"] == "downloads" for c in calls)


def test_hf_search_failure_is_reported_not_raised(monkeypatch):
    import asyncio

    import httpx

    from whichllm.models import hf

    async def boom(client, url, params=None, **_kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(hf, "get_with_retries", boom)
    report = hf.ModelFetchReport()
    assert asyncio.run(hf.fetch_uncensored_models(report=report)) == []
    assert len(report.failed) == 8
