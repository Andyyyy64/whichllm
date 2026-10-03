"""CLI validation, filtering, and hardware override helpers."""

from __future__ import annotations


from whichllm.cli_shared import _run_async


def _include_vision_candidates(profile: str) -> bool:
    """候補取得時にVLMを含めるべきプロファイルか判定する。"""
    return profile.lower() in {"vision", "any"}


def _fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    """上位表示で欠けている公開日時を補完し、更新有無を返す。"""
    missing_ids = [r.model.id for r in results if not r.model.published_at]
    if not missing_ids:
        return False
    published_map = _run_async(fetch_model_published_at(missing_ids))
    if not published_map:
        return False

    updated = False
    for model in all_models:
        published_at = published_map.get(model.id)
        if published_at and not model.published_at:
            model.published_at = published_at
            updated = True
    return updated


def _merge_model_eval_benchmarks(
    models: list,
    benchmark_scores: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Deprecated no-op kept for backward API compatibility.

    Previously this injected each model's uploader-reported ``hf_eval``
    value into the leaderboard scores dict under the model's id, which
    caused those values to be treated as ``direct`` benchmark evidence
    by the ranker. That elevated any account that wrote a high number
    in their model card to the top of the rankings.

    The hf_eval value is now consumed inside ``rank_models`` via
    ``BenchmarkEvidence.source == "self_reported"`` with a much lower
    weight and a dedicated display tag, so we no longer need to mutate
    the leaderboard dict here. Returning the input unchanged keeps any
    external callers working.
    """
    return benchmark_scores, 0
