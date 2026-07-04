"""Benchmark evidence lookup and inheritance rules."""

from __future__ import annotations

from whichllm.models.benchmark_index import (
    _extract_model_lines,
    _extract_params_b_from_id,
    _interpolate_line_score,
    build_line_bucket_index,
    build_score_index,
)
from whichllm.models.benchmark_types import BenchmarkEvidence


def _try_lookup(
    candidate: str, scores: dict[str, float], ci_index: dict[str, float]
) -> float | None:
    """Try exact match, then case-insensitive match."""
    if candidate in scores:
        return scores[candidate]
    lc = candidate.lower()
    if lc in ci_index:
        return ci_index[lc]
    return None


_REPO_SUFFIXES = ("-GGUF", "-gguf", "-AWQ", "-GPTQ", "-FP8", "-fp8", "-BF16", "-bf16")


def _generate_candidates(model_id: str) -> list[str]:
    """Generate candidate IDs to look up for a model."""
    candidates = [model_id]

    for suffix in _REPO_SUFFIXES:
        if model_id.endswith(suffix):
            candidates.append(model_id[: -len(suffix)])
            break

    base = candidates[-1]
    if base.endswith("-Instruct"):
        candidates.append(base[: -len("-Instruct")])
    else:
        candidates.append(base + "-Instruct")

    return candidates


def _append_unique(candidates: list[str], candidate: str) -> None:
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def _strip_repo_suffix(model_id: str) -> str:
    for suffix in _REPO_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _generate_score_name_candidates(
    model_id: str, scores: dict[str, float]
) -> list[str]:
    """Match community repo names to benchmark IDs with the same model name."""
    stripped = _strip_repo_suffix(model_id)
    repo_name = stripped.rsplit("/", 1)[-1]
    model_names = [repo_name]

    explicit_candidates: list[str] = []
    if "_" in repo_name:
        org, name = repo_name.split("_", 1)
        if org and name:
            _append_unique(explicit_candidates, f"{org}/{name}")
            _append_unique(model_names, name)

    score_candidates: list[str] = []
    wanted_names = {name.lower() for name in model_names if name}
    for score_id in scores:
        score_name = score_id.rsplit("/", 1)[-1].lower()
        if score_name in wanted_names:
            _append_unique(score_candidates, score_id)

    return explicit_candidates + [
        candidate
        for candidate in score_candidates
        if candidate not in explicit_candidates
    ]


def lookup_benchmark(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
) -> tuple[float, bool] | None:
    """Backward-compatible benchmark lookup helper."""
    evidence = lookup_benchmark_evidence(
        model_id,
        base_model,
        scores,
        ci_index=ci_index,
        line_index=line_index,
    )
    if evidence.score is None:
        return None
    return evidence.score, evidence.source == "direct"


def _params_compatible(actual_b: float | None, ref_id: str) -> bool:
    """Reject benchmark inheritance when actual and reference sizes diverge."""
    if actual_b is None or actual_b <= 0:
        return True
    ref_b = _extract_params_b_from_id(ref_id)
    if ref_b is None or ref_b <= 0:
        return True
    ratio = actual_b / ref_b
    return 0.5 <= ratio <= 2.0


def lookup_benchmark_evidence(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
    line_bucket_index: dict[str, list[tuple[float | None, float]]] | None = None,
    self_reported_score: float | None = None,
    actual_params_b: float | None = None,
) -> BenchmarkEvidence:
    """Look up benchmark evidence with confidence."""
    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)
    if line_bucket_index is None:
        line_bucket_index = build_line_bucket_index(scores)

    direct_result = _try_lookup(model_id, scores, ci_index)
    if direct_result is not None:
        return BenchmarkEvidence(score=direct_result, confidence=1.0, source="direct")

    variant_candidates = _generate_candidates(model_id)[1:]
    for candidate in _generate_score_name_candidates(model_id, scores):
        _append_unique(variant_candidates, candidate)
    for candidate in variant_candidates:
        result = _try_lookup(candidate, scores, ci_index)
        if result is not None:
            if not _params_compatible(actual_params_b, candidate):
                continue
            return BenchmarkEvidence(score=result, confidence=0.55, source="variant")

    if base_model:
        for candidate in _generate_candidates(base_model):
            result = _try_lookup(candidate, scores, ci_index)
            if result is not None:
                if not _params_compatible(actual_params_b, candidate):
                    continue
                return BenchmarkEvidence(
                    score=result, confidence=0.60, source="base_model"
                )

    size_hint = (
        actual_params_b
        or _extract_params_b_from_id(model_id)
        or _extract_params_b_from_id(base_model or "")
    )
    for mid in (model_id, base_model):
        if mid:
            for line in _extract_model_lines(mid):
                if line in line_bucket_index:
                    score, conf = _interpolate_line_score(
                        line_bucket_index[line], size_hint
                    )
                    if score > 0:
                        return BenchmarkEvidence(
                            score=score, confidence=conf, source="line_interp"
                        )
                if line in line_index:
                    return BenchmarkEvidence(
                        score=line_index[line], confidence=0.22, source="line_interp"
                    )

    if (
        self_reported_score is not None
        and isinstance(self_reported_score, (int, float))
        and self_reported_score > 0
    ):
        return BenchmarkEvidence(
            score=float(self_reported_score),
            confidence=0.40,
            source="self_reported",
        )

    return BenchmarkEvidence(score=None, confidence=0.0, source="none")
