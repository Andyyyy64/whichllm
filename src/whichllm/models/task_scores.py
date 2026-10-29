"""Task scores: coding, math and vision ability on the general score's scale.

A task profile should rank on evidence for that task, not on the general
index plus a name match. The evidence is per-benchmark results (fractions
solved), which are on their own scales and cover different sets of models.
Each benchmark is put on the general 0-100 scale by matching its mean and
spread to the general scores of the models it shares with them, so a model
that codes better than its general score predicts ranks higher under
``--profile coding``, and one without coding results keeps its general score.

A model with two or more of a profile's benchmarks gets their mean. With one,
that result is averaged with its general score, because a single benchmark
is noisy.

The "general" set does the same for models the general index has not scored
yet. AA's leaderboard lists new releases days after they appear, but its
Intelligence Index changed scale (v5) and is published for only a few open
models, so a new model's general score comes from the broad benchmarks AA
runs on everything instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import fmean, pstdev

# Benchmarks per profile. The vision index is the curated 0-100 snapshot in
# benchmark_sources/vision.py; the rest are Artificial Analysis benchmarks.
TASK_EVALS: dict[str, tuple[str, ...]] = {
    "general": (
        "gpqa-diamond",
        "humanitys-last-exam",
        "ifbench",
        "artificial-analysis-long-context-reasoning",
        "tau2-bench",
        "terminalbench-2-1",
        "scicode",
    ),
    "coding": (
        "terminalbench-2-1",
        "scicode",
        "livecodebench",
        "frontiercode",
        "terminalbench-hard",
    ),
    "math": ("aime-2025",),
    "vision": ("mmmu-pro", "vision-index"),
}
AA_TASK_EVALS = frozenset(
    e for evals in TASK_EVALS.values() for e in evals if e != "vision-index"
)
# Profiles whose score needs at least this many benchmarks. A general score
# fills a gap in the general index, so it is never steadied by one.
MIN_EVALS: dict[str, int] = {"general": 3}

# A benchmark needs this many models with a measured general score in common
# with the general index before it can be mapped onto that scale.
MIN_OVERLAP = 8

# Tokens that name a release's packaging rather than the model itself.
_PACKAGING_TOKENS = frozenset(
    {
        "instruct",
        "it",
        "chat",
        "hf",
        "gguf",
        "awq",
        "gptq",
        "fp8",
        "fp16",
        "bf16",
        "mlx",
        "nvfp4",
        "int4",
        "int8",
    }
)
_PAREN_RE = re.compile(r"\([^)]*\)")
_SPLIT_RE = re.compile(r"[\s\-_/]+")


def name_tokens(name: str) -> frozenset[str]:
    """Order-free tokens of a model name, without packaging words.

    ``"Llama 3.3 Instruct 70B"`` and ``"Llama-3.3-70B-Instruct"`` give the
    same set, so AA display names can be matched to repo names.
    """
    name = _PAREN_RE.sub(" ", name.lower())
    return frozenset(
        t for t in _SPLIT_RE.split(name) if t and t not in _PACKAGING_TOKENS
    )


def match_names_to_ids(
    names: Iterable[str], model_ids: Iterable[str]
) -> dict[str, list[str]]:
    """Repo ids whose name has exactly the same tokens as each display name.

    When a release's own repo is among the matches, re-uploads (``-GGUF``,
    ``-FP8``, ...) are dropped: lookups reach them through the original's name.
    """
    by_tokens: dict[frozenset[str], list[str]] = {}
    for model_id in dict.fromkeys(model_ids):
        tokens = name_tokens(model_id.rsplit("/", 1)[-1])
        if tokens:
            by_tokens.setdefault(tokens, []).append(model_id)
    out: dict[str, list[str]] = {}
    for name in names:
        ids = by_tokens.get(name_tokens(name))
        if not ids:
            continue
        plain = [i for i in ids if not _has_packaging(i)]
        out[name] = sorted(plain or ids)
    return out


def _has_packaging(model_id: str) -> bool:
    name = _PAREN_RE.sub(" ", model_id.rsplit("/", 1)[-1].lower())
    return any(
        t in _PACKAGING_TOKENS - {"instruct", "it", "chat"}
        for t in _SPLIT_RE.split(name)
    )


def _equating(
    raw: dict[str, float], anchor: dict[str, float]
) -> tuple[float, float, float, float] | None:
    shared = [k for k in raw if k in anchor]
    if len(shared) < MIN_OVERLAP:
        return None
    xs = [raw[k] for k in shared]
    ys = [anchor[k] for k in shared]
    sd_x = pstdev(xs)
    if sd_x <= 0:
        return None
    return fmean(xs), sd_x, fmean(ys), pstdev(ys)


def build_task_scores(
    evals_by_id: dict[str, dict[str, float]],
    anchor: dict[str, float],
    general: dict[str, float] | None = None,
    profiles: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """``{profile: {model id: 0-100}}`` from per-benchmark results.

    ``anchor`` holds measured general scores (no hand estimates) and sets the
    scale; ``general`` is the full general index, used to steady a model
    that has a single benchmark for a profile.
    """
    general = anchor if general is None else general
    per_eval: dict[str, dict[str, float]] = {}
    for model_id, results in evals_by_id.items():
        for eval_id, value in results.items():
            per_eval.setdefault(eval_id, {})[model_id] = value

    scores: dict[str, dict[str, float]] = {}
    for profile in TASK_EVALS if profiles is None else profiles:
        evals = TASK_EVALS[profile]
        min_evals = MIN_EVALS.get(profile, 1)
        maps = {}
        for eval_id in evals:
            fit = _equating(per_eval.get(eval_id, {}), anchor)
            if fit is not None:
                maps[eval_id] = fit
        means = _combine(evals_by_id, evals, maps, min_evals)
        # A benchmark that shares too few models with the general index (the
        # curated vision index covers VLMs that have no text benchmark) is
        # mapped through the profile's other benchmarks instead.
        # Failing that, through the whole general index, which includes the
        # frozen leaderboards that still score older VLMs.
        missing = [e for e in evals if e not in maps and e in per_eval]
        if missing:
            provisional = {k: v for k, (v, _) in means.items()}
            for extended in (
                {**provisional, **anchor},
                {**general, **provisional, **anchor},
            ):
                for eval_id in [e for e in missing if e not in maps]:
                    fit = _equating(per_eval[eval_id], extended)
                    if fit is not None:
                        maps[eval_id] = fit
            means = _combine(evals_by_id, evals, maps, min_evals)
        # Averaging several benchmarks narrows the spread (their noise cancels),
        # which would pull strong models toward the middle. Put the averages
        # themselves back on the general scale.
        fit = _equating({k: v for k, (v, _) in means.items()}, anchor)
        out: dict[str, float] = {}
        for model_id, (score, count) in means.items():
            if fit is not None:
                mean_x, sd_x, mean_y, sd_y = fit
                score = mean_y + sd_y * (score - mean_x) / sd_x
            if count == 1 and model_id in general:
                score = (score + general[model_id]) / 2
            out[model_id] = round(max(0.0, min(100.0, score)), 1)
        if out:
            scores[profile] = dict(sorted(out.items()))
    return scores


def _combine(
    evals_by_id: dict[str, dict[str, float]],
    evals: tuple[str, ...],
    maps: dict[str, tuple[float, float, float, float]],
    min_evals: int = 1,
) -> dict[str, tuple[float, int]]:
    """Mean of each model's mapped results, and how many there were."""
    out: dict[str, tuple[float, int]] = {}
    for model_id, results in evals_by_id.items():
        values = []
        for eval_id in evals:
            if eval_id in results and eval_id in maps:
                mean_x, sd_x, mean_y, sd_y = maps[eval_id]
                values.append(mean_y + sd_y * (results[eval_id] - mean_x) / sd_x)
        if values and len(values) >= min_evals:
            out[model_id] = (fmean(values), len(values))
    return out
