"""Benchmark data fetching."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass

import httpx

from whichllm.utils import _cache_dir, _current_version

logger = logging.getLogger(__name__)

CACHE_DIR = _cache_dir()
BENCHMARK_CACHE = CACHE_DIR / "benchmark.json"
DEFAULT_TTL_SECONDS = 24 * 3600  # 24 hours


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Benchmark evidence with confidence.

    source values, ordered from most trusted to least:
      - "direct"        : independent leaderboard / Arena ELO hit on exact id
      - "variant"       : suffix-stripped derivative of a direct leaderboard hit
      - "base_model"    : cardData.base_model pointer to a direct hit
      - "line_interp"   : size-aware interpolation within the same model line
      - "self_reported" : evalResults reported by the uploader themselves
      - "none"          : no usable signal
    """

    score: float | None
    confidence: float
    source: str  # see above


def load_benchmark_cache() -> dict[str, float] | None:
    """Load cached benchmark scores. Returns None if expired or missing."""
    if not BENCHMARK_CACHE.exists():
        return None
    try:
        data = json.loads(BENCHMARK_CACHE.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > DEFAULT_TTL_SECONDS:
            logger.debug("Benchmark cache expired")
            return None
        return data.get("scores", {})
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Benchmark cache corrupted: {e}")
        return None


def save_benchmark_cache(scores: dict[str, float]) -> None:
    """Save benchmark scores to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"cached_at": time.time(), "scores": scores}
    BENCHMARK_CACHE.write_text(json.dumps(data, ensure_ascii=False))
    logger.debug(f"Saved {len(scores)} benchmark scores to cache")


_LINEAGE_DEMOTION_REGEX = None


def _build_lineage_regex():
    """Compile MODEL_LINEAGE_VERSIONS once into (family, [(re, idx)]) form."""
    global _LINEAGE_DEMOTION_REGEX
    if _LINEAGE_DEMOTION_REGEX is not None:
        return _LINEAGE_DEMOTION_REGEX
    from whichllm.constants import MODEL_LINEAGE_VERSIONS

    out = {}
    for family, entries in MODEL_LINEAGE_VERSIONS.items():
        compiled = [(re.compile(pat), idx) for pat, idx in entries]
        max_idx = max(idx for _, idx in entries)
        out[family] = (compiled, max_idx)
    _LINEAGE_DEMOTION_REGEX = out
    return out


def _lineage_recency_factor(model_id: str) -> float:
    """Return a multiplicative recency factor for frozen-only scores.

    Newest generation in a known family → 1.0 (no demotion). Each generation
    older → another 12% off. Unknown families → 1.0.
    """
    if not model_id:
        return 1.0
    lower = model_id.lower()
    families = _build_lineage_regex()
    best_factor = 1.0
    for family, (patterns, max_idx) in families.items():
        for regex, idx in patterns:
            if regex.search(lower):
                gens_old = max(0, max_idx - idx)
                factor = max(0.55, 1.0 - 0.12 * gens_old)
                if factor < best_factor:
                    best_factor = factor
                break  # one family per id
    return best_factor


def _apply_lineage_recency_demotion(
    combined: dict[str, float],
    frozen: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    """Multiply frozen-only entries by a lineage-derived recency factor.

    A score is "frozen-only" when no current source (AA Index / LiveBench /
    Aider) provided a value for that id. Models with current coverage are
    left alone — their score already reflects 2026 evaluation methodology.
    """
    if not combined:
        return combined
    out: dict[str, float] = {}
    for k, v in combined.items():
        if k in current:
            out[k] = v
            continue
        factor = _lineage_recency_factor(k)
        out[k] = round(v * factor, 1)
    return out


async def fetch_benchmark_scores() -> dict[str, float]:
    """Fetch and combine benchmark scores from multiple sources.

    Sources, merged in this order (later overwrites earlier on conflict):
      1. Open LLM Leaderboard v2 (archived 2025-06, broad legacy coverage)
      2. Chatbot Arena ELO (frozen 2025-07-17, but still useful older signal)
      3. LiveBench (monthly refresh, current generation)
      4. Aider polyglot (coding-specific, current generation)
      5. Artificial Analysis Intelligence Index (covers DeepSeek V4, GLM-5,
         Kimi K2.6, MiMo V2.5, Qwen3.6 — fills the Arena/Leaderboard gap)

    Returns dict mapping model_id -> normalized score (0-100). All sources
    are fetched concurrently; any source that fails is logged and skipped,
    and the function never raises.
    """
    from whichllm.models.benchmark_sources import (
        fetch_aa_index_scores,
        fetch_aider_polyglot_scores,
        fetch_arena_scores,
        fetch_leaderboard_with_fallback,
        fetch_vision_scores,
        get_aa_curated_fallback,
        get_livebench_data,
    )

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        client.headers["User-Agent"] = f"whichllm/{_current_version()}"
        leaderboard_task = asyncio.create_task(fetch_leaderboard_with_fallback(client))
        arena_task = asyncio.create_task(fetch_arena_scores(client))
        aa_task = asyncio.create_task(fetch_aa_index_scores(client))
        aider_task = asyncio.create_task(fetch_aider_polyglot_scores(client))
        vision_task = asyncio.create_task(fetch_vision_scores(client))

        (
            lb_result,
            arena_result,
            aa_result,
            aider_result,
            vision_result,
        ) = await asyncio.gather(
            leaderboard_task,
            arena_task,
            aa_task,
            aider_task,
            vision_task,
            return_exceptions=True,
        )

    # Layered merge: build a "current" dict from live sources (AA, LiveBench,
    # Aider) and a "frozen" dict from archived sources (OLLB v2, Arena). The
    # current dict OVERRIDES the frozen one per-model — so a 2024-era model
    # with a stale-but-high OLLB number cannot beat a 2026 model that AA or
    # LiveBench measure as merely mid-tier. Frozen scores still cover the
    # long tail of older models that no live source tracks.
    frozen: dict[str, float] = {}
    current: dict[str, float] = {}

    # Frozen tier #1: Open LLM Leaderboard v2 (archived 2025-06)
    if isinstance(lb_result, BaseException):
        logger.warning(f"Leaderboard fetch failed: {lb_result}")
    else:
        frozen.update(lb_result)
        logger.debug(f"Leaderboard: {len(lb_result)} scores (frozen)")

    # Frozen tier #2: Chatbot Arena ELO (frozen 2025-07-17)
    if isinstance(arena_result, BaseException):
        logger.warning(f"Arena fetch failed, using fallback: {arena_result}")
    else:
        for k, v in arena_result.items():
            if frozen.get(k, 0.0) < v:
                frozen[k] = v
        logger.debug(f"Arena: {len(arena_result)} scores (frozen)")

    # Current tier: LiveBench (vendored snapshot)
    livebench_result = get_livebench_data()
    for k, v in livebench_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"LiveBench: {len(livebench_result)} scores (current)")

    # Current tier: Artificial Analysis Intelligence Index (~weekly refresh)
    if isinstance(aa_result, BaseException):
        logger.warning(f"AA Index fetch failed, will use fallback: {aa_result}")
        aa_result = get_aa_curated_fallback()

    for k, v in aa_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"AA Index: {len(aa_result)} scores (current)")

    # Current tier: Aider polyglot (coding-specific). Treat as a current
    # source but soft-merged — coding is one axis of capability, so a high
    # Aider score is informative but shouldn't unilaterally dethrone a
    # weaker-coding-but-strong-general AA result.
    if isinstance(aider_result, BaseException):
        logger.warning(f"Aider fetch failed: {aider_result}")
    else:
        for k, v in aider_result.items():
            if current.get(k, 0.0) < v * 0.85:
                current[k] = v * 0.85
        logger.debug(f"Aider polyglot: {len(aider_result)} scores (current, 0.85x)")

    # Current tier: vision-language capability index. Text leaderboards
    # don't score VLMs, so without this the only VLM with a direct hit
    # was a two-generations-old Qwen2-VL-7B. Profile filtering ensures
    # these scores only affect ``--profile vision`` rankings.
    if isinstance(vision_result, BaseException):
        logger.warning(f"Vision fetch failed: {vision_result}")
    else:
        for k, v in vision_result.items():
            if current.get(k, 0.0) < v:
                current[k] = v
        logger.debug(f"Vision: {len(vision_result)} scores (current)")

    # Build combined: current overrides frozen entry-by-entry, but frozen still
    # contributes for any id no current source has tracked.
    combined: dict[str, float] = dict(frozen)
    combined.update(current)

    # Apply lineage-aware demotion to frozen-only scores. Without this, models
    # that have no live coverage (e.g. Qwen2.5-72B-Instruct, Llama-3.1-70B
    # — both 2024 releases) retain their generous frozen leaderboard score
    # while their *newer* siblings (Qwen3-32B, Llama-3.3-70B) get held to
    # the live AA/LiveBench numbers. The result was older-generation 70B+
    # models ranking *above* the current-gen frontier on H100 / M2 Ultra.
    # Demote frozen-only entries from non-newest generations of known
    # families so the staleness penalty is uniform.
    combined = _apply_lineage_recency_demotion(combined, frozen, current)

    logger.debug(f"Combined: {len(combined)} benchmark scores")
    return combined


def _extract_params_b_from_id(model_id: str) -> float | None:
    """Extract parameter size in billions from model ID text."""
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        return max(float(v) for v in matches)
    except ValueError:
        return None


def _extract_model_lines(model_id: str) -> list[str]:
    """Extract model line candidates from a model ID (most specific first).

    E.g.:
        Qwen/Qwen3.5-27B -> [qwen/qwen3.5, qwen/qwen3]
        Qwen/Qwen3-32B -> [qwen/qwen3]
        Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -> [qwen/qwen3]
        meta-llama/Llama-3.3-70B-Instruct -> [meta-llama/llama-3.3, meta-llama/llama-3]
        google/gemma-3-27b-it -> [google/gemma-3]
        deepseek-ai/DeepSeek-V3.2 -> [deepseek-ai/deepseek-v3.2, deepseek-ai/deepseek-v3]
    """
    if "/" not in model_id:
        return []
    lower = model_id.lower()

    # Pre-strip repo/quant suffixes and date codes before line extraction
    stripped = re.sub(r"-(gguf|awq|gptq|fp8|fp16|bf16|nvfp4)$", "", lower)
    stripped = re.sub(r"-\d{4}(-hf)?$", "", stripped)  # date suffixes like -2507

    lines: list[str] = []

    # Remove size suffix: -32b, -70b, -0.6b, -235b-a22b, etc.
    # Allows trailing -instruct, -chat, -it, -base, -thinking, and arbitrary suffixes
    cleaned = re.sub(
        r"-\d+(\.\d+)?b(-a\d+b)?(-[a-z][-a-z0-9]*)*$",
        "",
        stripped,
    )
    if cleaned != stripped and "/" in cleaned:
        lines.append(cleaned)

    # Also strip minor version: qwen3.5 -> qwen3, llama-3.3 -> llama-3, v3.2 -> v3
    for line in list(lines) + ([stripped] if not lines else []):
        broader = re.sub(r"(\d+)\.\d+$", r"\1", line)
        if broader != line and broader not in lines:
            lines.append(broader)

    return lines


def _interpolate_line_score(
    bucket: list[tuple[float | None, float]],
    params_b: float | None,
) -> tuple[float, float]:
    """Interpolate score from same-model-line benchmarks with confidence."""
    if not bucket:
        return 0.0, 0.0

    valid = [(p, s) for p, s in bucket if p is not None]
    if not valid:
        vals = [s for _, s in bucket]
        return statistics.median(vals), 0.25

    if params_b is None or params_b <= 0:
        vals = [s for _, s in valid]
        return statistics.median(vals), 0.30

    weighted: list[tuple[float, float, float]] = []
    for p, s in valid:
        assert p is not None
        dist = abs(math.log2(max(params_b, 0.1) / max(p, 0.1)))
        w = 1.0 / (0.35 + dist)
        weighted.append((w, s, dist))

    score = sum(w * s for w, s, _ in weighted) / sum(w for w, _, _ in weighted)
    nearest = min(d for _, _, d in weighted)
    if nearest <= 0.15:
        conf = 0.45
    elif nearest <= 0.50:
        conf = 0.34
    else:
        conf = 0.26
    return score, conf


def build_score_index(
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build lookup indices from benchmark scores.

    Returns:
        (case_insensitive_index, line_index)
        - case_insensitive_index: lowercased model_id -> best score
        - line_index: model_line -> best score among all models in that line
    """
    ci_index: dict[str, float] = {}
    line_index: dict[str, float] = {}

    for key, val in scores.items():
        lk = key.lower()
        if lk not in ci_index or val > ci_index[lk]:
            ci_index[lk] = val

        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            # No size suffix (e.g., DeepSeek-V3, DeepSeek-R1) → use ID as its own line
            lines = [lk]
        for line in lines:
            if line not in line_index or val > line_index[line]:
                line_index[line] = val

    return ci_index, line_index


def build_line_bucket_index(
    scores: dict[str, float],
) -> dict[str, list[tuple[float | None, float]]]:
    """Build line -> [(params_b, score)] index for size-aware interpolation."""
    buckets: dict[str, list[tuple[float | None, float]]] = {}
    for key, val in scores.items():
        params_b = _extract_params_b_from_id(key)
        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            lines = [key.lower()]
        for line in lines:
            buckets.setdefault(line, []).append((params_b, val))
    return buckets


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

    # Strip common GGUF/quant repo suffixes
    for suffix in _REPO_SUFFIXES:
        if model_id.endswith(suffix):
            candidates.append(model_id[: -len(suffix)])
            break

    # Try adding/removing -Instruct suffix
    base = candidates[-1]  # use suffix-stripped version
    if base.endswith("-Instruct"):
        candidates.append(base[: -len("-Instruct")])
    else:
        candidates.append(base + "-Instruct")

    return candidates


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
    """Reject benchmark inheritance when the actual model size differs sharply
    from the size implied by a reference id. Catches cases like a 6.6B
    "imatrix-aligned" / draft / MTP head being matched to its 158B base model.

    Returns True if no actual size is provided (no check possible) or if
    ratio(actual, ref) stays inside [0.5, 2.0]. The window is wide enough
    that legitimate quantizations of the same model are unaffected.
    """
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
    """Look up benchmark evidence with confidence.

    Resolution order:
      direct (leaderboard) → variant → base_model → line_interp → self_reported

    self_reported_score should be the uploader-provided evalResults score from
    the model card. It is the lowest-trust source and is only returned when
    every leaderboard/inheritance path fails.

    actual_params_b: actual parameter count in billions. When provided, the
    function refuses to inherit from base_model/variant ids whose implied
    size is more than 2x off from actual (e.g. a 6.6B "imatrix-aligned"
    inheriting from a 158B base would be rejected).
    """
    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)
    if line_bucket_index is None:
        line_bucket_index = build_line_bucket_index(scores)

    # Only exact model_id match in an independent leaderboard is considered
    # direct evidence. Self-reported evalResults are handled at the very end.
    direct_result = _try_lookup(model_id, scores, ci_index)
    if direct_result is not None:
        return BenchmarkEvidence(score=direct_result, confidence=1.0, source="direct")

    # Try model_id-derived variants (inherited)
    for candidate in _generate_candidates(model_id)[1:]:
        result = _try_lookup(candidate, scores, ci_index)
        if result is not None:
            if not _params_compatible(actual_params_b, candidate):
                continue
            return BenchmarkEvidence(score=result, confidence=0.55, source="variant")

    # Try base_model and its variants
    if base_model:
        for candidate in _generate_candidates(base_model):
            result = _try_lookup(candidate, scores, ci_index)
            if result is not None:
                if not _params_compatible(actual_params_b, candidate):
                    continue
                return BenchmarkEvidence(
                    score=result, confidence=0.60, source="base_model"
                )

    # Fallback: size-aware interpolation within model line.
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

    # Last resort: uploader-reported eval. Anyone can write any number here so
    # we keep confidence low and require downstream consumers to weight this
    # source separately.
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
