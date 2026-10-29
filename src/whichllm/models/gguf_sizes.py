"""Real GGUF file sizes for the index.

The Hub's model list names a repository's files but not their sizes, so each
GGUF variant's size is estimated from the parameter count. That runs low for
models with a large vocabulary or mixed-precision quants: unsloth's
Qwen3.5-2B Q4_K_M was estimated at 1.06 GB and is 1.28 GB. The index build
reads the real sizes from the model detail endpoint (``blobs=true``). It reuses
the previous index's sizes while a repository's commit is unchanged, and
fetches at most ``MAX_SIZE_REQUESTS`` repositories per build, most downloaded
first, to stay inside the Hub's rate limit (500 API calls per 5 minutes
without a token). Once the Hub answers 429 it stops asking. The rest keep
their estimates until a later build.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from whichllm.models.gguf import _extract_gguf_variants
from whichllm.models.hf import _hf_api_url, _hf_headers
from whichllm.models.http import get_with_retries
from whichllm.models.types import ModelInfo

logger = logging.getLogger(__name__)

MAX_SIZE_REQUESTS = 300
_CONCURRENCY = 4


@dataclass
class SizeReport:
    reused: int = 0
    fetched: int = 0
    failed: int = 0
    # Repositories past the per-build cap; they keep their estimates.
    pending: int = 0


def _previous_sizes(previous: dict | None) -> dict[str, tuple[str, dict[str, int]]]:
    """``{repo: (sha, {filename: bytes})}`` for repos with real sizes."""
    known = {}
    for model in (previous or {}).get("models", []):
        variants = model.get("gguf_variants") or []
        sha = model.get("sha")
        if sha and variants and not any(v.get("is_estimated") for v in variants):
            known[model["id"]] = (
                sha,
                {v["filename"]: v["file_size_bytes"] for v in variants},
            )
    return known


def _reuse(model: ModelInfo, sizes: dict[str, int]) -> bool:
    if not all(v.filename in sizes for v in model.gguf_variants):
        return False
    for variant in model.gguf_variants:
        variant.file_size_bytes = sizes[variant.filename]
        variant.is_estimated = False
    return True


async def fill_gguf_sizes(
    models: list[ModelInfo],
    previous: dict | None = None,
    *,
    max_requests: int = MAX_SIZE_REQUESTS,
) -> SizeReport:
    """Replace estimated GGUF sizes with the files' real sizes, in place."""
    report = SizeReport()
    known = _previous_sizes(previous)
    todo: list[ModelInfo] = []
    for model in models:
        if not any(v.is_estimated for v in model.gguf_variants):
            continue
        prior = known.get(model.id)
        if prior and model.sha == prior[0] and _reuse(model, prior[1]):
            report.reused += 1
        else:
            todo.append(model)
    todo.sort(key=lambda m: -m.downloads)
    report.pending = max(0, len(todo) - max_requests)
    todo = todo[:max_requests]
    if not todo:
        return report

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    rate_limited = False
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers=_hf_headers()
    ) as client:

        async def fill(model: ModelInfo) -> None:
            nonlocal rate_limited
            async with semaphore:
                if rate_limited:
                    report.pending += 1
                    return
                try:
                    resp = await get_with_retries(
                        client,
                        _hf_api_url(f"models/{model.id}"),
                        params={"blobs": "true"},
                    )
                    if resp.status_code == 429:
                        rate_limited = True
                        report.pending += 1
                        return
                    resp.raise_for_status()
                    data = resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.debug("GGUF sizes for %s failed: %s", model.id, exc)
                    report.failed += 1
                    return
            variants = _extract_gguf_variants(data, model.parameter_count)
            if not variants or any(v.is_estimated for v in variants):
                report.failed += 1
                return
            model.gguf_variants = variants
            if isinstance(data.get("sha"), str):
                model.sha = data["sha"]
            report.fetched += 1

        await asyncio.gather(*(fill(model) for model in todo))
    return report
