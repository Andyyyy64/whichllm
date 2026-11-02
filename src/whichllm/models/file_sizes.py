"""Real file sizes for the index.

The Hub's model list names a repository's files but not their sizes, so GGUF
variants and pre-quantized checkpoints are estimated from the parameter count.
That runs low for models with a large vocabulary or mixed-precision quants
(unsloth's Qwen3.5-2B Q4_K_M was estimated at 1.06 GB and is 1.28 GB) and by
more than half for NVFP4 repos, whose reported count leaves out most of the
packed weights (nvidia's Qwen3.6-35B-A3B-NVFP4 was estimated at 10.5 GB and is
23.4 GB). The index build reads the real sizes from the model detail endpoint
(``blobs=true``). It reuses the previous index's sizes while a repository's
commit is unchanged, and fetches at most ``MAX_SIZE_REQUESTS`` repositories per
build, most downloaded first, to stay inside the Hub's rate limit (500 API
calls per 5 minutes without a token). Once the Hub answers 429 it stops asking.
The rest keep their estimates until a later build.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

from whichllm.engine.quantization import infer_non_gguf_quant_type
from whichllm.models.gguf import _extract_gguf_variants
from whichllm.models.hf import _hf_api_url, _hf_headers
from whichllm.models.http import get_with_retries
from whichllm.models.types import ModelInfo

logger = logging.getLogger(__name__)

MAX_SIZE_REQUESTS = 300
_CONCURRENCY = 4
_PICKLE_WEIGHTS_RE = re.compile(r"pytorch_model.*\.bin")


@dataclass
class SizeReport:
    reused: int = 0
    fetched: int = 0
    failed: int = 0
    # Repositories past the per-build cap; they keep their estimates.
    pending: int = 0


def is_quantized_checkpoint(model: ModelInfo) -> bool:
    """A non-GGUF repo whose size the parameter count does not give."""
    if model.gguf_variants:
        return False
    return model.base_model_relation == "quantized" or (
        infer_non_gguf_quant_type(model.id) not in ("FP16", "BF16")
    )


def _needs_size(model: ModelInfo) -> bool:
    if model.gguf_variants:
        return any(v.is_estimated for v in model.gguf_variants)
    return is_quantized_checkpoint(model) and not model.weight_bytes


def _reuse(model: ModelInfo, prior: dict) -> bool:
    """Copy sizes from the previous index's entry for this repo, if they fit."""
    if not model.gguf_variants:
        model.weight_bytes = prior.get("weight_bytes") or None
        return model.weight_bytes is not None
    variants = prior.get("gguf_variants") or []
    if not variants or any(v.get("is_estimated") for v in variants):
        return False
    sizes = {v["filename"]: v["file_size_bytes"] for v in variants}
    if not all(v.filename in sizes for v in model.gguf_variants):
        return False
    for variant in model.gguf_variants:
        variant.file_size_bytes = sizes[variant.filename]
        variant.is_estimated = False
    return True


def _checkpoint_bytes(data: dict) -> int:
    """Size of a checkpoint's weights: root safetensors, else pickle shards.

    Mistral repos carry the weights twice, as ``consolidated.safetensors`` and
    as Hugging Face shards, so the shards win when both exist.
    """
    sizes = {
        sib.get("rfilename", ""): sib.get("size")
        for sib in data.get("siblings") or []
        if isinstance(sib.get("size"), int)
    }
    root = {name: size for name, size in sizes.items() if "/" not in name}
    safetensors = {n: s for n, s in root.items() if n.endswith(".safetensors")}
    shards = {n: s for n, s in safetensors.items() if not n.startswith("consolidated")}
    pickle = (s for n, s in root.items() if _PICKLE_WEIGHTS_RE.fullmatch(n))
    return sum((shards or safetensors).values()) or sum(pickle)


def _apply_sizes(model: ModelInfo, data: dict) -> bool:
    if model.gguf_variants:
        variants = _extract_gguf_variants(data, model.parameter_count)
        if not variants or any(v.is_estimated for v in variants):
            return False
        model.gguf_variants = variants
        return True
    model.weight_bytes = _checkpoint_bytes(data) or None
    return model.weight_bytes is not None


async def fill_file_sizes(
    models: list[ModelInfo],
    previous: dict | None = None,
    *,
    max_requests: int = MAX_SIZE_REQUESTS,
) -> SizeReport:
    """Replace estimated sizes with the files' real sizes, in place."""
    report = SizeReport()
    known = {m["id"]: m for m in (previous or {}).get("models", []) if m.get("sha")}
    todo: list[ModelInfo] = []
    for model in models:
        if not _needs_size(model):
            continue
        prior = known.get(model.id)
        if prior and model.sha == prior["sha"] and _reuse(model, prior):
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
                    logger.debug("File sizes for %s failed: %s", model.id, exc)
                    report.failed += 1
                    return
            if not _apply_sizes(model, data):
                report.failed += 1
                return
            if isinstance(data.get("sha"), str):
                model.sha = data["sha"]
            report.fetched += 1

        await asyncio.gather(*(fill(model) for model in todo))
    return report
