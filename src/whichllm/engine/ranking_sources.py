"""Source trust constants used by ranking modules."""

from __future__ import annotations

_OFFICIAL_ORGS = frozenset(
    {
        "Qwen",
        "meta-llama",
        "google",
        "mistralai",
        "deepseek-ai",
        "microsoft",
        "nvidia",
        "01-ai",
        "tiiuae",
        "apple",
        "CohereForAI",
        "bigcode",
        # 2025+ frontier open-weights labs that publish safetensors-only
        # repos which the community immediately converts to GGUF.
        "openai",
        "zai-org",
        "moonshotai",
        "MiniMaxAI",
        "XiaomiMiMo",
        "allenai",
        "ibm-granite",
        "stepfun-ai",
    }
)

# Trusted GGUF converters — format converters that don't change model quality
_TRUSTED_CONVERTERS = frozenset(
    {
        "bartowski",
        "lmstudio-community",
        "QuantFactory",
        "unsloth",
        "ggml-org",
        "Mungert",
    }
)

# Known repackagers — typically reupload others' models without added value
_REPACKAGER_ORGS = frozenset(
    {
        "MaziyarPanahi",
        "TheBloke",
        "SanctumAI",
        "solidrust",
        "mradermacher",
    }
)

__all__ = [
    "_OFFICIAL_ORGS",
    "_REPACKAGER_ORGS",
    "_TRUSTED_CONVERTERS",
]
