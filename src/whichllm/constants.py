"""Constants for GPU bandwidth, quantization, and estimation parameters."""

_GiB = 1024**3

# GPU memory bandwidth in GB/s (theoretical peak)
# Key: substring matched against GPU name (case-insensitive)
GPU_BANDWIDTH: dict[str, float] = {
    # NVIDIA Consumer - RTX 50 series
    "RTX 5090": 1792.0,
    "RTX 5080": 960.0,
    "RTX 5070 Ti": 896.0,
    "RTX 5070": 672.0,
    "RTX 5060 Ti": 448.0,
    # NVIDIA Consumer - RTX 40 series
    "RTX 4090": 1008.0,
    "RTX 4080 SUPER": 736.0,
    "RTX 4080": 716.8,
    "RTX 4070 Ti SUPER": 672.0,
    "RTX 4070 Ti": 504.0,
    "RTX 4070 SUPER": 504.0,
    "RTX 4070": 504.0,
    "RTX 4060 Ti": 288.0,
    "RTX 4060": 272.0,
    # NVIDIA Consumer - RTX 30 series
    "RTX 3090 Ti": 1008.0,
    "RTX 3090": 936.2,
    "RTX 3080 Ti": 912.4,
    "RTX 3080": 760.3,
    "RTX 3070 Ti": 608.3,
    "RTX 3070": 448.0,
    "RTX 3060 Ti": 448.0,
    "RTX 3060": 360.0,
    # NVIDIA Consumer - RTX 20 series
    "RTX 2080 Ti": 616.0,
    "RTX 2080 SUPER": 496.0,
    "RTX 2080": 448.0,
    "RTX 2070 SUPER": 448.0,
    "RTX 2070": 448.0,
    "RTX 2060 SUPER": 448.0,
    "RTX 2060": 336.0,
    # NVIDIA Consumer - GTX 16 series
    "GTX 1660 Ti": 288.0,
    "GTX 1660 SUPER": 336.0,
    "GTX 1660": 192.0,
    "GTX 1650": 128.0,
    # NVIDIA Data Center
    "H100": 3350.0,
    "H200": 4800.0,
    "A100 80GB": 2039.0,
    "A100 40GB": 1555.0,
    "A100": 1555.0,
    "A6000": 768.0,
    "A5000": 768.0,
    "A4000": 448.0,
    "L40S": 864.0,
    "L40": 864.0,
    "L4": 300.0,
    "T4": 320.0,
    "V100": 900.0,
    "P100": 732.0,
    # AMD
    "RX 7900 XTX": 960.0,
    "RX 7900 XT": 800.0,
    "RX 7800 XT": 624.0,
    "RX 7700 XT": 432.0,
    "RX 7600": 288.0,
    "RX 6950 XT": 576.0,
    "RX 6900 XT": 512.0,
    "RX 6800 XT": 512.0,
    "RX 6800": 512.0,
    "RX 6700 XT": 384.0,
    "MI300X": 5300.0,
    "MI250X": 3276.0,
    "MI210": 1638.0,
    # Apple Silicon (unified memory bandwidth)
    "M1 Ultra": 800.0,
    "M1 Max": 400.0,
    "M1 Pro": 200.0,
    "M1": 68.25,
    "M2 Ultra": 800.0,
    "M2 Max": 400.0,
    "M2 Pro": 200.0,
    "M2": 100.0,
    "M3 Ultra": 800.0,
    "M3 Max": 400.0,
    "M3 Pro": 150.0,
    "M3": 100.0,
    "M4 Ultra": 819.2,
    "M4 Max": 546.0,
    "M4 Pro": 273.0,
    "M4": 120.0,
}

# Bytes per weight for each quantization type
QUANT_BYTES_PER_WEIGHT: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0625,
    "Q6_K": 0.8125,
    "Q5_K_M": 0.6875,
    "Q5_K_S": 0.6875,
    "Q5_0": 0.625,
    "Q4_K_M": 0.5625,
    "Q4_K_S": 0.5625,
    "Q4_0": 0.5,
    "Q3_K_M": 0.4375,
    "Q3_K_S": 0.4375,
    "Q3_K_L": 0.4375,
    "Q2_K": 0.3125,
    "IQ4_XS": 0.5,
    "IQ3_XXS": 0.375,
    "IQ2_XXS": 0.25,
    # Sub-2-bit / ternary tiers (extremely lossy)
    "Q1_0": 0.28,
    "Q2_0": 0.28,
    "TQ1_0": 0.21,
    "TQ2_0": 0.28,
    "IQ1_S": 0.21,
    "IQ1_M": 0.22,
    "IQ2_S": 0.275,
    "IQ2_M": 0.30,
    "IQ3_S": 0.40,
    "IQ3_M": 0.42,
    "IQ3_XS": 0.41,
    "IQ4_NL": 0.5,
}

# Quality penalty for each quantization type (fraction of quality lost)
# Sub-2-bit and ternary quants lose 30-60% of model quality - whichllm
# previously fell back to 5% which over-rewarded extreme quants.
QUANT_QUALITY_PENALTY: dict[str, float] = {
    "F32": 0.0,
    "F16": 0.0,
    "BF16": 0.0,
    "Q8_0": 0.01,
    "Q6_K": 0.02,
    "Q5_K_M": 0.03,
    "Q5_K_S": 0.035,
    "Q5_0": 0.035,
    "Q4_K_M": 0.05,
    "Q4_K_S": 0.055,
    "Q4_0": 0.06,
    "Q3_K_M": 0.08,
    "Q3_K_S": 0.12,
    "Q3_K_L": 0.075,
    "Q2_K": 0.25,
    "IQ4_XS": 0.05,
    "IQ4_NL": 0.055,
    "IQ3_XS": 0.16,
    "IQ3_S": 0.17,
    "IQ3_M": 0.16,
    "IQ3_XXS": 0.18,
    "IQ2_M": 0.30,
    "IQ2_S": 0.32,
    "IQ2_XXS": 0.40,
    "IQ1_M": 0.50,
    "IQ1_S": 0.55,
    "Q2_0": 0.45,
    "Q1_0": 0.55,
    "TQ2_0": 0.45,
    "TQ1_0": 0.55,
}

# Preferred quantization types ordered from best to acceptable.
# Sub-3-bit and 1-bit ternary variants sit at the tail so they are only
# selected when nothing else is available or when explicitly requested.
QUANT_PREFERENCE_ORDER = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q3_K_M",
    "Q3_K_L",
    "Q8_0",
    "IQ4_XS",
    "IQ4_NL",
    "Q4_0",
    "Q5_0",
    "Q3_K_S",
    "F16",
    "BF16",
    "IQ3_M",
    "IQ3_S",
    "IQ3_XS",
    "Q2_K",
    "IQ3_XXS",
    "IQ2_M",
    "IQ2_S",
    "IQ2_XXS",
    "IQ1_M",
    "IQ1_S",
    "Q2_0",
    "TQ2_0",
    "Q1_0",
    "TQ1_0",
]


# Generation lineage half-order.
# For each "family stem" we encode a monotone-increasing version map so that
# the ranker can apply a small bonus/penalty depending on whether a model
# represents the newest generation of its family. This avoids the situation
# where an older series with stale Open-LLM-Leaderboard data ranks above a
# newer release for which the leaderboard simply has no data yet.
#
# Each entry is a list of (regex_pattern, generation_index) tuples evaluated
# in order; first match wins. Patterns match against lowercased model_id.
# Higher index = newer.
MODEL_LINEAGE_VERSIONS: dict[str, list[tuple[str, int]]] = {
    "qwen": [
        # ordered newest -> oldest so the bonus reflects the strongest claim
        (r"qwen3\.6", 7),
        (r"qwen3\.5", 6),
        (r"qwen3-next", 6),
        (r"qwen3-coder-next", 6),
        (r"qwen3-omni", 5),
        (r"qwen3", 5),
        (r"qwq", 4),
        (r"qwen2\.5", 3),
        (r"qwen2(?!\.5)", 2),
        (r"qwen1", 1),
        (r"qwen-(7b|14b|72b)", 1),
    ],
    "llama": [
        (r"llama-?4\.5", 5),
        (r"llama-?4", 4),
        (r"llama-?3\.3", 3),
        (r"llama-?3\.2", 3),
        (r"llama-?3\.1", 3),
        (r"meta-llama-?3(?!\.)", 2),
        (r"llama-?2", 1),
    ],
    "deepseek": [
        (r"deepseek-v4", 5),
        (r"deepseek-v3\.2", 4),
        (r"deepseek-v3\.1", 4),
        (r"deepseek-r1-0528", 4),
        (r"deepseek-r1", 3),
        (r"deepseek-v3-0324", 3),
        (r"deepseek-v3(?!\.)", 3),
        (r"deepseek-v2\.5", 2),
        (r"deepseek-v2(?!\.5)", 1),
        (r"deepseek-coder-v2", 2),
        (r"deepseek-coder(?!-v2)", 1),
    ],
    "gemma": [
        (r"gemma-?4", 4),
        (r"gemma-?3", 3),
        (r"gemma-?2", 2),
        (r"gemma(?!-?[2-9])", 1),
    ],
    "phi": [
        (r"phi-?5", 5),
        (r"phi-?4", 4),
        (r"phi-?3\.5", 3),
        (r"phi-?3(?!\.5)", 2),
        (r"phi-?2", 1),
    ],
    "mistral_small": [
        (r"mistral-small-3\.2", 4),
        (r"mistral-small-2506", 4),
        (r"mistral-small-3\.1", 3),
        (r"mistral-small-3", 3),
        (r"mistral-small-2501", 3),
        (r"mistral-small.*2409", 2),
        (r"mistral-small", 1),
    ],
    "mistral_large": [
        (r"mistral-large-3", 4),
        (r"mistral-large-instruct-2411", 3),
        (r"mistral-large-2411", 3),
        (r"mistral-large-2407", 2),
        (r"mistral-large", 1),
    ],
    "mistral_7b": [
        (r"mistral-?7b-instruct-v0\.3", 3),
        (r"mistral-?7b-instruct-v0\.2", 2),
        (r"mistral-?7b-instruct-v0\.1", 1),
    ],
    "mixtral": [
        (r"mixtral-8x22b", 2),
        (r"mixtral-8x7b", 1),
    ],
    "gpt_oss": [
        (r"gpt-oss-120b", 2),
        (r"gpt-oss-20b", 2),
        (r"gpt-oss", 1),
    ],
    "glm": [
        (r"glm-?5\.1", 6),
        (r"glm-?5(?!\.)", 5),
        (r"glm-?4\.7", 4),
        (r"glm-?4\.6", 3),
        (r"glm-?4\.5", 3),
        (r"glm-?4(?!\.[5-9])", 2),
        (r"chatglm", 1),
    ],
    "kimi": [
        (r"kimi-?k2\.6", 4),
        (r"kimi-?k2\.5", 3),
        (r"kimi-?k2-thinking", 3),
        (r"kimi-?k2", 2),
        (r"kimi", 1),
    ],
    "mimo": [
        (r"mimo-?v2\.5", 3),
        (r"mimo-?v2", 2),
        (r"mimo-?7b", 1),
        (r"mimo", 1),
    ],
    "granite": [
        (r"granite-?4\.1", 5),
        (r"granite-?4", 4),
        (r"granite-?3\.[2-9]", 3),
        (r"granite-?3\.1", 2),
        (r"granite-?3\.0", 2),
        (r"granite", 1),
    ],
    "olmo": [
        (r"olmo-?3", 3),
        (r"olmo-?2", 2),
        (r"olmo(?!-?[2-9])", 1),
    ],
    "yi": [
        (r"yi-lightning", 3),
        (r"yi-1\.5", 2),
        (r"yi-(6b|9b|34b)(?!.*1\.5)", 1),
    ],
}

# Maximum bonus (in raw quality-score points) applied to the newest generation
# of a recognized family. The bonus interpolates downwards for older versions.
# These are larger than the initial pass because frozen leaderboards (OLLB v2,
# Arena 2025-07) systematically over-reward 2024-era models like Qwen2.5-32B
# that are no longer the current frontier; the lineage signal pulls newer
# releases past their older siblings even when the older one has stale-but-high
# leaderboard data.
MODEL_GENERATION_BONUS_MAX = 10.0
MODEL_GENERATION_PENALTY_MAX = 6.0

# Framework overhead in bytes (~500MB)
FRAMEWORK_OVERHEAD_BYTES = 500_000_000

# Minimum compute capability for common frameworks
MIN_COMPUTE_CAPABILITY_OLLAMA = (5, 0)
MIN_COMPUTE_CAPABILITY_VLLM = (7, 0)

# NVIDIA GPU compute capability lookup (substring match, case-insensitive)
NVIDIA_COMPUTE_CAPABILITY: dict[str, tuple[int, int]] = {
    # RTX 50 series (Blackwell)
    "RTX 5090": (10, 0),
    "RTX 5080": (10, 0),
    "RTX 5070": (10, 0),
    # RTX 40 series (Ada Lovelace)
    "RTX 4090": (8, 9),
    "RTX 4080": (8, 9),
    "RTX 4070": (8, 9),
    "RTX 4060": (8, 9),
    # RTX 30 series (Ampere)
    "RTX 3090": (8, 6),
    "RTX 3080": (8, 6),
    "RTX 3070": (8, 6),
    "RTX 3060": (8, 6),
    # RTX 20 series (Turing)
    "RTX 2080": (7, 5),
    "RTX 2070": (7, 5),
    "RTX 2060": (7, 5),
    # GTX 16 series (Turing)
    "GTX 1660": (7, 5),
    "GTX 1650": (7, 5),
    # GTX 10 series (Pascal)
    "GTX 1080": (6, 1),
    "GTX 1070": (6, 1),
    "GTX 1060": (6, 1),
    # Data Center
    "H100": (9, 0),
    "H200": (9, 0),
    "A100": (8, 0),
    "A6000": (8, 6),
    "A5000": (8, 6),
    "A4000": (8, 6),
    "L40": (8, 9),
    "L4": (8, 9),
    "T4": (7, 5),
    "V100": (7, 0),
    "P100": (6, 0),
}
