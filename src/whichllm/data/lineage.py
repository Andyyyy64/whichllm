"""Model lineage / generation half-order used to bonus or penalize family versions."""

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
