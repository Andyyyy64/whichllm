"""External benchmark sources beyond Chatbot Arena and Open LLM Leaderboard.

Each module here fetches an independent leaderboard / index, normalizes it to
the same 0-100 scale, and returns a ``dict[str, float]`` keyed by HuggingFace
model id (or a list of synonyms).

The functions are intentionally defensive: if a source is unreachable or
returns malformed data, they log a warning and return an empty dict so the
main benchmark merge pipeline does not abort.
"""

from whichllm.models.benchmark_sources.aa_index import (
    AA_PEER_IDS,
    AA_SNAPSHOT_READ_AT,
    fetch_aa_index_scores,
    get_aa_curated_fallback,
)
from whichllm.models.benchmark_sources.chatbot_arena import fetch_arena_scores
from whichllm.models.benchmark_sources.livebench import (
    LIVEBENCH_ESTIMATED_IDS,
    LIVEBENCH_READ_AT,
    get_livebench_data,
)
from whichllm.models.benchmark_sources.open_llm_leaderboard import (
    fetch_leaderboard_with_fallback,
)
from whichllm.models.benchmark_sources.vision import (
    VISION_READ_AT,
    fetch_vision_scores,
)

# Newest curated snapshot month across the sources. Live scrapes are merged
# on top when reachable, but they frequently are not (the leaderboard
# spaces change their JSON shape), so the user-visible ranking is anchored
# to this snapshot. Surface it in the CLI so a stale recommendation is
# self-evident rather than silently trusted. It follows the read_at dates of
# the data/*.json snapshots.
BENCHMARK_SNAPSHOT = max(AA_SNAPSHOT_READ_AT, LIVEBENCH_READ_AT, VISION_READ_AT)[:7]

__all__ = [
    "AA_PEER_IDS",
    "AA_SNAPSHOT_READ_AT",
    "BENCHMARK_SNAPSHOT",
    "LIVEBENCH_ESTIMATED_IDS",
    "LIVEBENCH_READ_AT",
    "VISION_READ_AT",
    "fetch_aa_index_scores",
    "fetch_arena_scores",
    "fetch_leaderboard_with_fallback",
    "fetch_vision_scores",
    "get_aa_curated_fallback",
    "get_livebench_data",
]
