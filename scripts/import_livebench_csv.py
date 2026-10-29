"""Refresh ``src/whichllm/data/livebench.json`` from a LiveBench leaderboard CSV.

LiveBench publishes their leaderboard as a dated CSV (e.g.
``https://livebench.ai/table_2026_01_08.csv``). The CSV's rows replace the
snapshot's models; the hand-estimated carryover is kept.

Usage:
    curl https://livebench.ai/table_2026_01_08.csv | python scripts/import_livebench_csv.py 2026-01-08

Then refit ``_LB_MIN`` / ``_LB_MAX`` in ``livebench.py``; a test checks the fit.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "whichllm" / "data" / "livebench.json"
)

# LiveBench CSV model name -> list of HuggingFace ids that share the score.
# When several CSV rows map onto the same HF id (e.g. thinking vs. base),
# the highest average wins.
CSV_NAME_TO_HF_IDS: dict[str, list[str]] = {
    "deepseek-v3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "deepseek-v3.2-exp": ["deepseek-ai/DeepSeek-V3.2-Exp"],
    "deepseek-v3.2-exp-thinking": ["deepseek-ai/DeepSeek-V3.2-Exp"],
    "deepseek-v3.2-thinking": ["deepseek-ai/DeepSeek-V3.2"],
    "deepseek-v4-flash": ["deepseek-ai/DeepSeek-V4-Flash"],
    "deepseek-v4-pro": ["deepseek-ai/DeepSeek-V4-Pro"],
    "devstral-2512": ["mistralai/Devstral-2512"],
    "gemma-4-31b-it": ["google/gemma-4-31b-it"],
    "glm-4.6": ["zai-org/GLM-4.6"],
    "glm-4.6v": ["zai-org/GLM-4.6V"],
    "glm-4.7": ["zai-org/GLM-4.7"],
    "glm-5": ["zai-org/GLM-5"],
    "glm-5.1": ["zai-org/GLM-5.1"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "kimi-k2-instruct": ["moonshotai/Kimi-K2-Instruct"],
    "kimi-k2-thinking": ["moonshotai/Kimi-K2-Thinking"],
    "kimi-k2.5-thinking": ["moonshotai/Kimi-K2.5"],
    "kimi-k2.6-thinking": ["moonshotai/Kimi-K2.6-Thinking"],
    "mimo-v2-pro": ["XiaomiMiMo/MiMo-V2-Pro"],
    "minimax-m2.5": ["MiniMaxAI/MiniMax-M2.5"],
    "minimax-m2.7": ["MiniMaxAI/MiniMax-M2.7"],
    "nemotron-3-super-120b-a12b": ["nvidia/Nemotron-3-Super-120B-A12B"],
    "qwen3-235b-a22b-instruct-2507": ["Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "qwen3-235b-a22b-thinking-2507": ["Qwen/Qwen3-235B-A22B-Thinking-2507"],
    "qwen3-30b-a3b-thinking": ["Qwen/Qwen3-30B-A3B-Thinking-2507"],
    "qwen3-32b-thinking": ["Qwen/Qwen3-32B"],
    "qwen3-next-80b-a3b-instruct": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "qwen3-next-80b-a3b-thinking": ["Qwen/Qwen3-Next-80B-A3B-Thinking"],
    "qwen3.6-27b": ["Qwen/Qwen3.6-27B"],
}


def row_average(row: dict[str, str]) -> float | None:
    nums: list[float] = []
    for key, value in row.items():
        if key == "model" or not value:
            continue
        try:
            nums.append(float(value))
        except ValueError:
            continue
    if not nums:
        return None
    return sum(nums) / len(nums)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", argv[1]):
        print(
            "usage: ... | python scripts/import_livebench_csv.py YYYY-MM-DD "
            "(the date in the CSV's name)",
            file=sys.stderr,
        )
        return 2
    read_at = argv[1]
    rows = list(csv.DictReader(sys.stdin))

    best: dict[str, float] = {}
    matched: set[str] = set()
    for row in rows:
        name = row["model"]
        hf_ids = CSV_NAME_TO_HF_IDS.get(name)
        if not hf_ids:
            continue
        avg = row_average(row)
        if avg is None:
            continue
        matched.add(name)
        for hf_id in hf_ids:
            if avg > best.get(hf_id, 0.0):
                best[hf_id] = avg

    unmapped_open = [
        row["model"]
        for row in rows
        if row["model"] not in matched
        and not any(
            tok in row["model"]
            for tok in (
                "claude",
                "gpt-",
                "gemini",
                "grok",
                "arcee",
                "elephant",
            )
        )
    ]
    if unmapped_open:
        print(
            f"# note: {len(unmapped_open)} unmapped row(s) in CSV — extend "
            "CSV_NAME_TO_HF_IDS if any are open-weight:",
            " ".join(sorted(unmapped_open)),
            file=sys.stderr,
        )

    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot["source"] = f"https://livebench.ai/table_{read_at.replace('-', '_')}.csv"
    snapshot["read_at"] = read_at
    snapshot["models"] = {hf_id: round(best[hf_id], 1) for hf_id in sorted(best)}
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(best)} models to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
