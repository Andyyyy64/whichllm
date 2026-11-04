"""Compare every engine card with its upstream repository.

Asks GitHub for each card's latest release and whether the repository is
archived, and writes a Markdown list of the cards that need a fresh look. It
also lists young repositories under inference topics that no card covers yet,
so a new engine shows up before someone asks for it. The weekly workflow keeps
that list in one issue. Nothing here edits a card. A person re-reads the
sources and updates it.

Usage:
    uv run --locked python scripts/check_engine_upstream.py --out report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from whichllm.engines.cards import EngineCard, load_cards

API = "https://api.github.com"
# A card with no repository to compare is re-checked by hand this often.
STALE_DAYS = 60
# Young repositories are created this recently and have this many stars.
NEW_REPO_DAYS = 14
NEW_REPO_MIN_STARS = 100
NEW_REPO_LIMIT = 15
TOPICS = ("llm-inference", "inference-engine", "llm-serving", "gguf", "llama-cpp")

_FULL_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# Repository descriptions are untrusted. Keeping words and plain punctuation
# means nothing in one can mention a user, add a link or format the issue.
_UNSAFE_RE = re.compile(r"[^\w\s.,;!?'\"/()+-]")
_MAX_DESCRIPTION = 140


@dataclass(frozen=True)
class Release:
    tag: str
    date: str


@dataclass(frozen=True)
class Repository:
    full_name: str
    stars: int
    created: str
    description: str


def _day(stamp: str) -> str:
    return stamp[:10]


def _version(tag: str) -> str:
    return tag[1:] if tag[:1] in ("v", "V") else tag


def _clean(text: str | None) -> str:
    text = " ".join(_UNSAFE_RE.sub("", text or "").split())
    if len(text) > _MAX_DESCRIPTION:
        text = text[: _MAX_DESCRIPTION - 3].rstrip() + "..."
    return text


def latest_release(client: httpx.Client, repo: str) -> Release | None:
    reply = client.get(f"/repos/{repo}/releases/latest")
    if reply.status_code == 404:
        return None
    reply.raise_for_status()
    data = reply.json()
    return Release(tag=data["tag_name"], date=_day(data["published_at"]))


def check_card(card: EngineCard, client: httpx.Client, today: dt.date) -> str | None:
    """Why the card needs a fresh look, or None when it matches upstream."""
    if card.github is None:
        age = (today - dt.date.fromisoformat(card.checked_at)).days
        if age < STALE_DAYS:
            return None
        return (
            f"Last checked {card.checked_at}, {age} days ago, with no repository "
            "to compare. Re-read its sources by hand."
        )
    reply = client.get(f"/repos/{card.github}")
    reply.raise_for_status()
    repo = reply.json()
    if repo.get("archived") and card.status != "archived":
        return f"The repository is archived, but the card says {card.status}."
    release = latest_release(client, card.github)
    if release is None:
        if card.version is not None:
            return f"The card says {card.version}, but the repository has no release."
        pushed = _day(repo["pushed_at"])
        if pushed > card.checked_at:
            return (
                f"No releases. The last push was on {pushed}, after the card "
                f"was checked on {card.checked_at}."
            )
        return None
    if card.version is None:
        return (
            f"The card has no version. The latest release is {release.tag} "
            f"from {release.date}."
        )
    if _version(release.tag) == _version(card.version):
        return None
    # GitHub's latest release skips prereleases. llama.cpp tags a prerelease
    # build (b12069) many times a day and a stable release (v0.8.0) every few
    # weeks, so a card that names a build newer than that release is current.
    if card.released_at is not None and card.released_at >= release.date:
        return None
    since = f" from {card.released_at}" if card.released_at else ""
    return (
        f"The card says {card.version}{since}. The latest release is "
        f"{release.tag} from {release.date}."
    )


def new_repositories(
    client: httpx.Client, known: set[str], today: dt.date
) -> list[Repository]:
    """Young, starred repositories under inference topics that no card covers."""
    since = (today - dt.timedelta(days=NEW_REPO_DAYS)).isoformat()
    found: dict[str, Repository] = {}
    for topic in TOPICS:
        query = f"topic:{topic} created:>={since} stars:>={NEW_REPO_MIN_STARS}"
        reply = client.get(
            "/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 30},
        )
        reply.raise_for_status()
        for item in reply.json().get("items", []):
            name = item.get("full_name") or ""
            if not _FULL_NAME_RE.fullmatch(name) or name.lower() in known:
                continue
            found[name.lower()] = Repository(
                full_name=name,
                stars=int(item["stargazers_count"]),
                created=_day(item["created_at"]),
                description=_clean(item.get("description")),
            )
    ranked = sorted(found.values(), key=lambda r: (-r.stars, r.full_name.lower()))
    return ranked[:NEW_REPO_LIMIT]


def render(
    behind: list[tuple[EngineCard, str]], young: list[Repository], today: dt.date
) -> str:
    lines = [
        f"The weekly engine check on {today.isoformat()} found these. Re-read a "
        "card's sources, update the card and its `checked_at`, and the next run "
        "drops it from this list. The issue closes when nothing is left.",
    ]
    if behind:
        lines += ["", "## Cards", ""]
        for card, reason in behind:
            where = f" (`{card.github}`)" if card.github else ""
            lines.append(f"- **{card.name}**{where} {reason}")
    if young:
        lines += [
            "",
            "## Young repositories",
            "",
            f"Created in the last {NEW_REPO_DAYS} days under "
            f"{', '.join(f'`{t}`' for t in TOPICS)}, with at least "
            f"{NEW_REPO_MIN_STARS} stars and no card yet.",
            "",
        ]
        for repo in young:
            about = f" {repo.description}" if repo.description else ""
            lines.append(
                f"- [{repo.full_name}](https://github.com/{repo.full_name}) "
                f"{repo.stars:,} stars, created {repo.created}.{about}"
            )
    return "\n".join(lines) + "\n"


def run(
    client: httpx.Client, cards: tuple[EngineCard, ...], today: dt.date
) -> tuple[int, str]:
    """How many items need a look, and the report listing them."""
    behind = []
    for card in cards:
        reason = check_card(card, client, today)
        if reason is not None:
            behind.append((card, reason))
    known = {card.github.lower() for card in cards if card.github}
    young = new_repositories(client, known, today)
    return len(behind) + len(young), render(behind, young, today)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="report file")
    args = parser.parse_args(argv)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    today = dt.datetime.now(dt.timezone.utc).date()
    with httpx.Client(base_url=API, headers=headers, timeout=30) as client:
        count, report = run(client, load_cards(), today)
    args.out.write_text(report, encoding="utf-8")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"count={count}\n")
    print(f"{count} item(s) to look at, written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
