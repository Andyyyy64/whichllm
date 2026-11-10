"""The weekly check of engine cards against their upstream repositories."""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import httpx

from whichllm.engines import load_cards


def _load_script():
    path = Path(__file__).parent.parent / "scripts" / "check_engine_upstream.py"
    spec = importlib.util.spec_from_file_location("check_engine_upstream", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the module up while building the script's classes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upstream = _load_script()
BY_ID = {card.id: card for card in load_cards()}
TODAY = dt.date(2026, 11, 9)
# A fixed version, so these tests do not move when the card is refreshed.
OLLAMA = dataclasses.replace(
    BY_ID["ollama"], version="0.40.14", released_at="2026-11-03"
)


def _client(routes: dict[str, object], searches: list[dict] | None = None):
    """A GitHub API stand-in. ``routes`` maps a path to its JSON, or to None for
    a 404; every search returns ``searches``."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/repositories":
            return httpx.Response(200, json={"items": searches or []})
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=body)

    return httpx.Client(base_url=upstream.API, transport=httpx.MockTransport(handle))


def _repo(archived=False, pushed="2026-11-01T00:00:00Z"):
    return {"archived": archived, "pushed_at": pushed}


def _release(tag, date="2026-11-08T12:00:00Z"):
    return {"tag_name": tag, "published_at": date}


def test_every_github_card_names_owner_and_repo():
    for card in load_cards():
        if card.github:
            owner, _, repo = card.github.partition("/")
            assert owner and repo, card.id


def test_a_newer_release_puts_the_card_on_the_list():
    card = OLLAMA
    client = _client(
        {
            "/repos/ollama/ollama": _repo(),
            "/repos/ollama/ollama/releases/latest": _release("v0.41.0"),
        }
    )

    reason = upstream.check_card(card, client, TODAY)

    assert reason == (
        f"The card says {card.version} from {card.released_at}. "
        "The latest release is v0.41.0 from 2026-11-08."
    )


def test_the_same_release_with_a_v_prefix_matches():
    card = OLLAMA
    client = _client(
        {
            "/repos/ollama/ollama": _repo(),
            "/repos/ollama/ollama/releases/latest": _release(f"v{card.version}"),
        }
    )

    assert upstream.check_card(card, client, TODAY) is None


def test_a_build_newer_than_the_stable_release_is_current():
    card = dataclasses.replace(
        BY_ID["llama.cpp"], version="b12069", released_at="2026-11-03"
    )
    path = "/repos/ggml-org/llama.cpp"
    older = _client(
        {path: _repo(), f"{path}/releases/latest": _release("v0.8.0", "2026-10-28")}
    )
    newer = _client(
        {path: _repo(), f"{path}/releases/latest": _release("v0.9.0", "2026-11-08")}
    )

    assert upstream.check_card(card, older, TODAY) is None
    assert "latest release is v0.9.0" in upstream.check_card(card, newer, TODAY)


def test_a_card_that_tracks_minor_versions_ignores_patch_releases():
    card = dataclasses.replace(
        BY_ID["abyss"], version="1.3.129", released_at="2026-11-01"
    )
    path = "/repos/zephel01/Abyss"
    patch = _client({path: _repo(), f"{path}/releases/latest": _release("v1.3.140")})
    minor = _client({path: _repo(), f"{path}/releases/latest": _release("v1.4.0")})

    fresh = dataclasses.replace(card, checked_at="2026-11-01")
    old = dataclasses.replace(card, checked_at="2026-10-01")

    assert card.track == "minor"
    assert upstream.check_card(fresh, patch, TODAY) is None
    assert upstream.check_card(old, patch, TODAY) == (
        "Last checked 2026-10-01, 39 days ago. It follows minor versions, and "
        "the latest release is v1.3.140 from 2026-11-08."
    )
    assert "latest release is v1.4.0" in upstream.check_card(fresh, minor, TODAY)


def test_only_the_cards_with_daily_patch_releases_track_minor_versions():
    minor = {c.id for c in load_cards() if c.track == "minor"}

    assert minor == {"abyss", "project-maya"}


def test_a_repository_without_releases_counts_its_last_push():
    card = dataclasses.replace(BY_ID["ik_llama.cpp"], checked_at="2026-11-04")
    path = "/repos/ikawrakow/ik_llama.cpp"

    quiet = _client({path: _repo(pushed="2026-11-03T08:00:00Z")})
    moved = _client({path: _repo(pushed="2026-11-07T08:00:00Z")})

    assert upstream.check_card(card, quiet, TODAY) is None
    assert "last push was on 2026-11-07" in upstream.check_card(card, moved, TODAY)


def test_an_archived_repository_is_reported():
    card = BY_ID["koboldcpp"]
    client = _client({"/repos/LostRuins/koboldcpp": _repo(archived=True)})

    reason = upstream.check_card(card, client, TODAY)

    assert reason == "The repository is archived, but the card says active."


def test_a_card_without_a_repository_waits_until_it_is_stale():
    card = BY_ID["lm-studio"]
    fresh = dataclasses.replace(card, checked_at="2026-10-01")
    stale = dataclasses.replace(card, checked_at="2026-08-01")
    client = _client({})

    assert upstream.check_card(fresh, client, TODAY) is None
    assert "100 days ago" in upstream.check_card(stale, client, TODAY)


def test_young_repositories_leave_out_carded_ones_and_untrusted_text():
    searches = [
        {
            "full_name": "someone/new-engine",
            "stargazers_count": 640,
            "created_at": "2026-11-02T00:00:00Z",
            "description": "Fast MoE runner @maintainer [click](https://x.test) <b>",
        },
        {
            "full_name": "ggml-org/llama.cpp",
            "stargazers_count": 99999,
            "created_at": "2026-11-02T00:00:00Z",
            "description": "already carded",
        },
        {
            "full_name": "bad name/with space",
            "stargazers_count": 500,
            "created_at": "2026-11-02T00:00:00Z",
            "description": None,
        },
    ]
    client = _client({}, searches)

    young = upstream.new_repositories(client, {"ggml-org/llama.cpp"}, TODAY)

    assert [repo.full_name for repo in young] == ["someone/new-engine"]
    assert young[0].description == "Fast MoE runner maintainer click(https//x.test) b"


def test_the_report_lists_cards_and_repositories():
    cards = (OLLAMA, BY_ID["llama.cpp"])
    client = _client(
        {
            "/repos/ollama/ollama": _repo(),
            "/repos/ollama/ollama/releases/latest": _release("v0.41.0"),
            "/repos/ggml-org/llama.cpp": _repo(),
            "/repos/ggml-org/llama.cpp/releases/latest": _release(
                BY_ID["llama.cpp"].version
            ),
        },
        [
            {
                "full_name": "someone/new-engine",
                "stargazers_count": 1234,
                "created_at": "2026-11-02T00:00:00Z",
                "description": "A runner",
            }
        ],
    )

    count, report = upstream.run(client, cards, TODAY)

    assert count == 2
    assert "- **Ollama** (`ollama/ollama`) The card says" in report
    assert "llama.cpp**" not in report
    assert (
        "- [someone/new-engine](https://github.com/someone/new-engine) "
        "1,234 stars, created 2026-11-02. A runner"
    ) in report


def test_nothing_to_report_counts_zero():
    card = OLLAMA
    client = _client(
        {
            "/repos/ollama/ollama": _repo(),
            "/repos/ollama/ollama/releases/latest": _release(card.version),
        }
    )

    count, _ = upstream.run(client, (card,), TODAY)

    assert count == 0
