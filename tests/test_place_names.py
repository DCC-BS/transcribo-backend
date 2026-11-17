"""Tests for the Basel place name register lookup."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from transcribo_backend.agents.place_name_agent import PlaceNameAgent
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.place_name_choice import PlaceNameChoice, PlaceNameChoices
from transcribo_backend.services import place_names
from transcribo_backend.services.place_names import (
    candidates,
    canonical,
    corrections_for,
    parse_streets,
    refresh_streets,
)


def location(term: str) -> Keyword:
    return Keyword(term=term, description="Ort in Basel", type="location")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("St. Johanns-Tor", "Sankt Johannstor"),
        ("Schulhaus De Wette", "schulhaus de wette"),
        ("Bläsi-Schulhaus", "Bläsi Schulhaus"),
        ("Strassburger-Denkmal", "Straßburger Denkmal"),
    ],
)
def test_canonical_ignores_spelling_conventions(left: str, right: str) -> None:
    assert canonical(left) == canonical(right)


def stub_agent(choices: list[PlaceNameChoice] | None = None, error: Exception | None = None) -> MagicMock:
    """A place name agent that answers with fixed choices, or fails."""
    agent = MagicMock(spec=PlaceNameAgent)
    agent.run = AsyncMock(side_effect=error, return_value=PlaceNameChoices(choices=choices or []))
    return agent


async def correct(terms: list[str], agent: MagicMock, transcript: str = "") -> list[tuple[str, str]]:
    corrections = await corrections_for([location(term) for term in terms], transcript, agent)
    return [(c.original, c.corrected) for c in corrections]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Bahnhof Rien", "Bahnhof Riehen"),
        ("Sevokelschulhaus", "Sevogelschulhaus"),
        ("Affenbäumen", "Affenbrunnen"),
        ("Zeltisberger Weglein", "Seltisbergerweglein"),
    ],
)
def test_candidates_include_the_misheard_name(heard: str, expected: str) -> None:
    assert expected in candidates(heard)


def test_candidates_rank_the_name_itself_first_when_it_is_spelled_out() -> None:
    assert candidates("Sankt Johanns Tor")[0] == "St. Johanns-Tor"


@pytest.mark.parametrize("term", ["im", "Sihlcity"])
def test_candidates_skip_terms_too_short_or_unlike_any_name(term: str) -> None:
    assert candidates(term) == []


@pytest.mark.anyio
async def test_spelling_variants_are_corrected_without_asking_the_agent() -> None:
    agent = stub_agent()

    assert await correct(["Sankt Johanns Tor", "Bahnhof Riehen"], agent) == [("Sankt Johanns Tor", "St. Johanns-Tor")]
    agent.run.assert_not_called()


@pytest.mark.anyio
async def test_the_agent_chooses_by_number_and_may_choose_none() -> None:
    affen = candidates("Affenbäumen").index("Affenbrunnen") + 1
    agent = stub_agent([PlaceNameChoice(term=1, candidate=affen), PlaceNameChoice(term=2, candidate=None)])

    corrected = await correct(["Affenbäumen", "Rosengarten Bern"], agent, "Wir sitzen am Affenbäumen.")

    assert corrected == [("Affenbäumen", "Affenbrunnen")]
    assert "Sentence: Wir sitzen am Affenbäumen." in agent.run.call_args.args[0]


@pytest.mark.anyio
async def test_numbers_outside_the_given_terms_and_candidates_are_ignored() -> None:
    agent = stub_agent([PlaceNameChoice(term=1, candidate=99), PlaceNameChoice(term=7, candidate=1)])

    assert await correct(["Affenbäumen"], agent) == []


@pytest.mark.anyio
async def test_a_failing_agent_leaves_the_terms_as_heard() -> None:
    agent = stub_agent(error=RuntimeError("model down"))

    assert await correct(["Sankt Johanns Tor", "Affenbäumen"], agent) == [("Sankt Johanns Tor", "St. Johanns-Tor")]


@pytest.mark.anyio
async def test_corrections_only_consider_locations() -> None:
    keywords = [
        Keyword(term="Bahnhof Rien", description="Person", type="person"),
        Keyword(term="Bahnhof Rien", description="Firma", type="institution"),
    ]
    agent = stub_agent()

    assert await corrections_for(keywords, "", agent) == []
    agent.run.assert_not_called()


def streets_csv(*names: str) -> str:
    return "﻿strassenname\n" + "\n".join(names) + "\n"


def client_returning(status: int, text: str = "") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, text=text)))


@pytest.fixture
def restore_register():
    """refresh_streets swaps module state; put the bundled list back afterwards."""
    yield
    place_names._current = None


def test_parse_streets_drops_bom_blanks_and_duplicates() -> None:
    assert sorted(parse_streets(streets_csv("Adlerstrasse", "", "Adlerstrasse", " Zürcherstrasse "))) == [
        "Adlerstrasse",
        "Zürcherstrasse",
    ]


@pytest.mark.anyio
@pytest.mark.usefixtures("restore_register")
async def test_refresh_streets_replaces_the_list() -> None:
    streets = [f"Teststrasse {i}" for i in range(1_000)] + ["Quokkaweg"]

    await refresh_streets(client_returning(200, streets_csv(*streets)))

    assert "Quokkaweg" in candidates("Quokaweg")
    # Object names are curated separately and survive the refresh.
    assert "Sevogelschulhaus" in candidates("Sevokelschulhaus")


@pytest.mark.anyio
@pytest.mark.usefixtures("restore_register")
@pytest.mark.parametrize(
    "client",
    [
        pytest.param(client_returning(503), id="server error"),
        pytest.param(client_returning(200, streets_csv("Quokkaweg")), id="too short"),
    ],
)
async def test_refresh_streets_keeps_the_old_list_on_failure(client: httpx.AsyncClient) -> None:
    await refresh_streets(client)

    assert "Bahnhof Riehen" in candidates("Bahnhof Rien")
    assert "Quokkaweg" not in candidates("Quokaweg")


@pytest.mark.anyio
@pytest.mark.usefixtures("restore_register")
async def test_refresh_streets_keeps_the_old_list_when_unreachable() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(request.url.host, request=request)

    await refresh_streets(httpx.AsyncClient(transport=httpx.MockTransport(unreachable)))

    assert "Horburgtunnel" in candidates("Homburgtunnel")
