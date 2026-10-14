"""Tests for the Basel place name register lookup."""

import pytest

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.services.place_names import canonical, corrections_for, match


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


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Bahnhof Rien", "Bahnhof Riehen"),
        ("Sevokelschulhaus", "Sevogelschulhaus"),
        ("Homburgtunnel", "Horburgtunnel"),
    ],
)
def test_match_repairs_misheard_names(heard: str, expected: str) -> None:
    hit = match(heard)
    assert hit is not None
    assert hit[0] == expected


@pytest.mark.parametrize("term", ["Limmatquai", "Sihlcity", "Kapellbrücke"])
def test_match_leaves_places_outside_basel_alone(term: str) -> None:
    assert match(term) is None


def test_match_ignores_terms_too_short_to_discriminate() -> None:
    assert match("im") is None


def test_corrections_for_only_considers_locations() -> None:
    keywords = [
        Keyword(term="Bahnhof Rien", description="Person", type="person"),
        Keyword(term="Bahnhof Rien", description="Firma", type="institution"),
    ]
    assert corrections_for(keywords) == []


def test_corrections_for_skips_names_already_correct() -> None:
    assert corrections_for([location("Bahnhof Riehen")]) == []


def test_corrections_for_carries_the_score_as_confidence() -> None:
    corrections = corrections_for([location("Sevokelschulhaus")])

    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.original == "Sevokelschulhaus"
    assert correction.corrected == "Sevogelschulhaus"
    assert 0.8 <= correction.confidence <= 1.0


@pytest.mark.parametrize("term", ["Lehenmatt", "Neubad", "Gundeldinger"])
def test_match_does_not_extend_a_name_into_a_longer_one(term: str) -> None:
    assert match(term) is None


def test_match_still_repairs_a_name_that_shares_a_prefix() -> None:
    assert match("Schulhaus Lehematt") == ("Schulhaus Lehenmatt", pytest.approx(0.971, abs=0.01))
