"""Snap misheard Basel street and place names onto their official spellings."""

import asyncio
import csv
import io
import unicodedata
from functools import lru_cache
from pathlib import Path

import httpx
from dcc_backend_common.logger import get_logger
from rapidfuzz import fuzz, process

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection

logger = get_logger(__name__)

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Street names come live from Basel open data; the bundled copy only bridges the
# time until the first download succeeds. Object names are curated by hand.
_STREETS_URL = (
    "https://data.bs.ch/api/explore/v2.1/catalog/datasets/100189/exports/csv?select=strassenname&delimiter=%3B"
)
_STREETS_REFRESH_SECONDS = 24 * 60 * 60
# A partial response would silently shrink the register, so it is discarded.
_MIN_STREETS = 1_000

# Lower values start rewriting places outside the register ("Paradeplatz" -> "Claraplatz").
_MIN_SCORE = 0.80
_REASON = "Amtliche Schreibweise aus dem Basler Strassen- und Objektnamenverzeichnis"

# Single letters and articles match everything and nothing; ignore them.
_MIN_TERM_LENGTH = 4


def canonical(text: str) -> str:
    """Drop spelling conventions so "St. Johanns-Tor" and "Sankt Johannstor" compare equal."""
    lowered = text.lower().strip().replace("ß", "ss").replace("sankt", "st")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if c.isalnum() and unicodedata.category(c) != "Mn")


def _read_asset(name: str) -> tuple[str, ...]:
    lines = (line.strip() for line in (_ASSETS_DIR / name).read_text("utf-8").splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


@lru_cache(maxsize=1)
def _objects() -> tuple[str, ...]:
    return _read_asset("basel_objects.txt")


type _Register = tuple[tuple[str, ...], tuple[str, ...]]
_current: _Register | None = None


def _build_register(streets: tuple[str, ...]) -> _Register:
    names = tuple(sorted({*streets, *_objects()}))
    return names, tuple(canonical(name) for name in names)


def _register() -> _Register:
    """The official names and their canonical forms."""
    global _current
    if _current is None:
        _current = _build_register(_read_asset("basel_streets.txt"))
    return _current


def parse_streets(csv_text: str) -> tuple[str, ...]:
    """Street names from the open data CSV export, deduplicated."""
    rows = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")), delimiter=";")
    return tuple({name for row in rows if (name := (row.get("strassenname") or "").strip())})


async def refresh_streets(client: httpx.AsyncClient) -> None:
    """Replace the street names with the current open data; on any failure keep the old ones."""
    global _current
    try:
        response = await client.get(_STREETS_URL)
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("Street names not refreshed, keeping the current list", error=str(error))
        return

    streets = parse_streets(response.text)
    if len(streets) < _MIN_STREETS:
        logger.warning("Street names not refreshed, response too short", received=len(streets))
        return

    _current = _build_register(streets)
    logger.info("Street names refreshed", count=len(streets))


async def keep_streets_fresh() -> None:
    """Refresh the street names now and then once a day, until cancelled."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        while True:
            await refresh_streets(client)
            await asyncio.sleep(_STREETS_REFRESH_SECONDS)


def match(term: str) -> tuple[str, float] | None:
    """Closest official name for ``term`` and its score, or None below the threshold."""
    query = canonical(term)
    if len(query) < _MIN_TERM_LENGTH:
        return None

    names, canonical_forms = _register()
    hit = process.extractOne(query, canonical_forms, scorer=fuzz.ratio, score_cutoff=_MIN_SCORE * 100)
    if hit is None:
        return None

    candidate, score, index = hit
    # Appending to what was heard makes a different name ("Lehenmatt" -> "Lehenmattweg").
    if query in candidate and query != candidate:
        return None

    return names[index], score / 100


def corrections_for(keywords: list[Keyword]) -> list[TranscriptCorrection]:
    """Corrections from location keywords to their official spelling; the match score is the confidence."""
    corrections: list[TranscriptCorrection] = []
    for keyword in keywords:
        term = keyword.term.strip()
        hit = match(term) if keyword.type == "location" else None
        if hit is not None and hit[0] != term:
            corrections.append(TranscriptCorrection(original=term, corrected=hit[0], reason=_REASON, confidence=hit[1]))
    return corrections
