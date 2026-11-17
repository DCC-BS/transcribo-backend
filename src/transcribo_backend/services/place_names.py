"""Snap misheard Basel street and place names onto their official spellings."""

import asyncio
import csv
import io
import unicodedata
from functools import lru_cache
from itertools import batched
from pathlib import Path

import httpx
from dcc_backend_common.logger import get_logger
from rapidfuzz import fuzz, process
from returns.future import future_safe
from returns.io import IOSuccess

from transcribo_backend.agents.place_name_agent import PlaceNameAgent
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.place_name_choice import PlaceNameChoice
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

# The agent chooses among this many closest official names. The cutoff only drops terms that resemble
# no official name at all: every misheard name in the transcribo-test sets scores 58 or more.
_CANDIDATES = 5
_CANDIDATE_CUTOFF = 55
# Context around a term, enough to tell "Rosengarten" in Basel from one in Bern.
_CONTEXT_CHARS = 80
# Terms per agent call: keeps every prompt small, and the batches run concurrently.
_TERMS_PER_CALL = 40
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


def candidates(term: str) -> list[str]:
    """The official names closest to ``term``, best first."""
    query = canonical(term)
    if len(query) < _MIN_TERM_LENGTH:
        return []

    names, canonical_forms = _register()
    hits = process.extract(query, canonical_forms, scorer=fuzz.ratio, score_cutoff=_CANDIDATE_CUTOFF, limit=_CANDIDATES)
    return [names[index] for _, _, index in hits]


def _official(term: str, name: str) -> TranscriptCorrection:
    return TranscriptCorrection(original=term, corrected=name, reason=_REASON, confidence=1.0)


def _context(transcript: str, term: str) -> str:
    position = transcript.find(term)
    if position < 0:
        return ""
    return transcript[max(position - _CONTEXT_CHARS, 0) : position + len(term) + _CONTEXT_CHARS]


def _choice_prompt(open_terms: list[tuple[str, list[str]]], transcript: str) -> str:
    """Numbered terms with their context and numbered candidates; the agent answers in numbers only."""
    blocks = []
    for number, (term, found) in enumerate(open_terms, start=1):
        options = " ".join(f"({index}) {name}" for index, name in enumerate(found, start=1))
        blocks.append(f"Term {number}: {term}\nSentence: {_context(transcript, term)}\nCandidates: {options}")
    return "\n\n".join(blocks)


def _chosen(open_terms: list[tuple[str, list[str]]], choices: list[PlaceNameChoice]) -> list[TranscriptCorrection]:
    """Corrections for the valid choices; numbers outside the given terms and candidates are ignored."""
    corrections = []
    for choice in choices:
        if choice.candidate is None or not 1 <= choice.term <= len(open_terms):
            continue
        term, found = open_terms[choice.term - 1]
        if 1 <= choice.candidate <= len(found):
            corrections.append(_official(term, found[choice.candidate - 1]))
    return corrections


@future_safe
async def _choose(
    agent: PlaceNameAgent, open_terms: list[tuple[str, list[str]]], transcript: str
) -> list[TranscriptCorrection]:
    result = await agent.run(_choice_prompt(open_terms, transcript))
    return _chosen(open_terms, result.choices)


async def corrections_for(
    keywords: list[Keyword], transcript: str, agent: PlaceNameAgent
) -> list[TranscriptCorrection]:
    """Corrections from location keywords to their official spelling.

    A term that is an official name up to spelling conventions is corrected directly. For every other
    term the agent chooses among the closest official names in the term's context, or none. A failed
    agent call only costs its terms: they stay as heard.

    Args:
        keywords: Keywords proposed for the transcript; only locations are considered.
        transcript: The transcript text the terms occur in.
        agent: Chooses among the candidates.

    Returns:
        Corrections onto official names, each one at full confidence.
    """
    corrections: list[TranscriptCorrection] = []
    open_terms: list[tuple[str, list[str]]] = []
    for term in dict.fromkeys(k.term.strip() for k in keywords if k.type == "location"):
        found = candidates(term)
        if found and canonical(found[0]) == canonical(term):
            if found[0] != term:
                corrections.append(_official(term, found[0]))
        elif found:
            open_terms.append((term, found))

    batches = [list(batch) for batch in batched(open_terms, _TERMS_PER_CALL, strict=False)]
    for batch, result in zip(
        batches, await asyncio.gather(*(_choose(agent, b, transcript) for b in batches)), strict=True
    ):
        if isinstance(result, IOSuccess):
            corrections += result.unwrap()._inner_value
        else:
            logger.warning(
                "Place names not checked, terms stay as heard", terms=len(batch), exc_info=result.failure()._inner_value
            )
    return corrections
