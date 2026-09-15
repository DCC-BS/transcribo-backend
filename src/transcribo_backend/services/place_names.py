"""Snap misheard Basel street and place names onto their official spellings."""

import unicodedata
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_ASSET_FILES = ("basel_streets.txt", "basel_objects.txt")

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


@lru_cache(maxsize=1)
def _register() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The official names and their canonical forms, read once per process."""
    lines = (line.strip() for name in _ASSET_FILES for line in (_ASSETS_DIR / name).read_text("utf-8").splitlines())
    names = tuple(sorted({line for line in lines if line and not line.startswith("#")}))
    return names, tuple(canonical(name) for name in names)


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
