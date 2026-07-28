"""Integration test against a live LLM for combined transcript post-processing.

Skipped unless ``LLM_URL`` is set (plus ``LLM_API_KEY``/``LLM_MODEL`` as required
by ``AppConfig.from_env``). Run with::

    LLM_URL=http://localhost:8001/v1 LLM_API_KEY=none \
    LLM_MODEL=Gemma/Gemma-4-31B \
    make test-integration

No audio is needed: post-processing operates purely on the diarized transcript
text, so synthetic fixtures are sufficient.
"""

import os

import pytest

from transcribo_backend.agents.transcript_postprocessing_agent import TranscriptPostProcessingAgent
from transcribo_backend.models.transcription_response import Segment
from transcribo_backend.services.transcript_postprocessing_service import TranscriptPostProcessingService
from transcribo_backend.utils.app_config import AppConfig

LLM_URL = os.getenv("LLM_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.skipif(not LLM_URL, reason="set LLM_URL to run LLM integration tests"),
]


def _make_service() -> TranscriptPostProcessingService:
    config = AppConfig.from_env()
    return TranscriptPostProcessingService(TranscriptPostProcessingAgent(config))


# Speaker task: self-introduction (SPEAKER_00), direct address with response
# (SPEAKER_01 = Anna), a mentioned-but-silent person (Herr Weber must not be
# assigned), and a speaker with no evidence (SPEAKER_02).
# Cleanup task: dominant-variant consistency (Cloudmigration vs misheard
# Cloudmigrazion) and a currency formatting rule — the rest must stay verbatim.
_UTTERANCES: list[tuple[str, str]] = [
    (
        "SPEAKER_00",
        "Guten Morgen zusammen, mein Name ist Lena Feldmann, ich leite heute die Sitzung.",
    ),
    ("SPEAKER_00", "Bevor wir starten: Herr Weber hat sich für heute entschuldigt."),
    (
        "SPEAKER_00",
        "Anna, kannst du kurz den Stand beim Projekt Cloudmigration-Analyse zusammenfassen?",
    ),
    (
        "SPEAKER_01",
        "Gerne. Mit Cloudmigrazion lässt sich laut unserer Umfrage kein Geld mehr verdienen.",
    ),
    (
        "SPEAKER_01",
        "Ein Betroffener hat 22000 CHF verloren, sein Büro war an der Feldbärgstrasse in Basel.",
    ),
    ("SPEAKER_02", "Dazu eine kurze Frage: Ist der Bericht zur Cloudmigration schon fertig?"),
    ("SPEAKER_01", "Ja, der ist seit letzter Woche fertig."),
    ("SPEAKER_00", "Danke Anna. Dann kommen wir zum nächsten Punkt."),
]

FIXTURE_SEGMENTS = [
    Segment(start=float(i), end=float(i + 1), text=text, speaker=speaker)
    for i, (speaker, text) in enumerate(_UTTERANCES)
]


async def test_post_processing_assigns_names_and_unifies_variants():
    service = _make_service()

    segments = [segment.model_copy(deep=True) for segment in FIXTURE_SEGMENTS]
    result_io = await service.post_process(segments)
    result = result_io.unwrap()._inner_value

    # --- speaker assignments ---
    by_label = {a.speaker: a for a in result.speaker_assignments}
    assert set(by_label) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}

    # Self-introduction: strongest evidence.
    assert by_label["SPEAKER_00"].name == "Lena Feldmann"
    assert by_label["SPEAKER_00"].confidence >= 0.8

    # Direct address followed by response.
    assert by_label["SPEAKER_01"].name == "Anna"

    # No evidence: must stay unassigned; in particular the mentioned-but-silent
    # "Herr Weber" must not be hallucinated onto this label.
    assert by_label["SPEAKER_02"].name is None

    # --- corrections ---
    full_text = " ".join(segment.text for segment in segments)

    # Dominant variant wins everywhere.
    assert "Cloudmigrazion" not in full_text
    assert "Cloudmigration" in segments[3].text

    # Untouched content stays verbatim.
    assert segments[0].text == FIXTURE_SEGMENTS[0].text
    assert segments[-1].text == FIXTURE_SEGMENTS[-1].text

    assert all(c.confidence >= 0.7 for c in result.corrections)
