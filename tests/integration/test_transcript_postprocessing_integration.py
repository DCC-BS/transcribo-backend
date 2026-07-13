"""Integration test against a live LLM for combined transcript post-processing.

Skipped unless ``LLM_URL`` is set (plus ``LLM_API_KEY``/``LLM_MODEL`` as required
by ``AppConfig.from_env``). Run with::

    LLM_URL=http://localhost:8001/v1 LLM_API_KEY=none \
    LLM_MODEL=Gemma/Gemma-4-31B \
    make test-integration

No audio is needed: post-processing operates purely on the diarized transcript
text (with Whisper word probabilities), so synthetic fixtures are sufficient.
"""

import os

import pytest

from transcribo_backend.agents.speaker_inference_agent import SpeakerInferenceAgent
from transcribo_backend.agents.transcript_cleanup_agent import TranscriptCleanupAgent
from transcribo_backend.models.transcription_response import Segment, Word
from transcribo_backend.services.transcript_postprocessing_service import TranscriptPostProcessingService
from transcribo_backend.utils.app_config import AppConfig

LLM_URL = os.getenv("LLM_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.skipif(not LLM_URL, reason="set LLM_URL to run LLM integration tests"),
]


def _word(token: str, probability: float) -> Word:
    return Word(start=0.0, end=1.0, word=token, probability=probability)


def _make_service() -> TranscriptPostProcessingService:
    config = AppConfig.from_env()
    return TranscriptPostProcessingService(config, SpeakerInferenceAgent(config), TranscriptCleanupAgent(config))


# Speaker task: self-introduction (SPEAKER_00), direct address with response
# (SPEAKER_01 = Anna), a mentioned-but-silent person (Herr Weber must not be
# assigned), and a speaker with no evidence (SPEAKER_02).
# Cleanup task: dominant-variant consistency (Dropshipping vs uncertain
# Jobshipping), a misheard Basel street (Feldbergstrasse is in the hotwords
# asset), and a currency formatting rule — the rest must stay verbatim.
_UTTERANCES: list[tuple[str, str, list[Word] | None]] = [
    (
        "SPEAKER_00",
        "Guten Morgen zusammen, mein Name ist Yanick Schraner, ich leite heute die Sitzung.",
        None,
    ),
    ("SPEAKER_00", "Bevor wir starten: Herr Weber hat sich für heute entschuldigt.", None),
    (
        "SPEAKER_00",
        "Anna, kannst du kurz den Stand beim Projekt Dropshipping-Analyse zusammenfassen?",
        None,
    ),
    (
        "SPEAKER_01",
        "Gerne. Mit Jobshipping lässt sich laut unserer Umfrage kein Geld mehr verdienen.",
        [_word(" Jobshipping", 0.3)],
    ),
    (
        "SPEAKER_01",
        "Ein Betroffener hat 22000 CHF verloren, sein Büro war an der Feldbärgstrasse in Basel.",
        [_word(" Feldbärgstrasse", 0.35)],
    ),
    ("SPEAKER_02", "Dazu eine kurze Frage: Ist der Bericht zum Dropshipping schon fertig?", None),
    ("SPEAKER_01", "Ja, der ist seit letzter Woche fertig.", None),
    ("SPEAKER_00", "Danke Anna. Dann kommen wir zum nächsten Punkt.", None),
]

FIXTURE_SEGMENTS = [
    Segment(start=float(i), end=float(i + 1), text=text, speaker=speaker, words=words)
    for i, (speaker, text, words) in enumerate(_UTTERANCES)
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
    assert by_label["SPEAKER_00"].name == "Yanick Schraner"
    assert by_label["SPEAKER_00"].confidence >= 0.8

    # Direct address followed by response.
    assert by_label["SPEAKER_01"].name == "Anna"

    # No evidence: must stay unassigned; in particular the mentioned-but-silent
    # "Herr Weber" must not be hallucinated onto this label.
    assert by_label["SPEAKER_02"].name is None

    # --- corrections ---
    full_text = " ".join(segment.text for segment in segments)

    # Dominant variant wins everywhere.
    assert "Jobshipping" not in full_text
    assert "Dropshipping" in segments[3].text

    # Misheard street corrected to the hotword spelling.
    assert "Feldbergstrasse" in segments[4].text

    # Untouched content stays verbatim.
    assert segments[0].text == FIXTURE_SEGMENTS[0].text
    assert segments[-1].text == FIXTURE_SEGMENTS[-1].text

    assert all(c.confidence >= 0.7 for c in result.corrections)
