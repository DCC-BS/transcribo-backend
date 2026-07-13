from unittest.mock import AsyncMock, MagicMock

import pytest

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerAssignmentResult, SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCleanupResult, TranscriptCorrection
from transcribo_backend.models.transcription_response import Segment, Word
from transcribo_backend.services.transcript_postprocessing_service import (
    TranscriptPostProcessingService,
    apply_corrections,
    build_postprocessing_transcript,
    enumerate_roles,
)
from transcribo_backend.utils.app_config import AppConfig


def _word(token: str, probability: float) -> Word:
    return Word(start=0.0, end=1.0, word=token, probability=probability)


def _make_service(
    cleanup_result: TranscriptCleanupResult, speaker_result: SpeakerAssignmentResult
) -> tuple[TranscriptPostProcessingService, MagicMock, MagicMock]:
    """Service wired with two mocked agents (cleanup runs first, then speaker)."""
    app_config = MagicMock(spec=AppConfig)
    cleanup_agent = MagicMock()
    cleanup_agent.run = AsyncMock(return_value=cleanup_result)
    speaker_agent = MagicMock()
    speaker_agent.run = AsyncMock(return_value=speaker_result)
    service = TranscriptPostProcessingService(app_config, speaker_agent, cleanup_agent)
    return service, cleanup_agent, speaker_agent


def test_build_transcript_merges_speakers_and_marks_uncertain_words():
    segments = [
        Segment(start=0.0, end=1.0, text=" Hallo zusammen. ", speaker="Speaker_00"),
        Segment(
            start=1.0,
            end=2.0,
            text="Wir reden über Jobshipping.",
            speaker="Speaker_00",
            words=[_word(" Wir", 0.99), _word(" Jobshipping", 0.31)],
        ),
        Segment(start=2.0, end=3.0, text="Genau.", speaker="Speaker_01", words=None),
        Segment(start=3.0, end=4.0, text="Ohne Sprecher.", speaker=None),
    ]

    transcript = build_postprocessing_transcript(segments)
    assert transcript == (
        "Speaker_00: Hallo zusammen. Wir reden über ⟨Jobshipping⟩.\nSpeaker_01: Genau.\nUnknown: Ohne Sprecher."
    )

    clamped = build_postprocessing_transcript(segments, max_chars=60)
    assert clamped == "Speaker_00: Hallo zusammen. Wir reden über ⟨Jobshipping⟩."


def test_apply_corrections_word_boundary_and_threshold():
    segments = [
        Segment(start=0.0, end=1.0, text="Jobshipping ist toll.", speaker="A"),
        Segment(start=1.0, end=2.0, text="Alles über Jobshipping, auch Jobshippingkurse.", speaker="B"),
        Segment(start=2.0, end=3.0, text="Um 14.30 geht es weiter.", speaker="A"),
    ]

    corrections = [
        TranscriptCorrection(original="Jobshipping", corrected="Dropshipping", reason="dominant", confidence=0.95),
        TranscriptCorrection(original="14.30", corrected="14:30 Uhr", reason="time rule", confidence=0.9),
        TranscriptCorrection(original="toll", corrected="super", reason="style", confidence=0.5),
        TranscriptCorrection(original="ist", corrected="ist", reason="no-op", confidence=1.0),
        TranscriptCorrection(original="nicht da", corrected="x", reason="no match", confidence=1.0),
    ]

    applied = apply_corrections(segments, corrections)

    assert segments[0].text == "Dropshipping ist toll."
    # Word boundary: the compound "Jobshippingkurse" stays untouched.
    assert segments[1].text == "Alles über Dropshipping, auch Jobshippingkurse."
    assert segments[2].text == "Um 14:30 Uhr geht es weiter."
    assert [c.original for c in applied] == ["Jobshipping", "14.30"]


def test_apply_corrections_skips_content_deleting_pairs():
    """A correction whose replacement drops letter-words must never be applied."""
    segments = [
        Segment(start=0.0, end=1.0, text="En qué país o en qué países.", speaker="A"),
        Segment(start=1.0, end=2.0, text="Italien, Frankreich und Spanien.", speaker="B"),
        Segment(start=2.0, end=3.0, text="Das kostet 22000 CHF.", speaker="A"),
    ]

    corrections = [
        # Shortens a clause -> would silently delete "o en qué países": rejected.
        TranscriptCorrection(
            original="En qué país o en qué países",
            corrected="En qué país",
            reason="rewrite",
            confidence=0.99,
        ),
        # Drops a trailing word -> rejected.
        TranscriptCorrection(
            original="Frankreich und Spanien", corrected="Frankreich", reason="rewrite", confidence=0.95
        ),
        # Legitimate reformatting only changes digit/word tokens (no letter-word
        # loss: "CHF" -> "Franken") -> still applied.
        TranscriptCorrection(original="22000 CHF", corrected="22'000 Franken", reason="currency rule", confidence=0.9),
    ]

    applied = apply_corrections(segments, corrections)

    assert segments[0].text == "En qué país o en qué países."
    assert segments[1].text == "Italien, Frankreich und Spanien."
    assert segments[2].text == "Das kostet 22'000 Franken."
    assert [c.original for c in applied] == ["22000 CHF"]


def test_enumerate_roles_numbers_only_nameless_duplicates():
    assignments = [
        SpeakerNameAssignment(speaker="A", name="Anna", role="Dolmetscher", confidence=0.9),
        SpeakerNameAssignment(speaker="B", name=None, role="Dolmetscher", confidence=0.6),
        SpeakerNameAssignment(speaker="C", name=None, role="Dolmetscher", confidence=0.6),
        SpeakerNameAssignment(speaker="D", name=None, role="Moderatorin", confidence=0.6),
    ]

    enumerate_roles(assignments)

    by_label = {a.speaker: a for a in assignments}
    # Named speaker keeps its role untouched.
    assert by_label["A"].role == "Dolmetscher"
    # Two nameless speakers sharing a role are numbered.
    assert {by_label["B"].role, by_label["C"].role} == {"Dolmetscher 1", "Dolmetscher 2"}
    # A unique nameless role is left as-is.
    assert by_label["D"].role == "Moderatorin"


@pytest.mark.anyio
async def test_post_process_runs_cleanup_then_speaker_and_returns_all():
    assignments = [
        SpeakerNameAssignment(speaker="A", name="Anna", confidence=0.9, evidence="Ich bin Anna."),
    ]
    cleanup_result = TranscriptCleanupResult(
        corrections=[
            TranscriptCorrection(original="Jobshipping", corrected="Dropshipping", reason="dominant", confidence=0.9),
            TranscriptCorrection(original="toll", corrected="super", reason="style", confidence=0.4),
        ],
        keywords=[Keyword(term="Dropshipping", description="Online-Handelsmodell", type="object")],
    )
    speaker_result = SpeakerAssignmentResult(assignments=assignments)

    service, cleanup_agent, speaker_agent = _make_service(cleanup_result, speaker_result)
    segments = [Segment(start=0.0, end=1.0, text="Jobshipping ist toll.", speaker="A")]

    result_io = await service.post_process(segments)
    result = result_io.unwrap()._inner_value

    # Cleanup sees the raw transcript.
    cleanup_agent.run.assert_called_once_with("A: Jobshipping ist toll.")
    # Corrections are applied in place, so the speaker call sees the fixed text.
    assert segments[0].text == "Dropshipping ist toll."
    speaker_agent.run.assert_called_once_with("A: Dropshipping ist toll.")

    # Only the applied correction is reported; the low-confidence one is dropped.
    assert [c.original for c in result.corrections] == ["Jobshipping"]
    assert result.speaker_assignments == assignments
    assert result.keywords[0].term == "Dropshipping"


@pytest.mark.anyio
async def test_post_process_propagates_agent_failure():
    app_config = MagicMock(spec=AppConfig)
    cleanup_agent = MagicMock()
    cleanup_agent.run = AsyncMock(side_effect=RuntimeError("llm down"))
    speaker_agent = MagicMock()
    speaker_agent.run = AsyncMock(return_value=SpeakerAssignmentResult(assignments=[]))
    service = TranscriptPostProcessingService(app_config, speaker_agent, cleanup_agent)

    result_io = await service.post_process([Segment(start=0.0, end=1.0, text="Hallo.", speaker="A")])

    error = result_io.failure()._inner_value
    assert isinstance(error, RuntimeError)
    assert str(error) == "llm down"


@pytest.mark.anyio
async def test_post_process_appends_user_keywords_to_both_prompts():
    service, cleanup_agent, speaker_agent = _make_service(
        TranscriptCleanupResult(corrections=[], keywords=[]),
        SpeakerAssignmentResult(assignments=[]),
    )
    segments = [Segment(start=0.0, end=1.0, text="Die Bibos Academy.", speaker="A")]
    keywords = [Keyword(term="BeeBoss", description="Name der Dropshipping-Academy", type="institution")]

    await service.post_process(segments, keywords=keywords)

    expected = "A: Die Bibos Academy.\n\nUser keywords:\nBeeBoss: Name der Dropshipping-Academy"
    cleanup_agent.run.assert_called_once_with(expected)
    # Speaker inference gets the keywords too, so assigned names use the
    # user-confirmed spellings.
    speaker_agent.run.assert_called_once_with(expected)
