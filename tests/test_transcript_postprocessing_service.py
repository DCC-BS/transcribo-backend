from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from transcribo_backend.agents.transcript_postprocessing_agent import TRANSCRIPT_POSTPROCESSING_INSTRUCTIONS
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection
from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult
from transcribo_backend.models.transcription_response import Segment, Word
from transcribo_backend.services.transcript_postprocessing_service import (
    TranscriptPostProcessingService,
    apply_corrections,
    apply_corrections_to_names,
    apply_keyword_spellings_to_names,
    build_postprocessing_transcript,
    decode_speaker_labels,
    encode_speaker_label,
    enumerate_roles,
)
from transcribo_backend.utils.app_config import AppConfig


def _word(token: str, probability: float) -> Word:
    return Word(start=0.0, end=1.0, word=token, probability=probability)


def _make_service(
    agent_result: TranscriptPostProcessingResult,
) -> tuple[TranscriptPostProcessingService, MagicMock]:
    """Service wired with the single mocked post-processing agent."""
    app_config = MagicMock(spec=AppConfig)
    agent = MagicMock()
    agent.run = AsyncMock(return_value=agent_result)
    service = TranscriptPostProcessingService(app_config, agent)
    return service, agent


def test_postprocessing_title_contract_is_short():
    assert "Use 3-6 words and no more than 60 characters." in TRANSCRIPT_POSTPROCESSING_INSTRUCTIONS

    with pytest.raises(ValidationError):
        TranscriptPostProcessingResult(
            title="x" * 61,
            speaker_assignments=[],
            corrections=[],
            keywords=[],
        )


def test_encode_speaker_label_round_trip():
    """Short labels save prompt tokens; the original spelling must come back."""
    segments = [
        Segment(start=0.0, end=1.0, text="a", speaker="Speaker_00"),
        Segment(start=1.0, end=2.0, text="b", speaker="SPEAKER_07"),
        Segment(start=2.0, end=3.0, text="c", speaker="Speaker_100"),
        Segment(start=3.0, end=4.0, text="d", speaker=None),
    ]

    assert encode_speaker_label("Speaker_00") == "S00"
    # whisper_service .capitalize()s labels, but the diarizer's own upper-case
    # form must shorten too — otherwise the saving silently disappears.
    assert encode_speaker_label("SPEAKER_07") == "S07"
    assert encode_speaker_label("Speaker_100") == "S100"
    # Anything that is not a numbered speaker label is left alone.
    assert encode_speaker_label("Unknown") == "Unknown"

    assignments = [
        SpeakerNameAssignment(speaker="S00", name="Anna", confidence=0.9),
        SpeakerNameAssignment(speaker="S07", name="Beat", confidence=0.9),
        SpeakerNameAssignment(speaker="S100", name=None, confidence=0.0),
        SpeakerNameAssignment(speaker="Unknown", name=None, confidence=0.0),
    ]
    decode_speaker_labels(assignments, segments)
    assert [a.speaker for a in assignments] == ["Speaker_00", "SPEAKER_07", "Speaker_100", "Unknown"]


def test_decode_speaker_labels_never_invents_a_label():
    """Decoding is a lookup against the transcript, so it only ever restores labels that exist."""
    segments = [Segment(start=0.0, end=1.0, text="a", speaker="Speaker_00")]

    # Answered with the long label instead of the short one: already correct.
    assignments = [SpeakerNameAssignment(speaker="Speaker_00", name="Anna", confidence=0.9)]
    decode_speaker_labels(assignments, segments)
    assert assignments[0].speaker == "Speaker_00"

    # Hallucinated a label that is not in the transcript: no correct expansion
    # exists, so it is passed through rather than fabricated into "Speaker_09".
    invented = [SpeakerNameAssignment(speaker="S09", name="Niemand", confidence=0.5)]
    decode_speaker_labels(invented, segments)
    assert invented[0].speaker == "S09"

    # A diarizer that natively emits "S0" keeps it — an inverse regex would
    # wrongly expand it to "Speaker_0".
    native_short = [SpeakerNameAssignment(speaker="S0", name="Anna", confidence=0.9)]
    decode_speaker_labels(native_short, [Segment(start=0.0, end=1.0, text="a", speaker="S0")])
    assert native_short[0].speaker == "S0"


def test_build_transcript_merges_speakers():
    segments = [
        Segment(start=0.0, end=1.0, text=" Hallo zusammen. ", speaker="Speaker_00"),
        Segment(start=1.0, end=2.0, text="Wir reden über Jobshipping.", speaker="Speaker_00"),
        Segment(start=2.0, end=3.0, text="Genau.", speaker="Speaker_01"),
        Segment(start=3.0, end=4.0, text="Ohne Sprecher.", speaker=None),
    ]

    # Labels are rendered short; merging still compares the original labels.
    transcript = build_postprocessing_transcript(segments)
    assert transcript == "S00: Hallo zusammen. Wir reden über Jobshipping.\nS01: Genau.\nUnknown: Ohne Sprecher."

    clamped = build_postprocessing_transcript(segments, max_chars=55)
    assert clamped == "S00: Hallo zusammen. Wir reden über Jobshipping."


def test_build_transcript_clamp_keeps_head_and_tail():
    segments = [
        Segment(start=0.0, end=1.0, text="a" * 20, speaker="S0"),
        Segment(start=1.0, end=2.0, text="b" * 20, speaker="S1"),
        Segment(start=2.0, end=3.0, text="c" * 20, speaker="S2"),
        Segment(start=3.0, end=4.0, text="Danke, Anna.", speaker="S3"),
    ]

    # Budget forces a cut: the head keeps the opening, the tail keeps the
    # sign-off (where names are often mentioned), the middle becomes […].
    clamped = build_postprocessing_transcript(segments, max_chars=60)
    assert clamped == f"S0: {'a' * 20}\n[…]\nS3: Danke, Anna."


def test_build_transcript_marks_uncertain_words():
    segments = [
        Segment(
            start=0.0,
            end=1.0,
            text="Das war gut. Wir reden über Jobshipping.",
            speaker="Speaker_00",
            words=[
                # Short function words stay unmarked even at low probability.
                _word(" Das", 0.1),
                _word(" gut", 0.2),
                _word(" Wir", 0.99),
                # A token that only occurs inside another word ("reden")
                # must never be marked mid-word.
                _word(" eden", 0.1),
                _word(" Jobshipping", 0.2),
            ],
        ),
        Segment(start=1.0, end=2.0, text="Genau.", speaker="Speaker_01", words=None),
    ]

    transcript = build_postprocessing_transcript(segments, mark_uncertain=True)
    assert transcript == "S00: Das war gut. Wir reden über ⟨Jobshipping⟩.\nS01: Genau."

    # Without the flag the marks stay off.
    assert "⟨" not in build_postprocessing_transcript(segments)


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


def test_apply_corrections_formats_phone_email_and_serial():
    """Formatting-rule pairs pass the letter-word guard: phone and serial pairs
    contain no letter words, and the spoken e-mail symbols are exempt."""
    segments = [
        Segment(start=0.0, end=1.0, text="Erreichbar unter 079, 123, 45, 67.", speaker="A"),
        Segment(start=1.0, end=2.0, text="Schreibt an info at basel punkt ch bitte.", speaker="B"),
        Segment(start=2.0, end=3.0, text="Die Seriennummer lautet 12B, 34, 17 18.", speaker="A"),
    ]

    corrections = [
        TranscriptCorrection(
            original="079, 123, 45, 67", corrected="079 123 45 67", reason="phone rule", confidence=0.9
        ),
        TranscriptCorrection(
            original="info at basel punkt ch", corrected="info@basel.ch", reason="email rule", confidence=0.9
        ),
        TranscriptCorrection(
            original="12B, 34, 17 18", corrected="12B-34-17-18", reason="serial number rule", confidence=0.9
        ),
    ]

    applied = apply_corrections(segments, corrections)

    assert segments[0].text == "Erreichbar unter 079 123 45 67."
    assert segments[1].text == "Schreibt an info@basel.ch bitte."
    assert segments[2].text == "Die Seriennummer lautet 12B-34-17-18."
    assert len(applied) == 3


def test_apply_keyword_spellings_snaps_close_names():
    assignments = [
        SpeakerNameAssignment(speaker="A", name="Lena Feldman", confidence=0.9),
        SpeakerNameAssignment(speaker="B", name="Petra Muster", confidence=0.9),
        SpeakerNameAssignment(speaker="C", name=None, role="Moderator", confidence=0.7),
    ]
    keywords = [Keyword(term="Lena Feldmann", description="", type="person")]

    apply_keyword_spellings_to_names(assignments, keywords)

    assert assignments[0].name == "Lena Feldmann"
    # A clearly different name stays untouched.
    assert assignments[1].name == "Petra Muster"
    assert assignments[2].name is None


def test_apply_corrections_to_names():
    assignments = [
        SpeakerNameAssignment(speaker="A", name="Pete Maier", confidence=0.9),
        SpeakerNameAssignment(speaker="B", name=None, role="Reporter", confidence=0.7),
    ]
    corrections = [
        TranscriptCorrection(original="Pete Maier", corrected="Peter Meier", confidence=0.9),
    ]

    apply_corrections_to_names(assignments, corrections)

    assert assignments[0].name == "Peter Meier"
    assert assignments[1].name is None


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
async def test_post_process_single_call_returns_all():
    assignments = [
        SpeakerNameAssignment(speaker="A", name="Anna", confidence=0.9, evidence="Ich bin Anna."),
    ]
    agent_result = TranscriptPostProcessingResult(
        title="Dropshipping im Gespräch",
        speaker_assignments=assignments,
        corrections=[
            TranscriptCorrection(original="Jobshipping", corrected="Dropshipping", reason="dominant", confidence=0.9),
            TranscriptCorrection(original="toll", corrected="super", reason="style", confidence=0.4),
        ],
        keywords=[Keyword(term="Dropshipping", description="Online-Handelsmodell", type="object")],
    )

    service, agent = _make_service(agent_result)
    segments = [Segment(start=0.0, end=1.0, text="Jobshipping ist toll.", speaker="A")]

    result_io = await service.post_process(segments)
    result = result_io.unwrap()._inner_value

    # One call on the raw labels and uncorrected text.
    agent.run.assert_called_once_with("A: Jobshipping ist toll.")
    # Corrections are applied to the segments afterwards.
    assert segments[0].text == "Dropshipping ist toll."

    # Only the applied correction is reported; the low-confidence one is dropped.
    assert [c.original for c in result.corrections] == ["Jobshipping"]
    assert result.speaker_assignments == assignments
    assert result.keywords[0].term == "Dropshipping"
    assert result.title == "Dropshipping im Gespräch"


@pytest.mark.anyio
async def test_post_process_returns_original_speaker_labels():
    """The prompt uses short labels, the API contract keeps the diarization ones."""
    agent_result = TranscriptPostProcessingResult(
        title="Sitzung",
        speaker_assignments=[SpeakerNameAssignment(speaker="S00", name="Anna", confidence=0.9)],
        corrections=[],
        keywords=[],
    )

    service, agent = _make_service(agent_result)
    segments = [Segment(start=0.0, end=1.0, text="Ich bin Anna.", speaker="Speaker_00")]

    result_io = await service.post_process(segments)
    result = result_io.unwrap()._inner_value

    # Short label goes out to the model ...
    agent.run.assert_called_once_with("S00: Ich bin Anna.")
    # ... the original label comes back to the caller.
    assert result.speaker_assignments[0].speaker == "Speaker_00"
    assert segments[0].speaker == "Speaker_00"


@pytest.mark.anyio
async def test_post_process_never_changes_segment_count_timestamps_or_words():
    """Cleanup only rewrites texts: count, start/end, speaker labels, and the
    word-level fragments must come out exactly as they went in."""
    agent_result = TranscriptPostProcessingResult(
        title="Dropshipping im Gespräch",
        speaker_assignments=[SpeakerNameAssignment(speaker="A", name="Anna", confidence=0.9)],
        corrections=[
            TranscriptCorrection(original="Jobshipping", corrected="Dropshipping", reason="dominant", confidence=0.9),
        ],
        keywords=[],
    )
    service, _agent = _make_service(agent_result)

    words = [_word(" Jobshipping", 0.2), _word(" toll", 0.9)]
    segments = [
        Segment(start=0.0, end=1.5, text="Jobshipping ist toll.", speaker="A", words=words),
        Segment(start=1.5, end=3.0, text="Genau.", speaker="B", words=None),
    ]

    await service.post_process(segments)

    assert len(segments) == 2
    assert [(s.start, s.end, s.speaker) for s in segments] == [(0.0, 1.5, "A"), (1.5, 3.0, "B")]
    # Text corrected, word fragments untouched (still the recognizer's output).
    assert segments[0].text == "Dropshipping ist toll."
    assert segments[0].words == words
    assert segments[1].words is None


@pytest.mark.anyio
async def test_post_process_propagates_agent_failure():
    app_config = MagicMock(spec=AppConfig)
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("llm down"))
    service = TranscriptPostProcessingService(app_config, agent)

    result_io = await service.post_process([Segment(start=0.0, end=1.0, text="Hallo.", speaker="A")])

    error = result_io.failure()._inner_value
    assert isinstance(error, RuntimeError)
    assert str(error) == "llm down"


@pytest.mark.anyio
async def test_post_process_appends_user_keywords_to_prompt():
    service, agent = _make_service(
        TranscriptPostProcessingResult(title=None, speaker_assignments=[], corrections=[], keywords=[]),
    )
    segments = [Segment(start=0.0, end=1.0, text="Die Bibos Academy.", speaker="A")]
    keywords = [Keyword(term="BeeBoss", description="Name der Dropshipping-Academy", type="institution")]

    await service.post_process(segments, keywords=keywords)

    expected = "A: Die Bibos Academy.\n\nUser keywords:\nBeeBoss: Name der Dropshipping-Academy"
    agent.run.assert_called_once_with(expected)
