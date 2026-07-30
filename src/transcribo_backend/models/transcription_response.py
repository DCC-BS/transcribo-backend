"""Request and response models of the transcription endpoints."""

from pydantic import BaseModel, Field

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection


class TaskResultRequest(BaseModel):
    """Request for a task result, carrying the user's confirmed vocabulary."""

    # Bounded: the vocabulary is client-supplied and every entry is compared against
    # every inferred speaker name, so an unbounded list is a cheap way to make the
    # request expensive. Far above any realistic user vocabulary.
    keywords: list[Keyword] = Field(default_factory=list, max_length=500)


class Word(BaseModel):
    """A single recognized word with its timing and recognition probability."""

    start: float
    end: float
    word: str
    probability: float
    speaker: str | None = None


class Segment(BaseModel):
    """One diarized utterance of the transcript."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    # Word-level timestamps/probabilities are internal only for the cleanup prompt
    words: list[Word] | None = Field(default=None, exclude=True)


class TranscriptionResponse(BaseModel):
    """A finished transcription plus whatever post-processing could derive from it."""

    segments: list[Segment]
    # Concise title inferred from the transcript. None when post-processing
    # failed or the topic was too ambiguous.
    title: str | None = None
    # One entry per diarization label, resolved to a real name where the
    # transcript provides evidence. None when assignment was not attempted
    # or failed (the transcription itself is valid without it).
    speaker_assignments: list[SpeakerNameAssignment] | None = None
    # Consistency corrections that were applied to the segment texts. None
    # when cleanup was not attempted or failed.
    applied_corrections: list[TranscriptCorrection] | None = None
    # Proposed keywords of special names/terms for human review. None when
    # post-processing was not attempted or failed.
    keywords: list[Keyword] | None = None
