from pydantic import BaseModel, ConfigDict, Field

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection


class TranscriptPostProcessingResult(BaseModel):
    """Combined output of the transcript post-processing agent (single LLM call)."""

    corrections: list[TranscriptCorrection] = Field(
        ..., description="Global find/replace pairs; empty when the transcript is already consistent."
    )
    speaker_assignments: list[SpeakerNameAssignment] = Field(
        ..., description="One entry per speaker label in the transcript."
    )
    keywords: list[Keyword] = Field(
        ...,
        description="Special names and terms in the transcript whose identity or spelling may need review.",
    )

    model_config = ConfigDict(extra="forbid")
