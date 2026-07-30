"""Combined output schema of the transcript post-processing agent."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection

# Mirrors the title contract stated in the agent instructions (Task 0).
_MIN_TITLE_WORDS = 3
_MAX_TITLE_WORDS = 6
_MAX_TITLE_CHARS = 60


class TranscriptPostProcessingResult(BaseModel):
    """Combined output of the transcript post-processing agent (single LLM call).

    Field order follows the task order in the prompt.
    """

    title: str | None = Field(
        ...,
        description="Short 3-6 word title in the transcript language, or null when the topic is ambiguous.",
    )
    speaker_assignments: list[SpeakerNameAssignment] = Field(
        ..., description="One entry per speaker label in the transcript."
    )
    corrections: list[TranscriptCorrection] = Field(
        ..., description="Global find/replace pairs; empty when the transcript is already consistent."
    )
    keywords: list[Keyword] = Field(
        ...,
        description="Special names and terms in the transcript whose identity or spelling may need review.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("title")
    @classmethod
    def _drop_title_off_contract(cls, value: str | None) -> str | None:
        """Discard a title that breaks the prompt's 3-6 word / 60 char contract.

        Dropped rather than rejected: the field is nullable by design, and a
        sentence-shaped title is not worth failing the call over — that would
        also throw away the speaker names and corrections from the same response.
        """
        if value is None:
            return None
        title = value.strip()
        if len(title) > _MAX_TITLE_CHARS or not (_MIN_TITLE_WORDS <= len(title.split()) <= _MAX_TITLE_WORDS):
            return None
        return title
