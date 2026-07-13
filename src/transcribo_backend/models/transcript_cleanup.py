from pydantic import BaseModel, ConfigDict, Field

from transcribo_backend.models.keywords import Keyword


class TranscriptCorrection(BaseModel):
    """One global surface-form replacement proposed by the cleanup agent."""

    original: str = Field(..., description="Exact wrong surface form as it appears in the transcript.")
    corrected: str = Field(..., description="The consistent, corrected surface form.")
    reason: str = Field(..., description="Short justification: dominant variant, hotword match, or formatting rule.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="How certain the correction is.")

    model_config = ConfigDict(extra="forbid")


class TranscriptCleanupResult(BaseModel):
    """What the cleanup agent proposes for a transcript: corrections + keywords."""

    corrections: list[TranscriptCorrection] = Field(
        ..., description="Global find/replace pairs; empty when the transcript is already consistent."
    )
    keywords: list[Keyword] = Field(
        ...,
        description="Special names and terms in the transcript whose identity or spelling may need review.",
    )

    model_config = ConfigDict(extra="forbid")
