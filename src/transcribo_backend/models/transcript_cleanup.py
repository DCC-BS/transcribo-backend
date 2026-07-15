from pydantic import BaseModel, ConfigDict, Field


class TranscriptCorrection(BaseModel):
    """One global surface-form replacement proposed by the cleanup agent."""

    original: str = Field(..., description="Exact wrong surface form as it appears in the transcript.")
    corrected: str = Field(..., description="The consistent, corrected surface form.")
    reason: str = Field(
        default="", description="Short justification: dominant variant, keyword match, or formatting rule."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="How certain the correction is.")

    model_config = ConfigDict(extra="forbid")
