from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SummaryType(Enum):
    """Enum for summary types."""

    VERHANDLUNGSPROTOKOLL = "verhandlungsprotokoll"  # Negotiation/Process Protocol
    KURZPROTOKOLL = "kurzprotokoll"  # Short Protocol
    ERGEBNISPROTOKOLL = "ergebnisprotokoll"  # Result/Decision Protocol


class Summary(BaseModel):
    """Response model for summarization endpoint."""

    summary: str = Field(..., description="Generated summary of the transcript.")

    model_config = ConfigDict(extra="forbid")


class SummaryRequest(BaseModel):
    """Request model for summarization endpoint."""

    transcript: str = Field(..., min_length=1, max_length=32_000 * 4, description="Transcript to summarize.")
    summary_type: SummaryType | None = Field(None, description="Type of summary to generate.")

    model_config = ConfigDict(extra="forbid")
