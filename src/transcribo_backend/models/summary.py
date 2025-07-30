from pydantic import BaseModel, Field


class Summary(BaseModel):
    """Response model for summarization endpoint."""

    summary: str = Field(..., description="Generated summary of the transcript.")

    class Config:
        extra = "forbid"


class SummaryRequest(BaseModel):
    """Request model for summarization endpoint."""

    transcript: str = Field(..., min_length=1, max_length=32_000 * 4, description="Transcript to summarize.")

    class Config:
        extra = "forbid"
