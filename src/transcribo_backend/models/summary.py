from pydantic import BaseModel


class Summary(BaseModel):
    summary: str


class SummaryRequest(BaseModel):
    """Request model for summarization endpoint."""

    transcript: str
