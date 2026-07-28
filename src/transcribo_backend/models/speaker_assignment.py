from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpeakerNameAssignment(BaseModel):
    """A single diarization label resolved to a human name, or left unknown."""

    # This model is both the LLM output schema and part of the public
    # TranscriptionResponse, so the description must stay true of the value the
    # API returns. The short form the prompt uses is documented where it is
    # introduced — in the agent instructions.
    speaker: str = Field(..., description="Diarization label as it appears in the transcript, e.g. 'Speaker_00'.")
    name: str | None = Field(
        default=None,
        description="Real name of this speaker, exactly as written in the transcript. None when no textual evidence exists.",
    )
    role: str | None = Field(
        default=None,
        description="Speaker's role or function if the transcript makes it clear (e.g. 'Dolmetscher', 'Moderatorin'). None when unclear.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="How certain the assignment is, 1.0 = explicit self-introduction."
    )
    evidence: str | None = Field(
        default=None, description="Short verbatim quote from the transcript that justifies the name or role."
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name", "role", "evidence", mode="before")
    @classmethod
    def _none_literals_to_none(cls, value: object) -> object:
        # Smaller LLMs sometimes emit the string "null"/"none" instead of JSON null.
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "unknown"}:
            return None
        return value
