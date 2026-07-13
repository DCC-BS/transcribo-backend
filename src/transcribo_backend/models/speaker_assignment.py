from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpeakerNameAssignment(BaseModel):
    """A single diarization label resolved to a human name, or left unknown."""

    speaker: str = Field(..., description="Diarization label as it appears in the transcript, e.g. 'SPEAKER_00'.")
    name: str | None = Field(
        None,
        description="Real name of this speaker, exactly as written in the transcript. None when no textual evidence exists.",
    )
    role: str | None = Field(
        None,
        description="Speaker's role or function if the transcript makes it clear (e.g. 'Dolmetscher', 'Moderatorin'). None when unclear.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="How certain the assignment is, 1.0 = explicit self-introduction."
    )
    evidence: str | None = Field(
        None, description="Short verbatim quote from the transcript that justifies the name or role."
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name", "role", "evidence", mode="before")
    @classmethod
    def _none_literals_to_none(cls, value: object) -> object:
        # Smaller LLMs sometimes emit the string "null"/"none" instead of JSON null.
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "unknown"}:
            return None
        return value


class SpeakerAssignmentResult(BaseModel):
    """Response model for the speaker assignment endpoint."""

    assignments: list[SpeakerNameAssignment] = Field(..., description="One entry per speaker label in the transcript.")

    model_config = ConfigDict(extra="forbid")
