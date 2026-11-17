"""The place name agent's decision for each numbered location term of a transcript."""

from pydantic import BaseModel, ConfigDict, Field


class PlaceNameChoice(BaseModel):
    """Which numbered candidate a numbered term refers to, if any."""

    term: int = Field(..., description="Number of the term.")
    candidate: int | None = Field(
        ..., description="Number of the chosen candidate; null when no candidate is meant or when unsure."
    )

    model_config = ConfigDict(extra="forbid")


class PlaceNameChoices(BaseModel):
    """One choice per given term."""

    choices: list[PlaceNameChoice]

    model_config = ConfigDict(extra="forbid")
