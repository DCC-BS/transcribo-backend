from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KeywordType = Literal["person", "location", "object", "institution"]


class Keyword(BaseModel):
    """A special name or term from the transcript with a short explanation."""

    term: str = Field(..., description="The term exactly as it is written in the (corrected) transcript.")
    description: str = Field(
        ...,
        description="Short explanation of what the term most likely is; states uncertainty when unsure.",
    )
    type: KeywordType = Field(
        ...,
        description=(
            "What the term names: a person, a place (location), an organization/company/authority "
            "(institution), or anything else such as a product, tool, or domain term (object)."
        ),
    )

    model_config = ConfigDict(extra="forbid")
