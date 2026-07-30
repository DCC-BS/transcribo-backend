"""Status of an asynchronous transcription task."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TaskStatusEnum(StrEnum):
    """Lifecycle states a Whisper transcription task can be in."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(BaseModel):
    """State of a transcription task, optionally enriched with a progress percentage."""

    task_id: str
    status: TaskStatusEnum = Field(default=TaskStatusEnum.IN_PROGRESS)
    created_at: datetime | None = None
    executed_at: datetime | None = None
    progress: float | None = None

    class Config:
        """Serialize the status as its plain string value."""

        use_enum_values = True
