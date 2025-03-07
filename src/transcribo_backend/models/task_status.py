from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatusEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class TaskStatus(BaseModel):
    task_id: str
    status: TaskStatusEnum = Field(default=TaskStatusEnum.IN_PROGRESS)
    created_at: datetime | None = None
    executed_at: datetime | None = None
    progress: float | None = None

    class Config:
        use_enum_values = True
