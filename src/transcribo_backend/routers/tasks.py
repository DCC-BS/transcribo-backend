"""Router for task-related endpoints (status and results)."""

from fastapi import APIRouter

from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.whisper_service import (
    transcribe_get_task_result,
    transcribe_get_task_status,
)

router = APIRouter(prefix="/task", tags=["tasks"])


@router.get("/{task_id}/status")
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Endpoint to get the status of a task by task_id.
    """
    status = await transcribe_get_task_status(task_id)
    return status


@router.get("/{task_id}/result")
async def get_task_result(task_id: str) -> TranscriptionResponse:
    """
    Endpoint to get the result of a task by task_id.
    """
    result = await transcribe_get_task_result(task_id)
    return result
