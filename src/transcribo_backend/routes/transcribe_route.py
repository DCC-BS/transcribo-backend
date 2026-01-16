from http import HTTPStatus
from pathlib import Path
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header, HTTPException, UploadFile

from transcribo_backend.container import Container
from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.whisper_service import WhisperService

logger = get_logger(__name__)


@inject
def create_router(  # noqa: C901
    whisper_service: WhisperService = Provide[Container.whisper_service.provider],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create the router for the transcription API.
    """
    logger.info("Creating transcription router")
    router = APIRouter()

    @router.get("/task/{task_id}/status")
    async def get_task_status(task_id: str) -> TaskStatus:
        """
        Endpoint to get the status of a task by task_id.
        """
        status = await whisper_service.transcribe_get_task_status(task_id)
        return status

    @router.get("/task/{task_id}/result")
    async def get_task_result(task_id: str) -> TranscriptionResponse:
        """
        Endpoint to get the status of a task by task_id.
        """
        status = await whisper_service.transcribe_get_task_result(task_id)
        return status

    @router.post("/transcribe")
    async def submit_transcribe(
        audio_file: UploadFile,
        num_speakers: int | None = None,
        language: str | None = None,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> TaskStatus:
        """
        Endpoint to submit a transcription task.
        """

        if audio_file.content_type is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Content type of the audio file is None")

        if audio_file.filename is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Filename of the audio file is None")

        if not is_audio_file(audio_file.content_type) and not is_video_file(audio_file.content_type):
            raise HTTPException(status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")

        # Read the uploaded file content
        audio_data = await audio_file.read()
        usage_tracking_service.log_event(
            module="transcribe_route",
            func="transcribe",
            user_id=x_client_id or "unknown",
            num_speakers=num_speakers,
            file_size=len(audio_data),
        )

        # Submit the transcription task
        extension = Path(audio_file.filename).suffix.lower().strip(".")
        try:
            status = await whisper_service.transcribe_submit_task(
                audio_data, extension, diarization_speaker_count=num_speakers, language=language
            )
        except HTTPException as e:
            logger.exception("Failed to submit transcription task", exc_info=e)
            if e.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="Too many requests") from None
            if e.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
                raise HTTPException(
                    status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="File is too large"
                ) from None
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to submit transcription task"
            ) from None
        except Exception as e:
            logger.exception("Failed to submit transcription task", exc_info=e)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to submit transcription task"
            ) from None
        return status

    return router
