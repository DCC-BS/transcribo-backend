from http import HTTPStatus
from pathlib import Path
from typing import Annotated

from dcc_backend_common.fastapi_error_handling import ApiErrorCodes, ApiErrorException, api_error_exception
from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header, UploadFile
from returns.result import Failure, Success

from transcribo_backend.container import Container
from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.whisper_service import WhisperService

logger = get_logger(__name__)


@inject
def create_router(  # noqa: C901
    whisper_service: WhisperService = Provide[Container.whisper_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create the router for the transcription API.
    """
    logger.info("Creating transcription router")
    router = APIRouter()

    @router.get("/task/{task_id}/status")
    async def get_task_status(task_id: str) -> TaskStatus:  # ty:ignore[invalid-return-type]
        """
        Endpoint to get the status of a task by task_id.
        """
        result = await whisper_service.transcribe_get_task_status(task_id)
        match result:
            case Success(status):
                return status
            case Failure(error):
                logger.exception(f"Failed to get task status for {task_id}", exc_info=error)
                raise ApiErrorException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    error_code=ApiErrorCodes.INTERNAL_ERROR,
                    message="Failed to get task status",
                ) from error

    @router.get("/task/{task_id}/result")
    async def get_task_result(task_id: str) -> TranscriptionResponse:  # ty:ignore[invalid-return-type]
        """
        Endpoint to get the status of a task by task_id.
        """
        result = await whisper_service.transcribe_get_task_result(task_id)
        match result:
            case Success(transcription):
                return transcription
            case Failure(error):
                logger.exception(f"Failed to get task result for {task_id}", exc_info=error)
                raise ApiErrorException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    error_code=ApiErrorCodes.INTERNAL_ERROR,
                    message="Failed to get task result",
                ) from error

    @router.post("/transcribe")
    async def submit_transcribe(
        audio_file: UploadFile,
        num_speakers: int | None = None,
        language: str | None = None,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> TaskStatus:  # ty:ignore[invalid-return-type]
        """
        Endpoint to submit a transcription task.
        """

        if audio_file.content_type is None:
            raise api_error_exception(
                errorId=ApiErrorCodes.INVALID_REQUEST,
                status=HTTPStatus.BAD_REQUEST,
                debugMessage="Content type of the audio file is None",
            )

        if audio_file.filename is None:
            raise api_error_exception(
                errorId=ApiErrorCodes.INVALID_REQUEST,
                status=HTTPStatus.BAD_REQUEST,
                debugMessage="Filename of the audio file is None",
            )

        if not is_audio_file(audio_file.content_type) and not is_video_file(audio_file.content_type):
            raise api_error_exception(
                errorId=ApiErrorCodes.VALIDATION_ERROR,
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                debugMessage="Unsupported file type",
            )

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
        result = await whisper_service.transcribe_submit_task(
            audio_data, extension, diarization_speaker_count=num_speakers, language=language
        )

        match result:
            case Success(status):
                return status
            case Failure(error):
                logger.exception("Failed to submit transcription task", exc_info=error)

                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                error_code = ApiErrorCodes.INTERNAL_ERROR
                message = "Failed to submit transcription task"

                if isinstance(error, HTTPException):
                    status_code = error.status_code
                    if status_code == HTTPStatus.TOO_MANY_REQUESTS:
                        message = "Too many requests"
                    elif status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
                        message = "File is too large"

                raise api_error_exception(
                    errorId=error_code,
                    status=status_code,
                    message=message,
                ) from error

    return router
