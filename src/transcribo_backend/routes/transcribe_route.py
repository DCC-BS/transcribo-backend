from http import HTTPStatus
from typing import Annotated, Any

import httpx
from dcc_backend_common.fastapi_error_handling import ApiErrorCodes, api_error_exception
from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Form, Header, HTTPException, UploadFile
from returns.io import IOSuccess

from transcribo_backend.container import Container
from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.whisper_service import WhisperService


def _is_not_found_error(error: Exception) -> bool:
    """Check if the error represents a 'not found' condition (404)."""
    if isinstance(error, HTTPException):
        return error.status_code == HTTPStatus.NOT_FOUND
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == HTTPStatus.NOT_FOUND
    return False


@inject
def create_router(  # noqa: C901
    whisper_service: WhisperService = Provide[Container.whisper_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """
    Create the router for the transcription API.
    """
    logger = get_logger(__name__)
    logger.info("Creating transcription router")
    router = APIRouter()

    def _unwrap_or_raise(result: Any, *, log_message: str, not_found_message: str, error_message: str) -> Any:
        """Return the value of a successful IOResult, or raise the mapped API error."""
        if isinstance(result, IOSuccess):
            return result.unwrap()._inner_value

        error = result.failure()._inner_value
        logger.exception(log_message, exc_info=error)
        if _is_not_found_error(error):
            raise api_error_exception(
                errorId=ApiErrorCodes.RESOURCE_NOT_FOUND,
                status=HTTPStatus.NOT_FOUND,
                debugMessage=not_found_message,
            ) from error

        raise api_error_exception(
            errorId=ApiErrorCodes.UNEXPECTED_ERROR,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            debugMessage=error_message,
        ) from error

    @router.get("/task/{task_id}/status")
    async def get_task_status(task_id: str) -> TaskStatus:
        """
        Endpoint to get the status of a task by task_id.
        """
        result = await whisper_service.transcribe_get_task_status(task_id)
        return _unwrap_or_raise(
            result,
            log_message=f"Failed to get task status for {task_id}",
            not_found_message=f"Task {task_id} not found",
            error_message="Failed to get task status",
        )

    @router.get("/task/{task_id}/result")
    async def get_task_result(task_id: str) -> TranscriptionResponse:
        """
        Endpoint to get the result of a task by task_id.
        """
        result = await whisper_service.transcribe_get_task_result(task_id)
        return _unwrap_or_raise(
            result,
            log_message=f"Failed to get task result for {task_id}",
            not_found_message=f"Task result for {task_id} not found",
            error_message="Failed to get task result",
        )

    @router.post("/transcribe")
    async def submit_transcribe(
        audio_file: UploadFile,
        num_speakers: Annotated[int | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        x_client_id: Annotated[str | None, Header()] = None,
    ) -> TaskStatus:
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

        # Reject oversized uploads early, before streaming any bytes.
        max_upload_bytes = whisper_service.app_config.max_upload_bytes
        if audio_file.size is not None and audio_file.size > max_upload_bytes:
            raise api_error_exception(
                errorId=ApiErrorCodes.VALIDATION_ERROR,
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                debugMessage="File is too large",
            )

        usage_tracking_service.log_event(
            module="transcribe_route",
            func="transcribe",
            user_id=x_client_id or "unknown",
            num_speakers=num_speakers,
            file_size=audio_file.size,
        )

        # Submit the transcription task. The file is streamed to disk and forwarded to
        # Whisper without ever being fully loaded into memory.
        try:
            result = await whisper_service.transcribe_submit_task(
                audio_file,
                diarization_speaker_count=num_speakers,
                language=language,
                max_upload_bytes=max_upload_bytes,
            )
        finally:
            await audio_file.close()

        if isinstance(result, IOSuccess):
            return result.unwrap()._inner_value

        error = result.failure()._inner_value
        logger.exception("Failed to submit transcription task", exc_info=error)

        status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        error_code = ApiErrorCodes.UNEXPECTED_ERROR
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
            debugMessage=message,
        ) from error

    return router
