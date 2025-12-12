from http import HTTPStatus
from pathlib import Path
from typing import Annotated

from backend_common.fastapi_error_handling import api_error_exception
from fastapi import APIRouter, Header, HTTPException, UploadFile

from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.error_codes import TranscriboErrorCodes
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.services.whisper_service import transcribe_submit_task
from transcribo_backend.utils.logger import get_logger
from transcribo_backend.utils.usage_tracking import get_pseudonymized_user_id

logger = get_logger("transcription_router")

router = APIRouter(prefix="/transcribe", tags=["transcription"])


@router.post("")
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
        raise api_error_exception(
            errorId=TranscriboErrorCodes.AUDIO_FILE_NO_CONTENT_TYPE,
            status=HTTPStatus.BAD_REQUEST,
            debugMessage="Content type of the audio file is None",
        )

    if audio_file.filename is None:
        raise api_error_exception(
            errorId=TranscriboErrorCodes.AUDIO_FILE_NO_FILENAME,
            status=HTTPStatus.BAD_REQUEST,
            debugMessage="Filename of the audio file is None",
        )

    if not is_audio_file(audio_file.content_type) and not is_video_file(audio_file.content_type):
        raise api_error_exception(
            errorId=TranscriboErrorCodes.UNSUPPORTED_FILE_TYPE,
            status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            debugMessage="Unsupported file type",
        )

    # Read the uploaded file content
    audio_data = await audio_file.read()

    # Extract X-Client-Id from the request headers
    pseudonym_id = get_pseudonymized_user_id(x_client_id or "unknown")
    logger.info(
        "app_event",
        extra={
            "pseudonym_id": pseudonym_id,
            "event": "transcribe",
            "num_speakers": num_speakers,
            "file_size": len(audio_data),
        },
    )

    # Submit the transcription task
    extension = Path(audio_file.filename).suffix.lower().strip(".")
    try:
        status = await transcribe_submit_task(
            audio_data, extension, diarization_speaker_count=num_speakers, language=language
        )
    except HTTPException as e:
        logger.exception("Failed to submit transcription task", exc_info=e)
        if e.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            raise api_error_exception(
                errorId=TranscriboErrorCodes.TOO_MANY_REQUESTS,
                status=HTTPStatus.TOO_MANY_REQUESTS,
                debugMessage=f"Failed to submit transcription task: Too many requests (HTTPException: {e})",
            ) from e
        if e.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            raise api_error_exception(
                errorId=TranscriboErrorCodes.FILE_TOO_LARGE,
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                debugMessage=f"Failed to submit transcription task: File is too large (HTTPException: {e})",
            ) from e
        raise api_error_exception(
            errorId=TranscriboErrorCodes.TRANSCRIPTION_TASK_FAILED,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            debugMessage=f"Failed to submit transcription task (HTTPException: {e})",
        ) from e
    except Exception as e:
        logger.exception("Failed to submit transcription task", exc_info=e)
        raise api_error_exception(
            errorId=TranscriboErrorCodes.TRANSCRIPTION_TASK_FAILED,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            debugMessage=f"Failed to submit transcription task: {e}",
        ) from e
    return status
