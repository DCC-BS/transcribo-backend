from http import HTTPStatus
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, UploadFile

from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.summary import Summary, SummaryRequest
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services import summary_service
from transcribo_backend.services.whisper_service import (
    transcribe_get_task_result,
    transcribe_get_task_status,
    transcribe_submit_task,
)
from transcribo_backend.utils.logger import get_logger, init_logger
from transcribo_backend.utils.usage_tracking import get_pseudonymized_user_id

init_logger()
logger = get_logger("app")

# Initialize FastAPI app
app = FastAPI()


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Endpoint to get the status of a task by task_id.
    """
    status = await transcribe_get_task_status(task_id)
    return status


@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str) -> TranscriptionResponse:
    """
    Endpoint to get the status of a task by task_id.
    """
    status = await transcribe_get_task_result(task_id)
    return status


@app.post("/transcribe")
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

    # Extract X-Client-Id from the request headers
    pseudonym_id = get_pseudonymized_user_id(x_client_id)
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
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="Too many requests") from None
        if e.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="File is too large") from None
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to submit transcription task"
        ) from None
    except Exception as e:
        logger.exception("Failed to submit transcription task", exc_info=e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to submit transcription task"
        ) from None
    return status


@app.post("/summarize")
async def summarize(request: SummaryRequest, x_client_id: Annotated[str | None, Header()] = None) -> Summary:
    """
    Endpoint to summarize a text.
    """
    model_context_length = 32_000

    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Transcript is empty")
    if len(request.transcript) > model_context_length * 4:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Transcript is too long. Maximum length is {model_context_length * 4} characters.",
        )
    # Extract X-Client-Id from the request headers
    pseudonym_id = get_pseudonymized_user_id(x_client_id)
    logger.info(
        "app_event",
        extra={"pseudonym_id": pseudonym_id, "event": "summarize", "transcript_length": len(request.transcript)},
    )

    try:
        summary = await summary_service.summarize(request.transcript)
    except Exception as e:
        logger.exception("Failed to summarize transcript", exc_info=e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to generate summary") from None
    else:
        return summary


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("transcribo_backend:app", host="127.0.0.1", port=8000, reload=True)
