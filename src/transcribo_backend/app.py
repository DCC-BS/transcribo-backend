import logging
from http import HTTPStatus
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from transcribo_backend.helpers.file_type import is_audio_file, is_video_file
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.whisper_service import (
    transcribe_get_task_result,
    transcribe_get_task_status,
    transcribe_submit_task,
)

logger = logging.getLogger(__name__)

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
async def submit_transcribe(audio_file: UploadFile) -> TaskStatus:
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

    # Submit the transcription task
    extension = Path(audio_file.filename).suffix.lower().strip(".")
    status = await transcribe_submit_task(audio_data, extension)
    return status


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("transcribo_backend:app", host="127.0.0.1", port=8000, reload=True)
