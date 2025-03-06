import json
from typing import Any

import aiohttp
from fastapi import APIRouter

from transcribo_backend.models.response_format import ResponseFormat
from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.audio_converter import convert_to_wav, is_wav_format

from ..config import settings

router = APIRouter()

# BentoML API endpoint
BENTOML_API_URL = f"{settings.whisper_api}/audio/transcriptions"


async def transcribe_get_task_status(task_id: str) -> TaskStatus:
    """
    Checks the status of an ongoing transcription task.

    Args:
        task_id: The ID of the task to check

    Returns:
        TaskStatus: The current status of the task
    """
    url = f"{settings.whisper_api}/audio/transcriptions/task/status?task_id={task_id}"

    # Get the status of the transcription task
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        response.raise_for_status()
        return TaskStatus(**await response.json())


async def transcribe_get_task_result(task_id: str) -> TranscriptionResponse:
    """
    Retrieves the result of a completed transcription task.

    Args:
        task_id: The ID of the completed task

    Returns:
        TranscriptionVerboseJsonResponse: The parsed transcription result
    """
    url = f"{settings.whisper_api}/audio/transcriptions/task/get?task_id={task_id}"

    # Get the transcription result
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        response.raise_for_status()
        result_data = await response.json()
        return TranscriptionResponse(**result_data)


async def transcribe_retry_task(task_id: str) -> TaskStatus:
    """
    Retries a failed transcription task.

    Args:
        task_id: The ID of the task to retry

    Returns:
        TaskStatus: The updated status of the task
    """
    url = f"{settings.whisper_api}/audio/transcriptions/task/retry?task_id={task_id}"

    async with aiohttp.ClientSession() as session, session.post(url) as response:
        response.raise_for_status()
        return TaskStatus(**await response.json())


async def transcribe_cancel_task(task_id: str) -> TaskStatus:
    """
    Cancels an ongoing transcription task.

    Args:
        task_id: The ID of the task to cancel

    Returns:
        TaskStatus: The updated status of the task
    """
    url = f"{settings.whisper_api}/audio/transcriptions/task/cancel?task_id={task_id}"

    async with aiohttp.ClientSession() as session, session.put(url) as response:
        response.raise_for_status()
        return TaskStatus(**await response.json())


async def transcribe_submit_task(
    audio_data: bytes,
    file_format: str,
    model: str = "large-v2",
    language: str | None = None,
    prompt: str | None = None,
    response_format: ResponseFormat = ResponseFormat.JSON_DIARIZED,
    temperature: float | list[float] | None = None,
    vad_filter: bool = True,
    diarization: bool = True,
    diarization_speaker_count: int | None = None,
    timestamp_granularities: str = "segment",
    **kwargs: Any,
) -> TaskStatus:
    """
    Submits a new transcription task with additional parameters.

    Args:
        audio_data: The binary audio data to transcribe
        file_format: The format of the audio data
        model: The Whisper model to use
        language: The language code for transcription
        prompt: Optional prompt for the model
        response_format: Format of the output (enum: ResponseFormat)
        temperature: Temperature value(s) for sampling
        vad_filter: Whether to use voice activity detection
        diarization: Whether to separate speakers
        diarization_speaker_count: Number of speakers to separate
        **kwargs: Additional parameters to pass to the API

    Returns:
        TaskStatus: The status of the created task
    """
    url = f"{settings.whisper_api}/audio/transcriptions/task/submit"

    if temperature is None:
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Convert to WAV if not already in WAV format
    if not is_wav_format(audio_data):
        audio_data = convert_to_wav(audio_data, file_format)

    # Prepare form data
    form_data = aiohttp.FormData()
    form_data.add_field("file", audio_data, filename="audio.wav")
    form_data.add_field("model", model)

    if language:
        form_data.add_field("language", language)
    if prompt:
        form_data.add_field("prompt", prompt)
    form_data.add_field("response_format", response_format.value)  # Use the enum value
    form_data.add_field("timestamp_granularities[]", timestamp_granularities)

    # Handle temperature which could be a single float or a list
    if isinstance(temperature, list):
        for temp in temperature:
            form_data.add_field("temperature", str(temp))
    else:
        form_data.add_field("temperature", str(temperature))

    if diarization_speaker_count:
        form_data.add_field("diarization_speaker_count", str(diarization_speaker_count))

    # Add boolean parameters
    form_data.add_field("vad_filter", str(vad_filter))
    form_data.add_field("diarization", str(diarization))

    # Add any additional parameters
    for key, value in kwargs.items():
        if isinstance(value, list | dict):
            form_data.add_field(key, json.dumps(value))
        else:
            form_data.add_field(key, str(value))

    # Send the request
    async with aiohttp.ClientSession() as session, session.post(url, data=form_data) as response:
        response.raise_for_status()
        return TaskStatus(**await response.json())
