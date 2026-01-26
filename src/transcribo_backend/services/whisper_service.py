import json
import uuid
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import HTTPException
from returns.future import future_safe

from transcribo_backend.models.progress import ProgressResponse
from transcribo_backend.models.response_format import ResponseFormat
from transcribo_backend.models.task_status import TaskStatus, TaskStatusEnum
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.audio_converter import (
    AudioConversionError,
    convert_to_mp3,
    is_mp3_format,
)
from transcribo_backend.utils.app_config import AppConfig


class WhisperService:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        one_day = 60 * 60 * 24
        self.taskId_to_progressId: TTLCache[str, str] = TTLCache(maxsize=1024, ttl=one_day)
        timeout = httpx.Timeout(10.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        api_key_header = {"Authorization": f"Bearer {self.app_config.api_key}"}
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits, headers=api_key_header)

    @future_safe
    async def transcribe_get_task_status(self, task_id: str) -> TaskStatus:
        """
        Checks the status of an ongoing transcription task.

        Args:
            task_id: The ID of the task to check

        Returns:
            TaskStatus: The current status of the task
        """
        if task_id not in self.taskId_to_progressId:
            raise HTTPException(status_code=404, detail="Task not found")
        whisper_url = self.app_config.whisper_url
        url = f"{whisper_url}/audio/transcriptions/task/status?task_id={task_id}"
        progress_url = f"{whisper_url}/progress/{self.taskId_to_progressId[task_id]}"

        # Get the status of the transcription task
        response = await self.client.get(url)
        if response.status_code == 404:
            return TaskStatus(task_id=task_id, status=TaskStatusEnum.FAILED)
        response.raise_for_status()

        progress_response = await self.client.get(progress_url)
        if progress_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Progress not found")
        progress_response.raise_for_status()

        progress = ProgressResponse(**progress_response.json())
        return TaskStatus(**response.json(), progress=progress.progress)

    @future_safe
    async def transcribe_get_task_result(self, task_id: str) -> TranscriptionResponse:
        """
        Retrieves the result of a completed transcription task.

        Args:
            task_id: The ID of the completed task

        Returns:
            TranscriptionVerboseJsonResponse: The parsed transcription result
        """
        whisper_url = self.app_config.whisper_url
        url = f"{whisper_url}/audio/transcriptions/task/get?task_id={task_id}"

        # Get the transcription result
        response = await self.client.get(url)
        response.raise_for_status()
        result_data = response.json()

        if task_id in self.taskId_to_progressId:
            del self.taskId_to_progressId[task_id]

        transcription = TranscriptionResponse(**result_data)
        for segment in transcription.segments:
            segment.text = segment.text.strip()
            segment.text = segment.text.replace("ß", "ss")
            segment.speaker = segment.speaker or "Unknown"
            segment.speaker = segment.speaker.strip().capitalize()

        return TranscriptionResponse(**result_data)

    @future_safe
    async def transcribe_retry_task(self, task_id: str) -> TaskStatus:
        """
        Retries a failed transcription task.

        Args:
            task_id: The ID of the task to retry

        Returns:
            TaskStatus: The updated status of the task
        """
        whisper_url = self.app_config.whisper_url
        url = f"{whisper_url}/audio/transcriptions/task/retry?task_id={task_id}"

        response = await self.client.post(url)
        response.raise_for_status()
        return TaskStatus(**response.json())

    @future_safe
    async def transcribe_cancel_task(self, task_id: str) -> TaskStatus:
        """
        Cancels an ongoing transcription task.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            TaskStatus: The updated status of the task
        """
        whisper_url = self.app_config.whisper_url
        url = f"{whisper_url}/audio/transcriptions/task/cancel?task_id={task_id}"

        response = await self.client.put(url)
        response.raise_for_status()
        return TaskStatus(**response.json())

    @future_safe
    async def transcribe_submit_task(
        self,
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
        whisper_url = self.app_config.whisper_url
        url = f"{whisper_url}/audio/transcriptions/task/submit"

        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Convert to MP3 if not already in MP3 format
        if not is_mp3_format(audio_data):
            try:
                # Convert with balanced quality settings
                # convert_to_mp3 now returns IOResult, we unwrap it to get the value or raise exception
                audio_data = convert_to_mp3(audio_data).unwrap()
            except AudioConversionError as e:
                raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}") from e

        # Prepare form data
        data: dict[str, Any] = {"model": model}
        files = {"file": ("audio.mp3", audio_data, "audio/mpeg")}

        progress_id = uuid.uuid4().hex
        data["progress_id"] = progress_id
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        data["response_format"] = response_format.value
        data["timestamp_granularities[]"] = timestamp_granularities
        data["temperature"] = str(temperature)

        if diarization_speaker_count:
            data["diarization_speaker_count"] = str(diarization_speaker_count)

        # Add boolean parameters
        data["vad_filter"] = str(vad_filter)
        data["diarization"] = str(diarization)

        # Add any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, list | dict):
                data[key] = json.dumps(value)
            else:
                data[key] = str(value)

        # Send the request
        response = await self.client.post(url, data=data, files=files)
        response.raise_for_status()
        status = TaskStatus(**response.json())
        self.taskId_to_progressId[status.task_id] = progress_id
        return status
