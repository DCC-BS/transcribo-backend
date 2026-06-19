import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import HTTPException, UploadFile
from returns.future import future_safe
from returns.pipeline import is_successful

from transcribo_backend.models.progress import ProgressResponse
from transcribo_backend.models.response_format import ResponseFormat
from transcribo_backend.models.task_status import TaskStatus, TaskStatusEnum
from transcribo_backend.models.transcription_response import TranscriptionResponse
from transcribo_backend.services.audio_converter import (
    convert_to_mp3,
    is_mp3_format,
)
from transcribo_backend.utils.app_config import AppConfig

# Size of chunks streamed from the upload to disk.
_STREAM_CHUNK_BYTES = 1024 * 1024
# Number of leading bytes inspected to detect the MP3 container.
_SNIFF_BYTES = 1024


class WhisperService:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        one_day = 60 * 60 * 24
        self.taskId_to_progressId: TTLCache[str, str] = TTLCache[str, str](maxsize=1024, ttl=one_day)
        # Short connect, but long write/read so large multi-hour uploads do not time out.
        timeout = httpx.Timeout(connect=10.0, write=None, read=300.0, pool=10.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        api_key_header = {"Authorization": f"Bearer {self.app_config.llm_api_key}"}
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits, headers=api_key_header)

    async def aclose(self) -> None:
        """Close the HTTP client to prevent connection leaks."""
        await self.client.aclose()

    def _task_endpoint(self, path: str) -> str:
        """Build a Whisper task endpoint URL (e.g. ``status?task_id=...``)."""
        return f"{self.app_config.whisper_url}/audio/transcriptions/task/{path}"

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
        url = self._task_endpoint(f"status?task_id={task_id}")
        progress_url = f"{self.app_config.whisper_url}/progress/{self.taskId_to_progressId[task_id]}"

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
        url = self._task_endpoint(f"get?task_id={task_id}")

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

        return transcription

    @future_safe
    async def transcribe_retry_task(self, task_id: str) -> TaskStatus:
        """
        Retries a failed transcription task.

        Args:
            task_id: The ID of the task to retry

        Returns:
            TaskStatus: The updated status of the task
        """
        url = self._task_endpoint(f"retry?task_id={task_id}")

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
        url = self._task_endpoint(f"cancel?task_id={task_id}")

        response = await self.client.put(url)
        response.raise_for_status()
        return TaskStatus(**response.json())

    async def _stream_upload_to_disk(self, audio_file: UploadFile, dest_path: str, max_bytes: int | None) -> None:
        """Stream the uploaded file to ``dest_path`` in chunks, enforcing ``max_bytes``."""
        total = 0
        await audio_file.seek(0)
        with open(dest_path, "wb") as dest:
            while chunk := await audio_file.read(_STREAM_CHUNK_BYTES):
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise HTTPException(status_code=413, detail="File is too large")
                dest.write(chunk)

    @staticmethod
    def _build_submit_form(
        progress_id: str,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: ResponseFormat,
        temperature: float | list[float] | None,
        vad_filter: bool,
        diarization: bool,
        diarization_speaker_count: int | None,
        timestamp_granularities: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the multipart form fields for a submit request (no I/O)."""
        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        data: dict[str, Any] = {
            "model": model,
            "progress_id": progress_id,
            "response_format": response_format.value,
            "timestamp_granularities[]": timestamp_granularities,
            "temperature": str(temperature),
            "vad_filter": str(vad_filter),
            "diarization": str(diarization),
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if diarization_speaker_count:
            data["diarization_speaker_count"] = str(diarization_speaker_count)
        for key, value in extra.items():
            data[key] = json.dumps(value) if isinstance(value, list | dict) else str(value)
        return data

    @staticmethod
    def _resolve_mp3_path(input_path: str) -> tuple[str, str | None]:
        """
        Ensure the audio at ``input_path`` is MP3, converting if needed.

        Returns ``(upload_path, converted_path)`` where ``converted_path`` is the
        ffmpeg output that the caller must delete, or ``None`` if no conversion happened.
        """
        # Sniff only the leading bytes instead of loading the whole file.
        with open(input_path, "rb") as fh:
            header = fh.read(_SNIFF_BYTES)

        if is_mp3_format(header):
            return input_path, None

        # convert_to_mp3 is @impure_safe: failures come back as an IOFailure, so inspect the
        # result instead of calling .unwrap() (which would raise UnwrapFailedError, not the
        # AudioConversionError, and bypass the 400 mapping below).
        result = convert_to_mp3(input_path)
        if not is_successful(result):
            error = result.failure()._inner_value
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {error}") from error
        converted_path: str = result.unwrap()._inner_value
        return converted_path, converted_path

    async def _post_submit(self, url: str, data: dict[str, Any], upload_path: str) -> TaskStatus:
        """Stream the MP3 file from disk to the Whisper API and parse the response."""
        with open(upload_path, "rb") as upload_fh:
            files = {"file": ("audio.mp3", upload_fh, "audio/mpeg")}
            response = await self.client.post(url, data=data, files=files)
        response.raise_for_status()
        return TaskStatus(**response.json())

    @future_safe
    async def transcribe_submit_task(
        self,
        audio_file: UploadFile,
        model: str = "large-v2",
        language: str | None = None,
        prompt: str | None = None,
        response_format: ResponseFormat = ResponseFormat.JSON_DIARIZED,
        temperature: float | list[float] | None = None,
        vad_filter: bool = True,
        diarization: bool = True,
        diarization_speaker_count: int | None = None,
        timestamp_granularities: str = "segment",
        max_upload_bytes: int | None = None,
        **kwargs: Any,
    ) -> TaskStatus:
        """
        Submits a new transcription task with additional parameters.

        The upload is streamed to disk, normalized to MP3 if needed, and forwarded to the
        Whisper API without ever holding the whole file in memory, so it is safe for
        multi-hour files under concurrent load.

        Args:
            audio_file: The uploaded audio/video file to transcribe
            model: The Whisper model to use
            language: The language code for transcription
            prompt: Optional prompt for the model
            response_format: Format of the output (enum: ResponseFormat)
            temperature: Temperature value(s) for sampling
            vad_filter: Whether to use voice activity detection
            diarization: Whether to separate speakers
            diarization_speaker_count: Number of speakers to separate
            max_upload_bytes: Hard cap on accepted upload size in bytes
            **kwargs: Additional parameters to pass to the API

        Returns:
            TaskStatus: The status of the created task
        """
        url = self._task_endpoint("submit")

        progress_id = uuid.uuid4().hex
        data = self._build_submit_form(
            progress_id=progress_id,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            vad_filter=vad_filter,
            diarization=diarization,
            diarization_speaker_count=diarization_speaker_count,
            timestamp_granularities=timestamp_granularities,
            extra=kwargs,
        )

        # Stream the upload to a temp file on disk (never fully in memory).
        with tempfile.NamedTemporaryFile(delete=False) as input_temp:
            input_path = input_temp.name

        converted_path: str | None = None
        try:
            await self._stream_upload_to_disk(audio_file, input_path, max_upload_bytes)
            # Sniffing and the ffmpeg subprocess are blocking (up to the 300s ffmpeg timeout);
            # run them in a worker thread so a single conversion does not stall the event loop.
            upload_path, converted_path = await asyncio.to_thread(self._resolve_mp3_path, input_path)
            status = await self._post_submit(url, data, upload_path)
            self.taskId_to_progressId[status.task_id] = progress_id
            return status
        finally:
            Path(input_path).unlink(missing_ok=True)
            if converted_path is not None:
                Path(converted_path).unlink(missing_ok=True)
