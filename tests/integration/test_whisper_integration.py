"""Integration tests against a live Whisper backend.

These are skipped unless ``WHISPER_BACKEND_URL`` is set. Run them with::

    WHISPER_BACKEND_URL=https://whisper.example.com \
    WHISPER_API_KEY=... \
    make test-integration

They exercise the real streaming submit -> status -> result roundtrip with a generated
audio file, verifying nothing in the memory-safe path broke the actual contract.
ffmpeg must be available locally (the WAV is converted to MP3 before upload).
"""

import asyncio
import os
import struct
import wave
from io import BytesIO
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, UploadFile
from returns.io import IOFailure, IOSuccess

from transcribo_backend.models.task_status import TaskStatusEnum
from transcribo_backend.services.whisper_service import WhisperService
from transcribo_backend.utils.app_config import AppConfig

WHISPER_URL = os.getenv("WHISPER_BACKEND_URL")
API_KEY = os.getenv("WHISPER_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.skipif(not WHISPER_URL, reason="set WHISPER_BACKEND_URL to run integration tests"),
]


def _make_service(max_upload_bytes: int = 500 * 1024 * 1024) -> WhisperService:
    cfg = SimpleNamespace(
        whisper_url=WHISPER_URL,
        llm_api_key=API_KEY,
        max_upload_bytes=max_upload_bytes,
    )
    return WhisperService(cast(AppConfig, cfg))


def _make_wav(seconds: int = 2, rate: int = 16000) -> bytes:
    """Generate a small mono silent WAV in memory (no ffmpeg needed to create it)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        sample_count = rate * seconds
        w.writeframes(struct.pack(f"<{sample_count}h", *([0] * sample_count)))
    return buf.getvalue()


def _upload(data: bytes, filename: str = "test.wav") -> UploadFile:
    return UploadFile(file=BytesIO(data), size=len(data), filename=filename)


async def test_submit_status_result_roundtrip():
    svc = _make_service()
    try:
        submit = await svc.transcribe_submit_task(
            _upload(_make_wav(2)), max_upload_bytes=svc.app_config.max_upload_bytes
        )
        assert isinstance(submit, IOSuccess), submit
        task = submit.unwrap()._inner_value
        assert task.task_id

        status = None
        deadline = asyncio.get_event_loop().time() + 300
        while asyncio.get_event_loop().time() < deadline:
            status_result = await svc.transcribe_get_task_status(task.task_id)
            assert isinstance(status_result, IOSuccess), status_result
            status = status_result.unwrap()._inner_value
            if status.status in (TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED):
                break
            await asyncio.sleep(3)

        assert status is not None and status.status == TaskStatusEnum.COMPLETED

        result = await svc.transcribe_get_task_result(task.task_id)
        assert isinstance(result, IOSuccess), result
        transcription = result.unwrap()._inner_value
        assert hasattr(transcription, "segments")
    finally:
        await svc.aclose()


async def test_oversized_upload_rejected_without_contacting_backend():
    svc = _make_service(max_upload_bytes=8)
    try:
        result = await svc.transcribe_submit_task(
            _upload(b"x" * 4096), max_upload_bytes=svc.app_config.max_upload_bytes
        )
        assert isinstance(result, IOFailure), result
        error = result.failure()._inner_value
        assert isinstance(error, HTTPException)
        assert error.status_code == 413
    finally:
        await svc.aclose()
