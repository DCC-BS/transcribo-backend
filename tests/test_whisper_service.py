"""Unit tests for WhisperService streaming submit + result handling.

These tests mock the HTTP client and never touch a real Whisper backend. They lock in
the memory-safety fixes:
  * the upload is streamed to disk and forwarded as a *file handle* (never the whole
    file as an in-memory ``bytes`` object),
  * already-MP3 uploads skip ffmpeg conversion,
  * non-MP3 uploads are converted and the temp files are cleaned up,
  * oversized uploads are rejected with HTTP 413 before any request is sent,
  * ``transcribe_get_task_result`` returns the *normalized* transcription (regression
    for the old double-parse bug that discarded the mutations).
"""

import os
import tempfile
from io import BytesIO
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from returns.io import IOFailure, IOSuccess

from transcribo_backend.services.whisper_service import WhisperService
from transcribo_backend.utils.app_config import AppConfig

# An MP3 file detected via its ID3 tag (see audio_converter.is_mp3_format).
MP3_BYTES = b"ID3" + b"\x00" * 4096
# A non-MP3 payload (WAV-ish header) that should trigger conversion.
NON_MP3_BYTES = b"RIFF" + b"\x00" * 4096


def _make_service(max_upload_bytes: int = 50 * 1024 * 1024) -> WhisperService:
    cfg = MagicMock(spec=AppConfig)
    cfg.whisper_url = "http://whisper.test"
    cfg.llm_api_key = "test-key"
    cfg.max_upload_bytes = max_upload_bytes
    return WhisperService(cfg)


def _make_upload(data: bytes, filename: str = "audio.bin") -> UploadFile:
    return UploadFile(file=BytesIO(data), size=len(data), filename=filename)


def _capturing_post(captured: dict):
    """Return an AsyncMock side effect that records how the file was sent."""

    async def _post(url, data=None, files=None):
        assert files is not None
        fh = files["file"][1]
        captured["url"] = url
        captured["data"] = data
        captured["filename"] = files["file"][0]
        # The fix forwards a file handle, not raw bytes. Record both facts.
        captured["is_file_handle"] = hasattr(fh, "read") and not isinstance(fh, bytes | bytearray)
        captured["body"] = fh.read()
        resp = MagicMock()
        resp.json.return_value = {"task_id": "task-1", "status": "in_progress"}
        resp.raise_for_status = MagicMock()
        return resp

    return AsyncMock(side_effect=_post)


@pytest.mark.anyio
async def test_submit_streams_mp3_as_file_handle_without_conversion():
    svc = _make_service()
    captured: dict = {}
    svc.client.post = _capturing_post(captured)

    with patch("transcribo_backend.services.whisper_service.convert_to_mp3") as convert:
        result = await svc.transcribe_submit_task(
            _make_upload(MP3_BYTES, "audio.mp3"), max_upload_bytes=svc.app_config.max_upload_bytes
        )

    assert isinstance(result, IOSuccess), result
    # Already MP3 -> no ffmpeg conversion.
    convert.assert_not_called()
    # Forwarded as a streamed file handle, not an in-memory bytes object.
    assert captured["is_file_handle"] is True
    assert captured["body"] == MP3_BYTES
    # Task id mapped to a progress id for later status lookups.
    status = result.unwrap()._inner_value
    assert status.task_id == "task-1"
    assert svc.taskId_to_progressId[status.task_id]

    await svc.aclose()


@pytest.mark.anyio
async def test_submit_converts_non_mp3_and_cleans_up_temp_file():
    svc = _make_service()
    captured: dict = {}
    svc.client.post = _capturing_post(captured)

    # Stand-in for ffmpeg output.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(b"CONVERTED_MP3")
        converted_path = f.name

    with patch("transcribo_backend.services.whisper_service.convert_to_mp3") as convert:
        convert.return_value = IOSuccess(converted_path)
        result = await svc.transcribe_submit_task(
            _make_upload(NON_MP3_BYTES, "audio.wav"), max_upload_bytes=svc.app_config.max_upload_bytes
        )

    assert isinstance(result, IOSuccess), result
    # Conversion was invoked with a path on disk (streamed input), not bytes.
    convert.assert_called_once()
    (input_path,) = convert.call_args.args
    assert isinstance(input_path, str)
    # The converted file was the one uploaded.
    assert captured["body"] == b"CONVERTED_MP3"
    assert captured["is_file_handle"] is True
    # Temp files are cleaned up afterwards (no disk leak).
    assert not os.path.exists(converted_path)
    assert not os.path.exists(input_path)

    await svc.aclose()


@pytest.mark.anyio
async def test_submit_rejects_oversized_upload_before_sending():
    svc = _make_service(max_upload_bytes=8)
    svc.client.post = cast(Any, AsyncMock())

    result = await svc.transcribe_submit_task(
        _make_upload(b"x" * 4096, "audio.mp3"), max_upload_bytes=svc.app_config.max_upload_bytes
    )

    assert isinstance(result, IOFailure), result
    error = result.failure()._inner_value
    assert isinstance(error, HTTPException)
    assert error.status_code == 413
    # No request was sent for an oversized upload.
    svc.client.post.assert_not_called()

    await svc.aclose()


@pytest.mark.anyio
async def test_get_task_result_returns_normalized_transcription():
    """Regression: the old code built the response twice and discarded normalization."""
    svc = _make_service()
    svc.taskId_to_progressId["task-1"] = "progress-1"

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "  Straße  ", "speaker": "bob"},
            {"start": 1.0, "end": 2.0, "text": "hi", "speaker": None},
        ]
    }
    svc.client.get = cast(Any, AsyncMock(return_value=resp))

    result = await svc.transcribe_get_task_result("task-1")

    assert isinstance(result, IOSuccess), result
    transcription = result.unwrap()._inner_value
    # text stripped + ß -> ss; speaker capitalized.
    assert transcription.segments[0].text == "Strasse"
    assert transcription.segments[0].speaker == "Bob"
    # Missing speaker defaults to "Unknown".
    assert transcription.segments[1].speaker == "Unknown"
    # Completed result is evicted from the progress cache.
    assert "task-1" not in svc.taskId_to_progressId

    await svc.aclose()
