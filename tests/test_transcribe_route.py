"""Unit tests for the /transcribe route.

The router is built with injected mocks (no DI container, no Whisper backend) and driven
through a FastAPI TestClient. These cover the request-side fixes: oversized uploads are
rejected early with 413, unsupported types with 415, and a valid upload is forwarded to
the service as an UploadFile (not pre-read bytes).
"""

from unittest.mock import AsyncMock, MagicMock

from dcc_backend_common.fastapi_error_handling import inject_api_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from returns.io import IOSuccess
from starlette.datastructures import UploadFile

from transcribo_backend.models.task_status import TaskStatus
from transcribo_backend.routes import transcribe_route


def _build_client(whisper_service, usage_service) -> TestClient:
    app = FastAPI()
    inject_api_error_handler(app)
    app.include_router(
        transcribe_route.create_router(
            whisper_service=whisper_service,
            usage_tracking_service=usage_service,
        )
    )
    return TestClient(app)


def _make_services(max_upload_bytes: int = 50 * 1024 * 1024):
    whisper_service = MagicMock()
    whisper_service.app_config.max_upload_bytes = max_upload_bytes
    whisper_service.transcribe_submit_task = AsyncMock(return_value=IOSuccess(TaskStatus(task_id="task-1")))
    usage_service = MagicMock()
    return whisper_service, usage_service


def test_rejects_unsupported_content_type():
    whisper_service, usage_service = _make_services()
    client = _build_client(whisper_service, usage_service)

    resp = client.post("/transcribe", files={"audio_file": ("note.txt", b"hello", "text/plain")})

    assert resp.status_code == 415
    whisper_service.transcribe_submit_task.assert_not_called()


def test_rejects_oversized_upload():
    whisper_service, usage_service = _make_services(max_upload_bytes=8)
    client = _build_client(whisper_service, usage_service)

    resp = client.post("/transcribe", files={"audio_file": ("audio.mp3", b"x" * 4096, "audio/mpeg")})

    assert resp.status_code == 413
    # Rejected before reaching the service or the usage tracker.
    whisper_service.transcribe_submit_task.assert_not_called()
    usage_service.log_event.assert_not_called()


def test_valid_upload_is_forwarded_to_service():
    whisper_service, usage_service = _make_services()
    client = _build_client(whisper_service, usage_service)

    resp = client.post(
        "/transcribe",
        files={"audio_file": ("audio.mp3", b"ID3" + b"\x00" * 100, "audio/mpeg")},
        data={"num_speakers": "2", "language": "de"},
    )

    assert resp.status_code == 200
    assert resp.json()["task_id"] == "task-1"

    whisper_service.transcribe_submit_task.assert_awaited_once()
    call = whisper_service.transcribe_submit_task.await_args
    # The route forwards an UploadFile (streamed), not pre-read bytes.
    assert isinstance(call.args[0], UploadFile)
    assert call.kwargs["max_upload_bytes"] == whisper_service.app_config.max_upload_bytes
    assert call.kwargs["diarization_speaker_count"] == 2
    assert call.kwargs["language"] == "de"


def test_happy_path_logs_usage():
    whisper_service, usage_service = _make_services()
    client = _build_client(whisper_service, usage_service)

    client.post("/transcribe", files={"audio_file": ("audio.mp3", b"ID3" + b"\x00" * 100, "audio/mpeg")})

    usage_service.log_event.assert_called_once()
