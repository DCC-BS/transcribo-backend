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
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection
from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult
from transcribo_backend.models.transcription_response import Segment, TranscriptionResponse, Word
from transcribo_backend.routes import transcribe_route


def _build_client(whisper_service, usage_service, postprocessing_service=None) -> TestClient:
    app = FastAPI()
    inject_api_error_handler(app)
    app.include_router(
        transcribe_route.create_router(
            whisper_service=whisper_service,
            usage_tracking_service=usage_service,
            transcript_postprocessing_service=postprocessing_service or MagicMock(),
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


def test_task_result_preserves_fragments_and_never_leaks_word_level():
    """The client contract is fragment level: segment count and start/end
    timestamps reach the client exactly as the recognizer produced them,
    independent of post-processing; cleanup output is attached as separate
    fields; internal word-level data never appears in the response."""
    whisper_service, usage_service = _make_services()
    transcription = TranscriptionResponse(
        segments=[
            Segment(
                start=0.0,
                end=1.5,
                text="Dropshipping ist toll.",
                speaker="A",
                words=[Word(start=0.0, end=0.7, word=" Dropshipping", probability=0.2)],
            ),
            Segment(start=1.5, end=3.0, text="Genau.", speaker="B", words=None),
        ]
    )
    whisper_service.transcribe_get_task_result = AsyncMock(return_value=IOSuccess(transcription))

    postprocessing_service = MagicMock()
    postprocessing_service.post_process = AsyncMock(
        return_value=IOSuccess(
            TranscriptPostProcessingResult(
                title="Gespräch über Dropshipping",
                speaker_assignments=[],
                corrections=[
                    TranscriptCorrection(original="Jobshipping", corrected="Dropshipping", confidence=0.9),
                ],
                keywords=[],
            )
        )
    )

    client = _build_client(whisper_service, usage_service, postprocessing_service)
    resp = client.post("/task/task-1/result", json={})

    assert resp.status_code == 200
    body = resp.json()

    assert len(body["segments"]) == 2
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(0.0, 1.5), (1.5, 3.0)]
    # Internal word-level data never reaches the client — not even as a key.
    assert all("words" not in s for s in body["segments"])
    # Cleanup output rides along in separate fields.
    assert body["applied_corrections"][0]["original"] == "Jobshipping"
    assert body["title"] == "Gespräch über Dropshipping"


def test_task_result_survives_postprocessing_failure():
    """A failed cleanup must not block the transcription result."""
    whisper_service, usage_service = _make_services()
    transcription = TranscriptionResponse(segments=[Segment(start=0.0, end=1.0, text="Hallo.", speaker="A")])
    whisper_service.transcribe_get_task_result = AsyncMock(return_value=IOSuccess(transcription))

    postprocessing_service = MagicMock()
    failing = MagicMock()
    failing.failure.return_value._inner_value = RuntimeError("llm down")
    postprocessing_service.post_process = AsyncMock(return_value=failing)

    client = _build_client(whisper_service, usage_service, postprocessing_service)
    resp = client.post("/task/task-1/result", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["segments"]) == 1
    assert body["speaker_assignments"] is None
    assert body["applied_corrections"] is None
    assert body["title"] is None


def test_happy_path_logs_usage():
    whisper_service, usage_service = _make_services()
    client = _build_client(whisper_service, usage_service)

    response = client.post("/transcribe", files={"audio_file": ("audio.mp3", b"ID3" + b"\x00" * 100, "audio/mpeg")})

    assert response.status_code == 200
    usage_service.log_event.assert_called_once()
