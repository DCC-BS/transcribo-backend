from http import HTTPStatus
from typing import Annotated

from backend_common.fastapi_error_handling import api_error_exception
from fastapi import APIRouter, Header

from transcribo_backend.models.error_codes import TranscriboErrorCodes
from transcribo_backend.models.summary import Summary, SummaryRequest
from transcribo_backend.services import summary_service
from transcribo_backend.utils.logger import get_logger
from transcribo_backend.utils.usage_tracking import get_pseudonymized_user_id

logger = get_logger("summarization_router")

router = APIRouter(prefix="", tags=["summarization"])


@router.post("/summarize")
async def summarize(request: SummaryRequest, x_client_id: Annotated[str | None, Header()] = None) -> Summary:
    """
    Endpoint to summarize a text.
    """
    model_context_length = 32_000

    if not request.transcript or not request.transcript.strip():
        raise api_error_exception(
            errorId=TranscriboErrorCodes.TRANSCRIPT_EMPTY,
            status=HTTPStatus.BAD_REQUEST,
            debugMessage="Transcript is empty",
        )
    if len(request.transcript) > model_context_length * 4:
        raise api_error_exception(
            errorId=TranscriboErrorCodes.TRANSCRIPT_TOO_LONG,
            status=HTTPStatus.BAD_REQUEST,
            debugMessage=f"Transcript is too long. Maximum length is {model_context_length * 4} characters.",
        )
    # Extract X-Client-Id from the request headers
    pseudonym_id = get_pseudonymized_user_id(x_client_id or "unknown")
    logger.info(
        "app_event",
        extra={"pseudonym_id": pseudonym_id, "event": "summarize", "transcript_length": len(request.transcript)},
    )

    try:
        summary = await summary_service.summarize(request.transcript)
    except Exception as e:
        logger.exception("Failed to summarize transcript", exc_info=e)
        raise api_error_exception(
            errorId=TranscriboErrorCodes.SUMMARY_GENERATION_FAILED,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            debugMessage=f"Failed to generate summary: {e}",
        ) from None
    else:
        return summary
