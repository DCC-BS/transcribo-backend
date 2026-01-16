from http import HTTPStatus
from typing import Annotated

from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header, HTTPException

from transcribo_backend.container import Container
from transcribo_backend.models.summary import Summary, SummaryRequest
from transcribo_backend.services.summarization_service import SummarizationService

logger = get_logger(__name__)


@inject
def create_router(
    summarization_service: SummarizationService = Provide[Container.summarization_service],
    usage_tracking_service: UsageTrackingService = Provide[Container.usage_tracking_service],
) -> APIRouter:
    """Create the router for the summarize endpoint."""
    logger.info("Creating router for summarize endpoint")
    router = APIRouter()

    @router.post("/summarize")
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
        usage_tracking_service.log_event(
            module="summarize_route",
            func="summarize",
            user_id=x_client_id or "unknown",
            transcript_length=len(request.transcript),
        )

        try:
            summary = await summarization_service.summarize(request.transcript)
        except Exception as e:
            logger.exception("Failed to summarize transcript", exc_info=e)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to generate summary"
            ) from None
        else:
            return summary

    return router
