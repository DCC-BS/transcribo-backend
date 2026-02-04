from http import HTTPStatus
from typing import Annotated

from dcc_backend_common.fastapi_error_handling import ApiErrorCodes, api_error_exception
from dcc_backend_common.logger import get_logger
from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Header
from returns.io import IOSuccess

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
            raise api_error_exception(
                errorId=ApiErrorCodes.INVALID_REQUEST,
                status=HTTPStatus.BAD_REQUEST,
                debugMessage="Transcript cannot be empty",
            )
        if len(request.transcript) > model_context_length * 4:
            raise api_error_exception(
                errorId=ApiErrorCodes.INVALID_REQUEST,
                status=HTTPStatus.BAD_REQUEST,
                debugMessage=f"Transcript is too long. Maximum length is {model_context_length * 4} characters.",
            )
        # Extract X-Client-Id from the request headers
        usage_tracking_service.log_event(
            module="summarize_route",
            func="summarize",
            user_id=x_client_id or "unknown",
            transcript_length=len(request.transcript),
        )

        result = await summarization_service.summarize(request.transcript)

        if isinstance(result, IOSuccess):
            return result.unwrap()._inner_value

        error = result.failure()
        logger.exception("Failed to summarize transcript", exc_info=error)
        raise api_error_exception(
            errorId=ApiErrorCodes.UNEXPECTED_ERROR,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            debugMessage="Failed to generate summary",
        ) from error

    return router
