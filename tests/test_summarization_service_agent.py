from unittest.mock import AsyncMock, MagicMock

import pytest

from transcribo_backend.models.summary import SummaryType
from transcribo_backend.services.summarization_service import SummarizationService
from transcribo_backend.utils.app_config import AppConfig


@pytest.mark.anyio
async def test_summarize_calls_agent():
    # Mock AppConfig
    app_config = MagicMock(spec=AppConfig)

    # Mock the Agent and its run method
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="This is a summary.")

    # Create service with the mock agent
    service = SummarizationService(app_config, mock_agent)

    transcript = "Some meeting transcript."
    # Awaiting a FutureResult returns a Result (e.g., IOResult)
    # Result.unwrap() should give the value.
    result_io = await service.summarize(transcript)
    result = result_io.unwrap()._inner_value

    # Verify run was called with default ERGEBNISPROTOKOLL
    mock_agent.run.assert_called_once_with(transcript, deps=SummaryType.ERGEBNISPROTOKOLL)

    # Verify result - result should be a Summary object
    assert result.summary == "This is a summary."
