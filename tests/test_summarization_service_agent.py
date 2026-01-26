from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transcribo_backend.services.summarization_service import SummarizationService
from transcribo_backend.utils.app_config import AppConfig


@pytest.mark.anyio
async def test_summarize_calls_agent():
    # Mock AppConfig
    app_config = MagicMock(spec=AppConfig)

    # Mock the Agent and its run method
    mock_agent = MagicMock()
    mock_run_result = MagicMock()
    mock_run_result.output = "This is a summary."
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    # Patch create_summarize_agent to return our mock agent
    with patch(
        "transcribo_backend.services.summarization_service.create_summarize_agent", return_value=mock_agent
    ) as mock_create:
        service = SummarizationService(app_config)

        # Verify agent creation was called
        mock_create.assert_called_once_with(app_config)

        transcript = "Some meeting transcript."
        # Awaiting a FutureResult returns a Result (e.g., IOResult)
        # Result.unwrap() should give the value.
        result = (await service.summarize(transcript)).unwrap()

        # If result is an IO container, we get its value.
        # In 'returns', IO is often used to wrap values in IOResult.
        if hasattr(result, "_inner_value"):
            result = result._inner_value

        # Verify run was called
        mock_agent.run.assert_called_once_with(transcript)

        # Verify result
        assert result.summary == "This is a summary."
