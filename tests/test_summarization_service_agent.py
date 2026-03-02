from unittest.mock import AsyncMock, MagicMock

import pytest

from transcribo_backend.models.language import Language
from transcribo_backend.models.summary import SummaryDeps, SummaryType
from transcribo_backend.services.summarization_service import SummarizationService
from transcribo_backend.utils.app_config import AppConfig


@pytest.mark.anyio
async def test_summarize_calls_agent():
    app_config = MagicMock(spec=AppConfig)

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="This is a summary.")

    service = SummarizationService(app_config, mock_agent)

    transcript = "Some meeting transcript."
    result_io = await service.summarize(transcript)
    result = result_io.unwrap()._inner_value

    expected_deps = SummaryDeps(summary_type=SummaryType.ERGEBNISPROTOKOLL, language=None)
    mock_agent.run.assert_called_once_with(transcript, deps=expected_deps)

    assert result.summary == "This is a summary."


@pytest.mark.anyio
async def test_summarize_with_language():
    app_config = MagicMock(spec=AppConfig)

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="This is an English summary.")

    service = SummarizationService(app_config, mock_agent)

    transcript = "Some meeting transcript."
    result_io = await service.summarize(transcript, language=Language.EN)
    result = result_io.unwrap()._inner_value

    expected_deps = SummaryDeps(summary_type=SummaryType.ERGEBNISPROTOKOLL, language=Language.EN)
    mock_agent.run.assert_called_once_with(transcript, deps=expected_deps)

    assert result.summary == "This is an English summary."


@pytest.mark.anyio
async def test_summarize_with_type_and_language():
    app_config = MagicMock(spec=AppConfig)

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="This is a Kurzprotokoll in French.")

    service = SummarizationService(app_config, mock_agent)

    transcript = "Some meeting transcript."
    result_io = await service.summarize(transcript, summary_type=SummaryType.KURZPROTOKOLL, language=Language.FR)
    result = result_io.unwrap()._inner_value

    expected_deps = SummaryDeps(summary_type=SummaryType.KURZPROTOKOLL, language=Language.FR)
    mock_agent.run.assert_called_once_with(transcript, deps=expected_deps)

    assert result.summary == "This is a Kurzprotokoll in French."
