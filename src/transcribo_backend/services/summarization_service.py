from returns.future import future_safe

from transcribo_backend.agents.summarize_agent import SummarizeAgent
from transcribo_backend.models.summary import Summary, SummaryType
from transcribo_backend.utils.app_config import AppConfig


class SummarizationService:
    def __init__(self, app_config: AppConfig, summarize_agent: SummarizeAgent):
        self.app_config = app_config
        self.agent = summarize_agent

    @future_safe
    async def summarize(self, transcript: str, summary_type: SummaryType | None = None) -> Summary:
        """
        Summarize a transcript of a meeting.
        """
        # Use ERGEBNISPROTOKOLL as default if no type is specified
        if summary_type is None:
            summary_type = SummaryType.ERGEBNISPROTOKOLL

        result = await self.agent.run(transcript, deps=summary_type)
        return Summary(summary=result)
