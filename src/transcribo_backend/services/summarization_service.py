from returns.future import future_safe

from transcribo_backend.agents.summarize_agent import create_summarize_agent
from transcribo_backend.models.summary import Summary
from transcribo_backend.utils.app_config import AppConfig


class SummarizationService:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.agent = create_summarize_agent(app_config)

    @future_safe
    async def summarize(self, transcript: str) -> Summary:
        """
        Summarize a transcript of a meeting.
        """
        result = await self.agent.run(transcript)
        return Summary(summary=result.output)
