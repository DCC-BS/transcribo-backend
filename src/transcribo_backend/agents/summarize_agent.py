from openai import AsyncOpenAI
from pydantic_ai import Agent, TextOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from transcribo_backend.utils.app_config import AppConfig


def transform_to_swissgerman_style(text: str) -> str:
    return text.replace("ß", "ss")


def create_summarize_agent(app_config: AppConfig) -> Agent:
    client = AsyncOpenAI(max_retries=3, base_url=app_config.llm_base_url, api_key=app_config.api_key)
    model = OpenAIChatModel(model_name=app_config.llm_model, provider=OpenAIProvider(openai_client=client))
    summarize_agent: Agent = Agent(
        model=model,
        output_type=TextOutput(transform_to_swissgerman_style),
    )

    @summarize_agent.instructions
    def get_instructions() -> str:
        return """
You are a meeting summary expert.
You are given a transcript of a meeting and you need to summarize it.
You need to summarize the meeting in a way that is easy to understand and use.
You need to include the main points of the meeting, the decisions made, and the action items.
You need to include the names of the participants.
You use markdown to format the summary.
Use the same language as used in the transcript to summarize the meeting.
If you are not sure about the language, use German.
"""

    return summarize_agent
