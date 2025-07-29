from openai import AsyncOpenAI

from transcribo_backend.config import settings
from transcribo_backend.models.summary import Summary

system_prompt = """
You are a meeting summary expert.
You are given a transcript of a meeting and you need to summarize it.
You need to summarize the meeting in a way that is easy to understand and use.
You need to include the main points of the meeting, the decisions made, and the action items.
You need to include the names of the participants.
You use markdown to format the summary.
Use the same language as used in the transcript to summarize the meeting.
If you are not sure about the language, use German.
"""


async def summarize(transcript: str) -> Summary:
    """
    Summarize a transcript of a meeting.
    """
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.llm_api)
    models = await client.models.list()
    model = models.data[0].id
    transcript = transcript + "\nothink"
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}],
    )
    return Summary(summary=response.choices[0].message.content)
