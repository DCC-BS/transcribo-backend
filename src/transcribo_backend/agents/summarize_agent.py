from typing import override

from dcc_backend_common.config.app_config import LlmConfig
from dcc_backend_common.llm_agent import BaseAgent
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from transcribo_backend.models.summary import SummaryType

# Instruction prompts for different summary types
VERHANDLUNGSPROTOKOLL_INSTRUCTIONS = """
Du bist ein Experte für Verhandlungsprotokolle.
Du erhältst ein Transkript einer Besprechung und musst es als detailliertes Verhandlungsprotokoll zusammenfassen.
Fokussiere dich auf den Prozess und die Diskussionen, die zu den Entscheidungen geführt haben.
Du musst den Gesprächsverlauf und die wichtigsten Argumente erfassen.
Du musst die Namen der Teilnehmer auflisten.
Du musst alle Beschlüsse und Maßnahmen dokumentieren.
Verwende Markdown zur Formatierung des Protokolls.
Strukturiere das Protokoll mit folgenden Abschnitten:
1. Teilnehmerliste
2. Tagesordnungspunkte
3. Detaillierte Diskussionen zu jedem Punkt
4. Gefasste Beschlüsse
5. Maßnahmen mit Verantwortlichkeiten
Verwende die gleiche Sprache wie im Transkript. Wenn du unsicher bist, verwende Deutsch.
"""

KURZPROTOKOLL_INSTRUCTIONS = """
Du bist ein Experte für Kurzprotokolle.
Du erhältst ein Transkript einer Besprechung und musst es als prägnantes Kurzprotokoll zusammenfassen.
Sei sehr verständlich, überschaubar und zugleich präzise.
Beschränke dich auf die wichtigsten Informationen und lass unwichtige Details weg.
Du musst die wichtigsten Punkte der Besprechung, die Entscheidungen und die Aufgaben kurz zusammenfassen.
Du musst die Namen der Teilnehmer erwähnen.
Verwende Markdown zur Formatierung und halte es kurz und bündig.
Verwende die gleiche Sprache wie im Transkript. Wenn du unsicher bist, verwende Deutsch.
"""

ERGEBNISPROTOKOLL_INSTRUCTIONS = """
Du bist ein Experte für Ergebnisprotokolle.
Du erhältst ein Transkript einer Besprechung und musst es als Ergebnisprotokoll zusammenfassen.
Konzentriere dich ausschließlich auf Ergebnisse und Beschlüsse, nicht auf die Diskussionen.
Du musst alle Ergebnisse und Beschlüsse kurz und präzise auflisten.
Du musst die Maßnahmen seit dem letzten Treffen dokumentieren.
Du musst die geplanten Maßnahmen angeben und vermerken, wer dafür verantwortlich ist.
Du musst die Namen der Teilnehmer auflisten.
Verwende Markdown zur Formatierung mit klaren Abschnitten für Ergebnisse und Maßnahmen.
Verwende die gleiche Sprache wie im Transkript. Wenn du unsicher bist, verwende Deutsch.
"""

DEFAULT_INSTRUCTIONS = """
You are a meeting summary expert.
You are given a transcript of a meeting and you need to summarize it.
You need to summarize the meeting in a way that is easy to understand and use.
You need to include the main points of the meeting, the decisions made, and the action items.
You need to include the names of the participants.
You use markdown to format the summary.
Use the same language as used in the transcript to summarize the meeting.
If you are not sure about the language, use German.
"""


class SummarizeAgent(BaseAgent[SummaryType, str]):
    def __init__(self, config: LlmConfig):
        super().__init__(config, deps_type=SummaryType, output_type=str, enable_thinking=False)

    @override
    def create_agent(self, model: Model) -> Agent[SummaryType, str]:
        agent = Agent(model=model, deps_type=self.deps_type, output_type=self.output_type)

        @agent.instructions
        def get_instructions(ctx: RunContext[SummaryType]) -> str:
            summary_type = ctx.deps

            if summary_type == SummaryType.VERHANDLUNGSPROTOKOLL:
                return VERHANDLUNGSPROTOKOLL_INSTRUCTIONS
            elif summary_type == SummaryType.KURZPROTOKOLL:
                return KURZPROTOKOLL_INSTRUCTIONS
            elif summary_type == SummaryType.ERGEBNISPROTOKOLL:
                return ERGEBNISPROTOKOLL_INSTRUCTIONS
            else:
                return DEFAULT_INSTRUCTIONS

        return agent
