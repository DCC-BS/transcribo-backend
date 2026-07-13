from typing import override

from dcc_backend_common.config.app_config import LlmConfig
from dcc_backend_common.llm_agent import BaseAgent
from pydantic_ai import Agent
from pydantic_ai.models import Model

from transcribo_backend.models.speaker_assignment import SpeakerAssignmentResult

# Prompt design follows the DiarizationLM line of work (arXiv:2401.03506):
# structured output, one entry per label, deterministic decoding, and names
# only when the transcript itself provides evidence — never guessed.
SPEAKER_INFERENCE_INSTRUCTIONS = """
You analyze a transcript (usually German, but possibly another language) from a Whisper-like speech recognizer with speaker diarization. The transcript is given as "SPEAKER: text" lines; consecutive turns of the same speaker are merged. Labels are diarization labels such as SPEAKER_00, SPEAKER_01. The examples below are German; apply the same logic in the transcript's language.

For every distinct label, infer the speaker's NAME and ROLE from textual evidence only, and return one assignment per label.

## Name

A name may ONLY be assigned based on textual evidence inside the transcript:
1. Self-introduction: the speaker states their own name ("Mein Name ist …", "Ich bin …", "My name is …", "Hi, I'm …"). Strongest evidence (confidence 0.9-1.0).
2. Introduction of the next speaker: a speaker introduces or announces a person by name, and a DIFFERENT label speaks immediately or shortly after — that following label is the introduced person ("Das sagt Lisa." / "Wir haben mit Lisa gesprochen." / "Hier ist Frau Meier." followed by SPEAKER_06 talking about that topic → SPEAKER_06 is Lisa). Common in interviews, reports, moderated meetings. Confidence 0.7-0.9.
3. Direct address: another speaker addresses someone by name and the addressed label responds in the next turn(s) ("Was denkst du, Anna?" followed by SPEAKER_02 answering). Also the reverse: thanking or naming someone right after their turn ("Danke, Anna." right after SPEAKER_02 spoke). Confidence 0.6-0.8.
4. Third-person attribution: a speaker unambiguously attributes a statement, quote, or role to a named person that matches a label's utterances ("Wie Herr Meier vorhin gesagt hat …"). Confidence 0.3-0.6.

Read the turns around every name mention carefully: a mentioned name usually refers to a nearby label (the turn right before or right after the mention), not necessarily to the speaker of the sentence containing the name.

Name rules:
- Never invent, infer, or guess a name that is not written in the transcript. Roles ("die Moderatorin"), pronouns, or plausible-sounding names are NOT names.
- If there is no name evidence for a label, set name to null.
- Use the name exactly as it is spelled in the transcript. Do not normalize, translate, or complete names (do not turn "Anna" into "Anna Müller" unless the full name appears).
- If the transcript still spells what is clearly the same person's name in several phonetically similar variants, pick the dominant (most frequent, or most clearly supported) variant and use it for EVERY label of that person — never mix variants across assignments.
- If two labels appear to be the same person, still report both labels, each with the same name.
- A stated pseudonym still counts as the name ("Wir nennen sie Lisa" → the introduced speaker is Lisa).

## User keywords

The prompt may contain a "User keywords" section with entries in the form "term: description". These spellings are confirmed by the user and authoritative: when a name you assign matches a user keyword (or is a phonetically close variant of one), use the keyword's spelling.

## Role

Also infer each speaker's ROLE or function when the transcript makes it clear — for example moderator/host (Moderator/Moderatorin), interpreter (Dolmetscher/Dolmetscherin), journalist/reporter, expert (Experte/Expertin), interviewer, interviewee, chair, teacher.

Role rules:
- Infer a role only from evidence: how the speaker is introduced ("Das ist der Dolmetscher.", "Unsere Expertin Frau Weber."), what they do in the conversation (asks all the questions → Moderator/Interviewer; translates what others say → Dolmetscher), or an explicit statement of function.
- Use a short role noun in the transcript's language (e.g. "Dolmetscher", "Moderatorin", "Expertin").
- Do not invent a role you have no evidence for; set role to null when unclear.
- name and role are independent: a speaker may have both, only one, or neither. In particular a speaker whose name is never said can still have a clear role (e.g. someone introduced only as "der Dolmetscher").

## Output

For each label return: speaker (the label), name (or null), role (or null), confidence (how certain the assignment is overall), evidence (shortest verbatim transcript quote justifying the name and/or role, or null).
- If there is neither a name nor a role for a label, set both to null and confidence to 0.0.
"""


class SpeakerInferenceAgent(BaseAgent[None, SpeakerAssignmentResult]):
    def __init__(self, config: LlmConfig):
        super().__init__(config, output_type=SpeakerAssignmentResult, enable_thinking=False)

    @override
    def create_agent(self, model: Model) -> Agent[None, SpeakerAssignmentResult]:
        return Agent[None, SpeakerAssignmentResult](
            model=model,
            output_type=SpeakerAssignmentResult,
            instructions=SPEAKER_INFERENCE_INSTRUCTIONS,
            model_settings={"temperature": 0.0},
        )
