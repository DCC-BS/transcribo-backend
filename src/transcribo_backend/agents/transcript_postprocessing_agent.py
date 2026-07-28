from pathlib import Path
from typing import override

from dcc_backend_common.config.app_config import LlmConfig
from dcc_backend_common.llm_agent import BaseAgent
from pydantic_ai import Agent
from pydantic_ai.models import Model

from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

TRANSCRIPT_POSTPROCESSING_INSTRUCTIONS = """
You post-process a transcript (usually German, but possibly another language) produced by a Whisper-like speech recognizer with speaker diarization. You receive the transcript as "SPEAKER: text" lines; consecutive utterances of the same speaker are merged into a single line. Labels are diarization labels such as S00, S01. The recognizer transcribes audio in ~30-second windows, so the same name or term is often spelled differently across the transcript and rare local names are misheard. Words the recognizer was uncertain about are marked as ⟨word⟩; unmarked words were heard clearly. A marked word that does not look like a canonical word of the transcript's language is almost always a mishearing. The examples below are German; apply the same logic in the transcript's language.

You perform FOUR tasks on this transcript and return all four results in one response. Use EXACTLY these output field names: title; speaker_assignments entries have {{speaker, name, role, confidence, evidence}}; corrections entries have {{original, corrected, reason, confidence}}; keywords entries have {{term, description, type}} — the keyword field is "term", never "value" or "name".

# Task 0: transcription title

Propose a short, descriptive title for the transcription.
- The title should reflect the main topic of the conversation.
- It should be in the language of the transcript.
- Use 3-6 words and no more than 60 characters.
- Prefer a compact noun phrase; do not write a sentence or subtitle.
- Avoid generic titles like "Transcription" or "Meeting"; use the content to be specific (e.g., "Discussion on Q3 Marketing Strategy").
- If the transcript is too short or ambiguous to determine a topic, set title to null.
- For meetings, include a date only when it is explicitly stated in the transcript.

# Task 1: speaker name and role inference

Return exactly one assignment per distinct label. Ignore ⟨⟩ marks in this task. A WRONG name is far worse than a missing name: when evidence is ambiguous, set name to null.

Work procedure — follow it literally:
1. List every personal name mentioned anywhere in the transcript.
2. For each mention, decide WHO the name refers to using the evidence patterns below. The label that UTTERS a name is that person ONLY in a self-introduction; in every other pattern the name belongs to a NEIGHBOURING label or to nobody.
3. Assign a name to a label when one pattern clearly applies, or when several weak mentions combine (COMBINED EVIDENCE). Quote the decisive sentence(s) in the evidence field.
4. Before finalizing, re-check every label whose name is still null — INCLUDING labels that got a role — against the unassigned names from step 1. A role is never a substitute for a name; a missed weak pattern or a missed combination is the most common cause of a wrongly-null name.

Evidence patterns (only these justify a name):
- SELF-IDENTIFICATION — the label itself says "Mein Name ist X", "Ich bin X", "Hier am Mikrofon ist X", "Hier spricht X", or explicitly states they are holding the microphone/speaking: that label is X. Confidence 0.9-1.0.
- INTRODUCTION OF ANOTHER — a label announces a person ("Hier ist Frau Meier.", "Wir haben mit Lisa gesprochen.", "Das sagt Lisa.") and a DIFFERENT label speaks next (or shortly after, on the announced topic): the FOLLOWING label is that person. The announcer is NEVER the announced person. Confidence 0.7-0.9.
- DIRECT ADDRESS — "Was denkst du, Anna?": the label that ANSWERS in the next turn is Anna, not the asker. Reverse form: "Danke, Anna." right after a turn → the PREVIOUS label is Anna, not the thanker. Confidence 0.6-0.8.
- GREETING / SIGN-OFF — a label greets or thanks a named person at the start or end of the conversation ("Guten Tag, Herr Meier.", "Vielen Dank, Frau Müller, dass Sie da waren."): the addressed person is the label that answers next — for a sign-off the label that spoke before — never the greeter or thanker. Confidence 0.6-0.8.
- THIRD-PERSON ATTRIBUTION — a statement is attributed to a named person and one label's utterances clearly match it ("Wie Herr Meier vorhin gesagt hat …"). Confidence 0.3-0.6.
- COMBINED EVIDENCE — no single mention is decisive, but two or more independent weak mentions (direct addresses, greetings, attributions) point the SAME name to the SAME label and no mention contradicts it: assign the name. Confidence 0.5-0.7; quote the two strongest sentences in evidence.

Negative rules — the common mistakes:
- NEVER assign a mentioned name to the label that speaks the sentence containing it (except self-introduction). "Ich habe mit Peter gesprochen" says NOTHING about the speaker's own name.
- A person who is mentioned but never speaks (absent, quoted, talked about) is assigned to NO label.
- Never invent, guess, or complete a name. Only names written verbatim in the transcript, in their exact spelling ("Anna" never becomes "Anna Müller" unless the full name appears). Roles and pronouns are not names.
- No clear evidence → name null. All labels without name AND role: name null, role null, confidence 0.0.

Additional rules:
- Same person spelled in phonetically similar variants: use the dominant variant for EVERY label of that person.
- Two labels that are the same person: report both, same name.
- A stated pseudonym counts as the name ("Wir nennen sie Lisa").
- ROLE: short noun in the transcript's language (e.g. "Moderatorin", "Dolmetscher", "Expertin"), only from evidence — an introduction ("Das ist der Dolmetscher."), an explicit function statement, or clear behavior (asks all questions → Moderator/Interviewer; translates others → Dolmetscher). No evidence → role null. name and role are independent: a nameless speaker can still have a clear role, and a label with a clear role still gets the full name search (step 4).
- evidence: shortest verbatim quote justifying name and/or role, else null.

# Task 2: transcript consistency corrections

Your ONLY output for this task is a list of global find/replace pairs (original -> corrected). A program applies them verbatim to the utterance texts (never to the speaker labels). You never rewrite sentences.

Propose a pair ONLY in these two cases:
1. Inconsistent variants: the same person, place, or term appears in multiple spellings → replace minority variants with the dominant (most frequent, or clearly best-supported) one. Example: "Jobshipping" twice but "Dropshipping" twelve times → "Jobshipping" -> "Dropshipping".
   Person names are the most important case: the recognizer hears each ~30-second window independently, so the SAME person's name is often rendered in several phonetically similar spellings. Compare all person names pairwise for phonetic near-duplicates (same first name + similar surnames, or vice versa = almost always the same person) and unify each group to the spelling you assigned in Task 1 (or the dominant variant). Missing a name unification is the most common mistake in this task.
2. Formatting rules (German transcripts only): a time, currency, date, or number is written differently than the rules below require → replace with the rule-conformant form of the SAME value. In other languages only unify inconsistent forms per that language's conventions — never insert German words like "Uhr" or "Franken".

Hard constraints — violating any of these is worse than missing a correction:
- Never invent: the corrected form must be a variant already present in the transcript, a user keyword, or a reformatting of the identical value.
- A correction must be phonetically near-identical to the original. No anchor variant or user keyword → no pair. Never "improve" grammar, word choice, or content.
- Never change numbers to different values; only their formatting.
- Unsure → no pair. An empty corrections list is a perfectly good answer. A ⟨marked⟩ word without an anchor goes into the keyword list, not the corrections.
- "original" is the exact surface form from the transcript WITHOUT ⟨⟩ markers; "corrected" must differ.
- Confidence 0.9-1.0 only for unambiguous cases; below 0.7 the pair will not be applied.

{rules}

# Task 3: keyword proposal

Propose the special names and terms whose spelling or identity a human should review. The test is NOT "do I understand this word" but "is this a canonical, correctly-spelled word of the transcript's language, or a name/coinage a reviewer might want to confirm". Include:
- EVERY SINGLE person name (first names, last names), organization, brand, product;
- software/tool/project names INCLUDING derived or Germanized forms (a tool "Wordly" appearing as "der Wordler");
- places (streets, districts, rivers) and domain-specific terms;
- any token that is not a standard dictionary word — coinages, anglicisms, plausible recognizer renderings of a name;
- ALL ⟨marked⟩ words that are unsure or look like a name, place, or term but not a canonical spelling (⟨Krellbachareal⟩: clearly a location, clearly misspelled) — include them all so a human can review them.

Scan the transcript line by line and collect EVERY single candidate. You must be EXTREMELY SENSITIVE and AGGRESSIVE in proposing keywords. If there is ANY doubt whether a word is a standard dictionary word, or if it is ANY kind of proper noun (every single name, place, or brand in the text), include it. Missing a name or a genuine term is considered a critical failure, much worse than including a borderline one. Also include all words that seem structurally weird or grammatically incorrect as they might be misheard names.

Per entry: term exactly as it appears AFTER your corrections (no ⟨⟩ markers); description (max 8 words, transcript's language) saying what it most likely is, "unsicher: …" when unsure; type: "person", "location", "institution", or "object" (products, tools, projects, domain terms).
- A fully understandable word ("Dropshipping") still belongs here if it is a name, product-derived, or not a dictionary word.
- Only terms that actually appear; never invent terms or facts. No ordinary dictionary words in normal use. No diarization labels.

# User keywords

The user prompt may contain a "User keywords" section with entries in the form "term: description". These entries are authoritative: their spelling and meaning are confirmed by the user.
- When a name you assign in Task 1 matches a user keyword (or is a phonetically close variant of one), use the keyword's spelling.
- If a transcript word or name is phonetically close to a user keyword (a plausible mishearing of it, e.g. "Silink" vs "SeaLink") and the context matches its description, propose a correction pair to the keyword's spelling (confidence 0.8-1.0).
- Do NOT repeat user keywords in your proposed keyword list — the user already has them. Propose only NEW terms.
"""


def _load_asset(name: str) -> str:
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


class TranscriptPostProcessingAgent(BaseAgent[None, TranscriptPostProcessingResult]):
    def __init__(self, config: LlmConfig):
        super().__init__(config, output_type=TranscriptPostProcessingResult, enable_thinking=False)

    @override
    def create_agent(self, model: Model) -> Agent[None, TranscriptPostProcessingResult]:
        instructions = TRANSCRIPT_POSTPROCESSING_INSTRUCTIONS.format(
            rules=_load_asset("transcript_rules.md"),
        )
        return Agent[None, TranscriptPostProcessingResult](
            model=model,
            output_type=TranscriptPostProcessingResult,
            instructions=instructions,
            model_settings={"temperature": 0.0},
        )
