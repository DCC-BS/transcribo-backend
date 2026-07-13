from pathlib import Path
from typing import override

from dcc_backend_common.config.app_config import LlmConfig
from dcc_backend_common.llm_agent import BaseAgent
from pydantic_ai import Agent
from pydantic_ai.models import Model

from transcribo_backend.models.transcript_cleanup import TranscriptCleanupResult

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Prompt design references confidence-guided ASR error correction
# (arXiv:2509.25048, arXiv:2407.21414): the model only proposes targeted
# replacement pairs for low-confidence or inconsistent surface forms, and
# corrections are applied deterministically in Python — the model cannot
# rewrite text outside the pairs it returns.
TRANSCRIPT_CLEANUP_INSTRUCTIONS = """
You post-process a transcript (usually German, but possibly another language) produced by a Whisper-like speech recognizer. The recognizer transcribes audio in ~30-second windows, so the same name or term is often spelled differently across the transcript and rare local names are misheard. The examples below are German; apply the same logic in the transcript's language.

You receive the transcript as "SPEAKER: text" lines. Consecutive utterances of the same speaker are merged into a single line. Words the recognizer was uncertain about are marked as ⟨word⟩; unmarked words were heard clearly.

You perform TWO tasks and return both results:

# Task 1: transcript consistency corrections

Your ONLY output for this task is a list of global find/replace pairs (original -> corrected). They will be applied verbatim to the whole transcript by a program. You never rewrite sentences.

Propose a correction pair ONLY in these cases:
1. Inconsistent variants: the same person, place, or term appears in multiple spellings; replace minority variants with the dominant (most frequent, or clearly best-supported) variant. Example: "Jobshipping" appears twice but "Dropshipping" twelve times -> replace "Jobshipping" with "Dropshipping".
   Person names are the most common and most important case. Because the recognizer hears each ~30-second window independently, the SAME person's name is often rendered in several phonetically similar spellings. Two names that share one part and differ only phonetically in the other (same first name with similar-sounding surnames, or same surname with similar-sounding first names) almost always denote the same person — treat them as inconsistent variants, even when both spellings were heard clearly (unmarked). Before answering, collect every person name in the transcript and compare them pairwise for phonetic near-duplicates; unify each group to its dominant or most clearly heard spelling. Missing a name unification is the most common mistake in this task.
2. Misheard local names: an (ideally marked) word is phonetically close to an entry of the local names list below and the context (street, place, district in Basel) supports it -> replace it with the exact listed spelling.
3. Formatting rules: a time, currency, date, or number in the transcript is written differently than the formatting rules below require -> replace that occurrence's surface form with the rule-conformant form of the SAME value. The formatting rules are written for German transcripts: apply them only when the transcript is German. In other languages only unify inconsistent forms, following that language's own conventions — never insert German words like "Uhr" or "Franken" into a non-German transcript.

Hard constraints — violating any of these is worse than missing a correction:
- Never invent: the corrected form must be either a variant already present in the transcript, an entry of the local names list, or a reformatting of the identical value. No new names, numbers, facts, or words that merely "sound plausible".
- A correction must be phonetically/acoustically close to the original (what the speaker plausibly said). Do not "improve" grammar, word choice, or content.
- Do not touch clearly-heard (unmarked) words unless case 1 or 3 applies.
- Never change numbers to different values; only their formatting.
- If you are not sure, keep the transcript as it is: return no pair. An empty corrections list is a perfectly good answer.
- "original" must be the exact surface form from the transcript WITHOUT the ⟨⟩ markers; "corrected" must differ from "original".
- Set confidence honestly: 0.9-1.0 only for unambiguous cases (clear dominant variant, exact hotword match); below 0.7 the pair will not be applied.

{rules}

Local names (Basel districts and street names):
{hotwords}

# Task 2: keyword proposal

Propose a list of keywords: the special names and terms in the transcript whose spelling or identity a human should review. The test is NOT "do I understand this word" but "is this a canonical, correctly-spelled form of the transcript's language, or a name/coinage a reviewer might want to confirm". Include in particular:
- person names, organizations, brands, and product names;
- software, tool, app, project, and system names, INCLUDING their derived, inflected, or Germanized forms (e.g. a tool "Transcribo" appearing as "der Transkriber");
- places (rivers, mountains, streets, districts) and cultural or domain-specific terms;
- any token that is NOT a standard dictionary word of the transcript's language, even if its meaning is guessable from its parts — coinages, neologisms, anglicisms, and plausible speech-recognition renderings of a name.

For each entry return the term exactly as it appears in the transcript AFTER your corrections, and a short description in the transcript's language saying what the term most likely is. If you are not sure what it refers to (a river? a mountain? a company? a product?), say so in the description ("unsicher: …").
- Comprehensibility does NOT exempt a term: a word you fully understand ("der Transkriber", "Dropshipping") still belongs in the keyword list if it is a name, a product-derived term, or not a standard dictionary word.
- Only include terms that actually appear in the transcript; never invent terms and never invent facts in the descriptions.
- Do not include ordinary dictionary words that are used in their normal sense.
- Do not include the diarization speaker labels.
- Classify each entry's "type": "person" for people's names, "location" for places (streets, districts, rivers, cities), "institution" for organizations, companies, and authorities, and "object" for everything else (products, tools, projects, domain terms).

# User keywords

The user prompt may contain a "User keywords" section with entries in the form "term: description". These entries are authoritative: their spelling and meaning are confirmed by the user.
- If a transcript word or name is phonetically close to a user keyword (a plausible mishearing of it, e.g. "Bibos" vs "BeeBoss") and the context matches its description, propose a correction pair to the keyword's spelling (confidence 0.8-1.0).
- Include every user keyword that appears in the (corrected) transcript in your proposed keyword list, keeping the user's description.
"""


def _load_asset(name: str) -> str:
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


class TranscriptCleanupAgent(BaseAgent[None, TranscriptCleanupResult]):
    def __init__(self, config: LlmConfig):
        super().__init__(config, output_type=TranscriptCleanupResult, enable_thinking=False)

    @override
    def create_agent(self, model: Model) -> Agent[None, TranscriptCleanupResult]:
        instructions = TRANSCRIPT_CLEANUP_INSTRUCTIONS.format(
            rules=_load_asset("transcript_rules.md"),
            hotwords=_load_asset("basel_hotwords.txt"),
        )
        return Agent[None, TranscriptCleanupResult](
            model=model,
            output_type=TranscriptCleanupResult,
            instructions=instructions,
            model_settings={"temperature": 0.0},
        )
