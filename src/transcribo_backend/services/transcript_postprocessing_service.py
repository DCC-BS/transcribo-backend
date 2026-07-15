import re
from collections import Counter
from difflib import SequenceMatcher

from returns.future import future_safe

from transcribo_backend.agents.transcript_postprocessing_agent import TranscriptPostProcessingAgent
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection
from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult
from transcribo_backend.models.transcription_response import Segment
from transcribo_backend.utils.app_config import AppConfig

# Same bound as the summarize endpoint; name evidence typically appears early,
# so clamping long transcripts keeps the LLM call within context.
_MAX_TRANSCRIPT_CHARS = 32_000 * 4

# Corrections below this confidence are proposed by the model but not applied
# (arXiv:2407.21414 found ~0.7 a good over-correction guard).
_MIN_APPLY_CONFIDENCE = 0.7

# Words below this Whisper probability are marked as uncertain in the cleanup
# prompt (the primary correction and keyword candidates). Kept low so only
# genuinely dubious words are marked — on noisy recordings a higher threshold
# floods the prompt with ordinary words.
_UNCERTAIN_WORD_THRESHOLD = 0.3

# Tokens shorter than this are never marked: they are function words whose
# recognition probability is noisy but whose spelling is never in question.
_MIN_MARK_TOKEN_LENGTH = 4


def _mark_uncertain_words(segment: Segment) -> str:
    """Mark low-probability Whisper words in the segment text as ⟨word⟩."""
    text = segment.text.strip()
    for word in segment.words or []:
        token = word.word.strip()
        if len(token) < _MIN_MARK_TOKEN_LENGTH or word.probability >= _UNCERTAIN_WORD_THRESHOLD:
            continue
        # Whole-token match only — never mark a substring inside a word.
        text = re.sub(
            rf"(?<![\w⟨]){re.escape(token)}(?![\w⟩])",
            f"⟨{token}⟩",
            text,
            count=1,
        )
    return text


def build_postprocessing_transcript(
    segments: list[Segment],
    max_chars: int = _MAX_TRANSCRIPT_CHARS,
    mark_uncertain: bool = False,
) -> str:
    """
    Render segments as "SPEAKER: text" lines, clamped to ``max_chars``.

    Consecutive segments of the same speaker are merged into one line (more
    context per turn). When ``mark_uncertain`` is set, words the recognizer
    was uncertain about are marked ⟨…⟩ — the primary correction and keyword
    candidates for the post-processing prompt.
    """
    merged: list[tuple[str, list[str]]] = []
    for segment in segments:
        speaker = segment.speaker or "Unknown"
        text = _mark_uncertain_words(segment) if mark_uncertain else segment.text.strip()
        if merged and merged[-1][0] == speaker:
            merged[-1][1].append(text)
        else:
            merged.append((speaker, [text]))

    lines: list[str] = []
    total = 0
    for speaker, texts in merged:
        line = f"{speaker}: {' '.join(texts)}"
        total += len(line) + 1
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)


# Spoken forms of symbols that legitimately disappear when a rule rewrites
# them to the symbol itself (e-mail rule: "at"/"ät" -> @, "punkt"/"dot" -> .).
_SPOKEN_SYMBOL_WORDS = {"at", "ät", "punkt", "dot"}


def _letter_word_count(text: str) -> int:
    """Count letter-only tokens, ignoring digits, punctuation, and spoken symbol words."""
    return len([w for w in re.findall(r"[^\W\d_]+", text) if w.lower() not in _SPOKEN_SYMBOL_WORDS])


def apply_corrections(
    segments: list[Segment],
    corrections: list[TranscriptCorrection],
    min_confidence: float = _MIN_APPLY_CONFIDENCE,
) -> list[TranscriptCorrection]:
    """
    Apply high-confidence corrections to the segment texts in place.

    Replacements are exact, word-boundary, case-sensitive — the deterministic
    application is what guarantees consistency across all fragments and that
    nothing outside the returned pairs changes. Returns the corrections that
    actually changed at least one segment.

    A correction is a targeted surface-form fix (spelling, proper name, formatting)
    and must never delete words: reformatting only rewrites digit/punctuation
    tokens (``1430 Uhr`` -> ``14:30 Uhr``), so it never lowers the letter-word
    count. A correction whose ``corrected`` has fewer letter words than its
    ``original`` is the model trying to shorten/rewrite content — applying it
    globally would silently drop text, so it is skipped.
    """
    applied: list[TranscriptCorrection] = []
    for correction in corrections:
        original = correction.original.strip()
        corrected = correction.corrected.strip()
        if not original or not corrected or original == corrected or correction.confidence < min_confidence:
            continue
        if _letter_word_count(corrected) < _letter_word_count(original):
            continue

        pattern = re.compile(rf"(?<!\w){re.escape(original)}(?!\w)")
        changed = False
        for segment in segments:
            new_text = pattern.sub(corrected, segment.text)
            if new_text != segment.text:
                segment.text = new_text
                changed = True
        if changed:
            applied.append(correction)
    return applied


def apply_keyword_spellings_to_names(
    assignments: list[SpeakerNameAssignment],
    keywords: list[Keyword],
    min_ratio: float = 0.8,
) -> list[SpeakerNameAssignment]:
    """
    Snap inferred speaker names to user-confirmed keyword spellings.

    The speaker prompt asks the model to prefer keyword spellings, but that is
    unreliable — this deterministic pass guarantees it: a name close enough to
    a keyword term (e.g. "Lena Feldman" vs "Lena Feldmann") is replaced
    by the keyword's exact spelling. Only person keywords are considered: a
    speaker name must never snap to a merely similar location/institution/
    object term (e.g. a speaker "Basler" to the location "Basel").
    """
    person_terms = [k.term.strip() for k in keywords if k.type == "person" and k.term.strip()]
    for assignment in assignments:
        if not assignment.name:
            continue
        for term in person_terms:
            if assignment.name == term:
                continue
            ratio = SequenceMatcher(None, assignment.name.lower(), term.lower()).ratio()
            if ratio >= min_ratio:
                assignment.name = term
                break
    return assignments


def apply_corrections_to_names(
    assignments: list[SpeakerNameAssignment], corrections: list[TranscriptCorrection]
) -> list[SpeakerNameAssignment]:
    """Apply the already-applied text corrections to the inferred names too."""
    for correction in corrections:
        pattern = re.compile(rf"(?<!\w){re.escape(correction.original.strip())}(?!\w)")
        for assignment in assignments:
            if assignment.name:
                assignment.name = pattern.sub(correction.corrected.strip(), assignment.name)
    return assignments


def enumerate_roles(assignments: list[SpeakerNameAssignment]) -> list[SpeakerNameAssignment]:
    """
    Disambiguate speakers that fall back to a role (no name) but share it.

    When several nameless speakers have the same role, number them
    ("Dolmetscher 1", "Dolmetscher 2") so the label stays unique. A speaker
    with a name keeps its name and is never enumerated.
    """
    fallback_counts = Counter(a.role for a in assignments if not a.name and a.role)
    running: dict[str, int] = {}
    for assignment in assignments:
        role = assignment.role
        if not assignment.name and role and fallback_counts[role] > 1:
            running[role] = running.get(role, 0) + 1
            assignment.role = f"{role} {running[role]}"
    return assignments


class TranscriptPostProcessingService:
    def __init__(
        self,
        app_config: AppConfig,
        transcript_postprocessing_agent: TranscriptPostProcessingAgent,
    ):
        self.app_config = app_config
        self.agent = transcript_postprocessing_agent

    @future_safe
    async def post_process(
        self, segments: list[Segment], keywords: list[Keyword] | None = None
    ) -> TranscriptPostProcessingResult:
        """
        Run speaker name/role inference, cleanup, and keyword proposal as ONE
        LLM call on the transcript (raw diarization labels, uncertain words
        marked). ``keywords`` entries are user-confirmed and treated as
        authoritative spellings by the prompt; they are additionally enforced
        deterministically on the inferred names.

        The returned correction pairs are applied deterministically to the
        segment texts (and to the inferred names, so a misheard surname
        unified in the text follows in the assignment too). Returns the
        applied corrections, the speaker assignments, and the proposed
        keywords.
        """
        keywords_section = ""
        if keywords:
            entries = "\n".join(f"{entry.term}: {entry.description}" for entry in keywords)
            keywords_section = f"\n\nUser keywords:\n{entries}"

        prompt = build_postprocessing_transcript(segments, mark_uncertain=True) + keywords_section
        result: TranscriptPostProcessingResult = await self.agent.run(prompt)

        assignments = enumerate_roles(result.speaker_assignments)
        if keywords:
            assignments = apply_keyword_spellings_to_names(assignments, keywords)

        applied = apply_corrections(segments, result.corrections)
        assignments = apply_corrections_to_names(assignments, applied)

        return TranscriptPostProcessingResult(
            corrections=applied,
            speaker_assignments=assignments,
            keywords=result.keywords,
        )
