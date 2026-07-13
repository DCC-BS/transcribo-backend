import re
from collections import Counter

from returns.future import future_safe

from transcribo_backend.agents.speaker_inference_agent import SpeakerInferenceAgent
from transcribo_backend.agents.transcript_cleanup_agent import TranscriptCleanupAgent
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerAssignmentResult, SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCleanupResult, TranscriptCorrection
from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult
from transcribo_backend.models.transcription_response import Segment
from transcribo_backend.utils.app_config import AppConfig

# Same bound as the summarize endpoint; name evidence typically appears early,
# so clamping long transcripts keeps the LLM call within context.
_MAX_TRANSCRIPT_CHARS = 32_000 * 4

# Words below this Whisper probability are marked as uncertain in the prompt
# (the cleanup model's primary correction candidates).
_UNCERTAIN_WORD_THRESHOLD = 0.5

# Corrections below this confidence are proposed by the model but not applied
# (arXiv:2407.21414 found ~0.7 a good over-correction guard).
_MIN_APPLY_CONFIDENCE = 0.7


def _mark_uncertain_words(segment: Segment) -> str:
    """Mark low-probability Whisper words in the segment text as ⟨word⟩."""
    text = segment.text.strip()
    for word in segment.words or []:
        token = word.word.strip()
        if not token or word.probability >= _UNCERTAIN_WORD_THRESHOLD:
            continue
        text = re.sub(rf"(?<!⟨){re.escape(token)}(?!⟩)", f"⟨{token}⟩", text, count=1)
    return text


def build_postprocessing_transcript(
    segments: list[Segment], max_chars: int = _MAX_TRANSCRIPT_CHARS, mark_uncertain: bool = True
) -> str:
    """
    Render segments as "SPEAKER: text" lines, clamped to ``max_chars``.

    Consecutive segments of the same speaker are merged into one line (more
    context per turn). When ``mark_uncertain`` is set, words the recognizer was
    uncertain about are marked ⟨…⟩ — useful for the cleanup task, noise for the
    speaker task, so the speaker call passes ``mark_uncertain=False``.
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


def _letter_word_count(text: str) -> int:
    """Count letter-only tokens, ignoring digits and punctuation."""
    return len(re.findall(r"[^\W\d_]+", text))


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

    A correction is a targeted surface-form fix (spelling, hotword, formatting)
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
        speaker_inference_agent: SpeakerInferenceAgent,
        transcript_cleanup_agent: TranscriptCleanupAgent,
    ):
        self.app_config = app_config
        self.speaker_agent = speaker_inference_agent
        self.cleanup_agent = transcript_cleanup_agent

    @future_safe
    async def post_process(
        self, segments: list[Segment], keywords: list[Keyword] | None = None
    ) -> TranscriptPostProcessingResult:
        """
        Run cleanup (+ keywords) and then speaker name/role inference as two
        sequential LLM calls. ``keywords`` entries are user-confirmed and
        treated as authoritative spellings by the cleanup prompt.

        Cleanup runs first and its corrections are applied in place, so the
        speaker inference sees the consistent surface forms. Returns the
        applied corrections, the speaker assignments, and the proposed keywords.
        """
        keywords_section = ""
        if keywords:
            entries = "\n".join(f"{entry.term}: {entry.description}" for entry in keywords)
            keywords_section = f"\n\nUser keywords:\n{entries}"

        # 1. Cleanup + keywords on the uncertainty-marked transcript.
        cleanup_prompt = build_postprocessing_transcript(segments, mark_uncertain=True) + keywords_section
        cleanup_result: TranscriptCleanupResult = await self.cleanup_agent.run(cleanup_prompt)
        applied = apply_corrections(segments, cleanup_result.corrections)

        # 2. Speaker name + role inference on the now-corrected plain transcript.
        # The keywords go to this agent too so assigned names use the
        # user-confirmed spellings.
        speaker_prompt = build_postprocessing_transcript(segments, mark_uncertain=False) + keywords_section
        speaker_result: SpeakerAssignmentResult = await self.speaker_agent.run(speaker_prompt)
        assignments = enumerate_roles(speaker_result.assignments)

        return TranscriptPostProcessingResult(
            corrections=applied,
            speaker_assignments=assignments,
            keywords=cleanup_result.keywords,
        )
