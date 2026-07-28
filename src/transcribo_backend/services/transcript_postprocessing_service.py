import re
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher

from returns.future import future_safe

from transcribo_backend.agents.transcript_postprocessing_agent import TranscriptPostProcessingAgent
from transcribo_backend.models.keywords import Keyword
from transcribo_backend.models.speaker_assignment import SpeakerNameAssignment
from transcribo_backend.models.transcript_cleanup import TranscriptCorrection
from transcribo_backend.models.transcript_postprocessing import TranscriptPostProcessingResult
from transcribo_backend.models.transcription_response import Segment

# Same bound as the summarize endpoint, keeps the LLM call within context.
_MAX_TRANSCRIPT_CHARS = 32_000 * 4

# Share of the clamp budget reserved for the transcript tail: name evidence
# clusters at both ends of a conversation — introductions at the start,
# thanks and sign-offs ("Vielen Dank, Frau Müller") at the end.
_TAIL_CLAMP_SHARE = 0.3

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

# Marks the gap where the middle of an over-long transcript was dropped.
_ELISION_LINE = "[…]"


# Fallback label for segments the diarizer left unattributed. whisper_service
# applies the same default upstream; this is the guard for segments that never
# went through it.
_UNKNOWN_SPEAKER = "Unknown"

# Diarization labels are shortened for the prompt ("Speaker_00" -> "S00"): the
# long label costs 7 tokens per transcript line against 5 for the short one
# (measured with the Gemma tokenizer), ~5% of a prompt near the clamp. The
# encoding never leaves the prompt boundary — assignments are decoded back
# before anything downstream sees them.
#
# Case-insensitive because whisper_service normalises labels with
# .capitalize(): the pipeline emits "Speaker_00" where the diarizer's own form
# is "SPEAKER_00", and both must shorten.
_LONG_SPEAKER_LABEL_RE = re.compile(r"^speaker[_ ]?(\d+)$", re.IGNORECASE)


def encode_speaker_label(label: str) -> str:
    """Shorten a diarization label for the prompt: ``Speaker_00`` -> ``S00``.

    Digits are preserved verbatim, so zero-padding survives the round trip.
    Any other shape (e.g. ``"Unknown"``) passes through unchanged — the
    encoding is an optimisation, never a requirement.
    """
    match = _LONG_SPEAKER_LABEL_RE.match(label)
    return f"S{match.group(1)}" if match else label


def decode_speaker_labels(assignments: list[SpeakerNameAssignment], segments: list[Segment]) -> None:
    """Restore the original diarization labels on ``assignments``, in place.

    A lookup against the labels actually present in ``segments``, not an
    inverse regex: that restores the original spelling exactly and cannot
    expand a diarizer's native short label into one the transcript never had.
    Labels with no counterpart in the transcript are left untouched rather than
    given a fabricated expansion.
    """
    by_short = {encode_speaker_label(label): label for label in _distinct_speakers(segments)}
    for assignment in assignments:
        assignment.speaker = by_short.get(assignment.speaker, assignment.speaker)


def _distinct_speakers(segments: list[Segment]) -> set[str]:
    """The distinct speaker labels in ``segments``, unattributed ones folded into the fallback."""
    return {segment.speaker or _UNKNOWN_SPEAKER for segment in segments}


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


def _line_cost(line: str) -> int:
    """Budget a line consumes: its length plus the newline that joins it."""
    return len(line) + 1


def _take_lines_within(lines: Iterable[str], budget: int) -> tuple[list[str], int]:
    """Take lines in order while they fit ``budget`` chars; return them and the chars used."""
    taken: list[str] = []
    used = 0
    for line in lines:
        cost = _line_cost(line)
        if used + cost > budget:
            break
        taken.append(line)
        used += cost
    return taken, used


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

    Clamping keeps the head AND the tail of the transcript (middle replaced
    by a […] line): introductions cluster at the start, but thanks and
    sign-offs naming speakers cluster at the end.
    """
    merged: list[tuple[str, list[str]]] = []
    for segment in segments:
        speaker = segment.speaker or _UNKNOWN_SPEAKER
        text = _mark_uncertain_words(segment) if mark_uncertain else segment.text.strip()
        if merged and merged[-1][0] == speaker:
            merged[-1][1].append(text)
        else:
            merged.append((speaker, [text]))

    # Grouping above compares the original labels; only the rendered line uses
    # the short form, so merging is unaffected by the encoding. Encoding is
    # resolved once per speaker rather than once per line.
    short = {speaker: encode_speaker_label(speaker) for speaker, _ in merged}
    lines = [f"{short[speaker]}: {' '.join(texts)}" for speaker, texts in merged]
    if sum(_line_cost(line) for line in lines) <= max_chars:
        return "\n".join(lines)

    tail, tail_used = _take_lines_within(reversed(lines), int(max_chars * _TAIL_CLAMP_SHARE))
    tail.reverse()
    # The elision line joins the two halves and costs budget like any other line.
    elision_used = _line_cost(_ELISION_LINE) if tail else 0
    head, _ = _take_lines_within(lines[: len(lines) - len(tail)], max_chars - tail_used - elision_used)

    if not tail:
        return "\n".join(head)
    return "\n".join([*head, _ELISION_LINE, *tail])


# Spoken forms of symbols that legitimately disappear when a rule rewrites
# them to the symbol itself (e-mail rule: "at"/"ät" -> @, "punkt"/"dot" -> .).
_SPOKEN_SYMBOL_WORDS = {"at", "ät", "punkt", "dot"}


def _letter_word_count(text: str) -> int:
    """Count letter-only tokens, ignoring digits, punctuation, and spoken symbol words."""
    return len([w for w in re.findall(r"[^\W\d_]+", text) if w.lower() not in _SPOKEN_SYMBOL_WORDS])


def _boundary_pattern(surface_form: str) -> re.Pattern[str]:
    """Match ``surface_form`` only as a whole word, never inside a longer one.

    Shared by the text and the name pass so the two can never disagree on what
    counts as a word boundary.
    """
    return re.compile(rf"(?<!\w){re.escape(surface_form)}(?!\w)")


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

        pattern = _boundary_pattern(original)
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

    The closest term above ``min_ratio`` wins, not the first one found: with
    two similar confirmed spellings in the vocabulary the outcome must depend
    on similarity, never on the order of the keyword list.
    """
    # Lowered once per term, not once per (assignment, term) pair.
    person_terms = [(k.term.strip(), k.term.strip().lower()) for k in keywords if k.type == "person" and k.term.strip()]
    for assignment in assignments:
        name = assignment.name
        if not name:
            continue
        lowered = name.lower()
        ratio, term = max(
            ((SequenceMatcher(None, lowered, low).ratio(), term) for term, low in person_terms),
            default=(0.0, name),
        )
        if ratio >= min_ratio:
            assignment.name = term
    return assignments


def apply_corrections_to_names(
    assignments: list[SpeakerNameAssignment], corrections: list[TranscriptCorrection]
) -> list[SpeakerNameAssignment]:
    """Apply the already-applied text corrections to the inferred names too."""
    for correction in corrections:
        pattern = _boundary_pattern(correction.original.strip())
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


def render_keywords(keywords: list[Keyword] | None) -> list[str]:
    """Render user-confirmed keywords as the ``term: description`` lines the prompt uses."""
    return [f"{entry.term}: {entry.description}" for entry in keywords or []]


class TranscriptPostProcessingService:
    """Runs the post-processing agent over a transcript and applies its output.

    Attributes:
        agent: The agent performing title, speaker, cleanup, and keyword tasks.
    """

    def __init__(
        self,
        transcript_postprocessing_agent: TranscriptPostProcessingAgent,
    ):
        """Initialize the service.

        Args:
            transcript_postprocessing_agent: The post-processing agent to run.
        """
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
        inferred title, applied corrections, speaker assignments, and proposed
        keywords.
        """
        keywords_section = ""
        if keywords:
            keywords_section = "\n\nUser keywords:\n" + "\n".join(render_keywords(keywords))

        prompt = build_postprocessing_transcript(segments, mark_uncertain=True) + keywords_section
        result: TranscriptPostProcessingResult = await self.agent.run(prompt)

        # Undo the prompt-side label shortening immediately: everything below
        # this line, and everything the route returns, deals in real labels.
        decode_speaker_labels(result.speaker_assignments, segments)

        assignments = enumerate_roles(result.speaker_assignments)
        if keywords:
            assignments = apply_keyword_spellings_to_names(assignments, keywords)

        applied = apply_corrections(segments, result.corrections)
        assignments = apply_corrections_to_names(assignments, applied)

        return TranscriptPostProcessingResult(
            title=result.title,
            corrections=applied,
            speaker_assignments=assignments,
            keywords=result.keywords,
        )
