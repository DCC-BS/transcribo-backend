from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float
    speaker: str | None = None


class VerboseSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    speaker: str | None = None


class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[Word]
    segments: list[VerboseSegment]


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscriptionResponse(BaseModel):
    segments: list[Segment]
