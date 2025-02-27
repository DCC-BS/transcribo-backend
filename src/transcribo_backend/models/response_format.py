from enum import Enum


class ResponseFormat(str, Enum):
    """
    Enum representing the possible response formats for whisper transcription.

    Attributes:
        TEXT: Plain text format
        JSON: Standard JSON format
        JSON_DIARIZED: JSON format with speaker diarization
        VERBOSE_JSON: Detailed JSON format with additional information
        SRT: SubRip subtitle format
        VTT: WebVTT subtitle format
    """

    TEXT = "text"
    JSON = "json"
    JSON_DIARIZED = "json_diarized"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"
