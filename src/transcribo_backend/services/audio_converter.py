import io

from fastapi import HTTPException
from pydub import AudioSegment


def is_wav_format(audio_data: bytes) -> bool:
    """
    Check if the audio data is already in WAV format.
    """
    # WAV files start with the RIFF header
    return audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]


def convert_to_wav(file_data: bytes, original_format: str) -> bytes:
    """
    Convert audio or video data to WAV format using pydub.

    Args:
        file_data: The audio/video bytes
        original_format: The format of the input file (e.g., 'mp3', 'm4a', 'mp4')
                        If None, will try to determine the format

    Returns:
        Bytes of the audio in WAV format
    """
    try:
        # Create a BytesIO object for input
        file_io = io.BytesIO(file_data)

        # If format is not provided, try with common formats
        if original_format is None:
            # Include both audio and video formats
            formats_to_try = ["mp3", "m4a", "ogg", "flac", "mp4", "avi", "mkv", "mov", "webm"]
            for fmt in formats_to_try:
                try:
                    file_io.seek(0)
                    audio = AudioSegment.from_file(file_io, format=fmt)
                    break
                except Exception:  # noqa: S112
                    continue
            else:  # If loop completes without break, all formats failed
                raise ValueError("Could not determine file format")
        else:
            audio = AudioSegment.from_file(file_io, format=original_format)

        # Export to WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to WAV: {e!s}")
