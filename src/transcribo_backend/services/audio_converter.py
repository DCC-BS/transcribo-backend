import io

from fastapi import HTTPException
from pydub import AudioSegment


def is_mp3_format(audio_data: bytes) -> bool:
    """
    Check if the audio data is already in MP3 format.
    """
    # MP3 files typically start with ID3 tag or 0xFFFB/0xFFF3/0xFFF2 frame sync
    return audio_data.startswith(b"ID3") or (
        len(audio_data) > 2 and audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0
    )


def convert_to_mp3(file_data: bytes, original_format: str | None) -> bytes:
    """
    Convert audio or video data to MP3 format using pydub.

    Args:
        file_data: The audio/video bytes
        original_format: The format of the input file (e.g., 'mp3', 'm4a', 'mp4')
                        If None, will try to determine the format

    Returns:
        Bytes of the audio in MP3 format
    """
    try:
        # Create a BytesIO object for input
        file_io = io.BytesIO(file_data)

        # If format is not provided, try with common formats
        if original_format is None:
            # Include both audio and video formats
            formats_to_try = ["wav", "m4a", "ogg", "flac", "mp4", "avi", "mkv", "mov", "webm"]
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

        # Export to MP3
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3")
        return mp3_io.getvalue()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to MP3: {e!s}")
