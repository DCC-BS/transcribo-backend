import subprocess
import tempfile
from pathlib import Path

from dcc_backend_common.logger import get_logger
from fastapi import HTTPException

logger = get_logger(__name__)


class AudioConversionError(Exception):
    """Custom exception for audio conversion errors."""

    def __init__(self, error_message: str):
        super().__init__(f"FFmpeg conversion failed: {error_message}")


def is_mp3_format(audio_data: bytes) -> bool:
    """
    Check if the audio data is already in MP3 format with improved detection.

    Args:
        audio_data: The binary data to check

    Returns:
        True if the data appears to be MP3 format
    """
    if len(audio_data) < 3:
        return False

    # Check for ID3 tag
    if audio_data.startswith(b"ID3"):
        return True

    # Check for MP3 frame sync (more thorough check)
    for i in range(min(1024, len(audio_data) - 1)):  # Check first 1KB
        if audio_data[i] == 0xFF and (audio_data[i + 1] & 0xE0) == 0xE0 and i + 3 < len(audio_data):
            header = (audio_data[i] << 24) | (audio_data[i + 1] << 16) | (audio_data[i + 2] << 8) | audio_data[i + 3]
            # Check if it's a valid MP3 frame header
            version = (header >> 19) & 0x3
            layer = (header >> 17) & 0x3
            if version != 1 and layer != 0:  # Valid version and layer
                return True

    return False


def convert_to_mp3(file_data: bytes) -> bytes:
    """
    Convert audio or video data to MP3 format using FFmpeg with balanced quality settings.

    Args:
        file_data: The audio/video bytes

    Returns:
        Bytes of the audio in MP3 format

    Raises:
        AudioConversionError: If conversion fails
        HTTPException: For HTTP-specific errors
    """
    try:
        file_size_mb = len(file_data) / (1024 * 1024)
        logger.info(f"Starting FFmpeg audio conversion, file size: {file_size_mb:.1f}MB")

        # Create temporary files for input and output
        with (
            tempfile.NamedTemporaryFile(delete=False) as input_temp,
            tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as output_temp,
        ):
            input_path = input_temp.name
            output_path = output_temp.name

            try:
                # Write input data to temporary file
                input_temp.write(file_data)
                input_temp.flush()

                # Build FFmpeg command with balanced quality settings and resample to 16kHz
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ac",
                    "1",
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "64k",
                    "-ar",
                    "16000",
                    output_path,
                ]

                logger.info("Running FFmpeg conversion with balanced quality (64k bitrate)")

                # Run FFmpeg
                result = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode != 0:
                    error_msg = result.stderr or "Unknown FFmpeg error"
                    logger.error(f"FFmpeg conversion failed: {error_msg}")
                    raise AudioConversionError(error_msg)

                # Read the converted file
                with open(output_path, "rb") as f:
                    converted_data = f.read()

                output_size_mb = len(converted_data) / (1024 * 1024)
                compression_ratio = file_size_mb / output_size_mb if output_size_mb > 0 else 0
                logger.info(
                    f"FFmpeg conversion completed. Output size: {output_size_mb:.1f}MB (compression ratio: {compression_ratio:.1f}x)"
                )

                return converted_data

            finally:
                # Clean up temporary files
                try:
                    Path(input_path).unlink(missing_ok=True)
                    Path(output_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary files: {e}")

    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.exception("FFmpeg subprocess error")
        raise AudioConversionError(str(e)) from e
    except AudioConversionError:
        # Re-raise our custom exceptions as-is
        raise
    except Exception as e:
        logger.exception("Unexpected error during audio conversion")
        raise HTTPException(status_code=500, detail=f"Error converting to MP3: {e!s}") from e
