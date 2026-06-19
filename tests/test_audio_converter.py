"""Tests for the improved audio converter module."""

import shutil
import subprocess
from pathlib import Path

import pytest
from returns.io import IOResult
from returns.pipeline import is_successful
from returns.unsafe import unsafe_perform_io

from transcribo_backend.services.audio_converter import (
    AudioConversionError,
    convert_to_mp3,
    is_mp3_format,
)

ASSETS_DIR = Path(__file__).parent / "assets"

# Real sample files (downloaded once, committed under tests/assets) covering the input
# formats we expect in production: raw PCM, an already-MP3 file, lossless, AAC-in-MP4,
# Vorbis and a video container with an audio track.
SAMPLE_FILES = [
    "sample-3s.wav",
    "sample-3s.mp3",
    "sample-5s.flac",
    "sample-5s.m4a",
    "sample-5s.ogg",
    "sample-5s.mp4",
]

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


class TestIsMp3Format:
    """The MP3 sniff only inspects leading bytes, so we never load the whole file."""

    def test_detects_id3_tag(self):
        assert is_mp3_format(b"ID3" + b"\x00" * 100) is True

    def test_rejects_too_short(self):
        assert is_mp3_format(b"AB") is False

    def test_rejects_non_mp3(self):
        assert is_mp3_format(b"RIFF" + b"\x00" * 100) is False

    def test_works_on_header_chunk_only(self):
        # An ID3 tag at the very start is detected even when only a 1KB chunk is passed.
        header = (b"ID3" + b"\x00" * 4096)[:1024]
        assert is_mp3_format(header) is True


class TestAudioConversionError:
    """Test custom exceptions."""

    def test_audio_conversion_error(self):
        """Test AudioConversionError can be raised and caught."""
        with pytest.raises(AudioConversionError):
            raise AudioConversionError("Test error message")  # noqa: TRY003


def test_convert_to_mp3_returns_io_result():
    """Test that convert_to_mp3 returns an IOResult.

    A non-existent input path makes ffmpeg fail, but the @impure_safe wrapper should
    still surface that as an IOResult (Failure) rather than raising.
    """
    result = convert_to_mp3("/nonexistent/path/to/input.bin")
    assert isinstance(result, IOResult)
    assert not is_successful(result)


@requires_ffmpeg
class TestConvertRealFiles:
    """Convert real sample files of various formats end-to-end with ffmpeg."""

    @pytest.mark.parametrize("filename", SAMPLE_FILES)
    def test_converts_to_valid_mp3(self, filename: str):
        input_path = ASSETS_DIR / filename
        assert input_path.exists(), f"missing test asset: {input_path}"

        result = convert_to_mp3(str(input_path))

        assert is_successful(result), f"conversion failed for {filename}"
        output_path = Path(unsafe_perform_io(result.unwrap()))
        try:
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            # The leading bytes sniff as MP3...
            assert is_mp3_format(output_path.read_bytes())
            # ...and ffprobe confirms the codec plus the conversion settings
            # (mono, 16 kHz) the converter is configured to produce.
            stream = self._probe(output_path)
            assert stream["codec_name"] == "mp3"
            assert stream["channels"] == "1"
            assert stream["sample_rate"] == "16000"
        finally:
            output_path.unlink(missing_ok=True)

    @staticmethod
    def _probe(path: Path) -> dict[str, str]:
        ffprobe = shutil.which("ffprobe")
        assert ffprobe is not None, "ffprobe not installed"
        out = subprocess.run(  # noqa: S603
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels,sample_rate",
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream: dict[str, str] = {}
        for line in out.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                stream[key] = value
        return stream
