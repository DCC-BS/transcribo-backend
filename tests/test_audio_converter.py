"""Tests for the improved audio converter module."""

import pytest
from returns.io import IOResult

from transcribo_backend.services.audio_converter import (
    AudioConversionError,
    convert_to_mp3,
)


class TestAudioConversionError:
    """Test custom exceptions."""

    def test_audio_conversion_error(self):
        """Test AudioConversionError can be raised and caught."""
        with pytest.raises(AudioConversionError):
            raise AudioConversionError("Test error message")


def test_convert_to_mp3_returns_io_result():
    """Test that convert_to_mp3 returns an IOResult."""
    # We don't necessarily need to run ffmpeg here, just check the type if we mock or pass empty data
    # Passing empty data will likely fail in ffmpeg but should still return a Failure(IOResult)
    result = convert_to_mp3(b"")
    assert isinstance(result, IOResult)
