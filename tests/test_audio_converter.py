"""Tests for the improved audio converter module."""

import pytest

from transcribo_backend.services.audio_converter import (
    AudioConversionError,
)


class TestAudioConversionError:
    """Test custom exceptions."""

    def test_audio_conversion_error(self):
        """Test AudioConversionError can be raised and caught."""
        with pytest.raises(AudioConversionError):
            raise AudioConversionError("Test error message")
