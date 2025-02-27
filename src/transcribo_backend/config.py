"""Configuration module for the Transcribo backend.

This module loads configuration values from environment variables.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    whisper_api: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings: Application settings object

        Raises:
            ValueError: If a required environment variable is missing
        """
        load_dotenv()  # Load .env file if present

        whisper_api = os.getenv("WHISPER_API")
        if not whisper_api:
            raise ValueError("WHISPER_API environment variable is required")

        return cls(whisper_api=whisper_api)


# Create a global settings instance for easy import
settings = Settings.from_env()
