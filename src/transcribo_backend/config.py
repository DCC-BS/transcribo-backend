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
    llm_api: str
    api_key: str

    def __repr__(self) -> str:
        """Custom repr that masks sensitive data."""
        return (
            f"Settings(whisper_api={self.whisper_api}, llm_api={self.llm_api}, api_key=********, hmac_secret=********)"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings: Application settings object

        Raises:
            ValueError: If a required environment variable is missing
        """
        if os.path.exists(".env"):
            load_dotenv()  # Load .env file if present

        whisper_api = os.getenv("WHISPER_API")
        llm_api = os.getenv("LLM_API")
        api_key = os.getenv("API_KEY")
        hmac_secret = os.getenv("HMAC_SECRET")

        if not whisper_api:
            raise ValueError("WHISPER_API environment variable is required")

        if not llm_api:
            raise ValueError("LLM_API environment variable is required")

        if not api_key:
            raise ValueError("API_KEY environment variable is required")

        if not hmac_secret:
            raise ValueError("HMAC_SECRET environment variable is required")

        return cls(whisper_api=whisper_api, llm_api=llm_api, api_key=api_key, hmac_secret=hmac_secret)


# Create a global settings instance for easy import
settings = Settings.from_env()
