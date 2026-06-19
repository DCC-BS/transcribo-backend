import os

from dcc_backend_common.config import get_env_or_throw, log_secret
from dcc_backend_common.config.app_config import LlmConfig
from dcc_backend_common.logger import get_logger
from pydantic import Field

logger = get_logger(__name__)

# Default maximum upload size: 2 GiB
_DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class AppConfig(LlmConfig):
    client_url: str = Field(description="The URL for the client application")
    hmac_secret: str = Field(description="The secret key for HMAC authentication")
    whisper_url: str = Field(description="The URL for the Whisper API")
    whisper_health_check_url: str = Field(description="The URL for the Whisper API health check endpoint")
    llm_health_check_url: str = Field(description="The URL for the LLM API health check endpoint")
    max_upload_bytes: int = Field(
        default=_DEFAULT_MAX_UPLOAD_BYTES,
        description="Maximum accepted upload size in bytes for transcription requests",
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        llm_base_url: str = get_env_or_throw("LLM_URL")
        llm_health_check_url: str = get_env_or_throw("LLM_HEALTH_CHECK_URL")
        api_key: str = get_env_or_throw("LLM_API_KEY")
        llm_model: str = get_env_or_throw("LLM_MODEL")
        client_url: str = get_env_or_throw("CLIENT_URL")
        hmac_secret: str = get_env_or_throw("HMAC_SECRET")
        whisper_url: str = get_env_or_throw("WHISPER_URL")
        whisper_health_check_url: str = get_env_or_throw("WHISPER_HEALTH_CHECK_URL")
        raw_max_upload_bytes = os.getenv("MAX_UPLOAD_BYTES", str(_DEFAULT_MAX_UPLOAD_BYTES))
        try:
            max_upload_bytes: int = int(raw_max_upload_bytes)
        except ValueError:
            logger.warning(
                "Invalid MAX_UPLOAD_BYTES=%r; falling back to default %d",
                raw_max_upload_bytes,
                _DEFAULT_MAX_UPLOAD_BYTES,
            )
            max_upload_bytes = _DEFAULT_MAX_UPLOAD_BYTES

        return cls(
            llm_url=llm_base_url,
            llm_health_check_url=llm_health_check_url,
            llm_api_key=api_key,
            llm_model=llm_model,
            client_url=client_url,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
            whisper_health_check_url=whisper_health_check_url,
            max_upload_bytes=max_upload_bytes,
        )

    def __str__(self) -> str:
        return f"""
        AppConfig(
            client_url={self.client_url},
            llm_url={self.llm_url},
            llm_health_check_url={self.llm_health_check_url},
            llm_api_key={log_secret(self.llm_api_key)},
            llm_model={self.llm_model},
            hmac_secret={log_secret(self.hmac_secret)},
            whisper_url={self.whisper_url}
            whisper_health_check_url={self.whisper_health_check_url},
            max_upload_bytes={self.max_upload_bytes},
        )
        """
