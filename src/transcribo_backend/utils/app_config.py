from dcc_backend_common.config import get_env_or_throw, log_secret
from dcc_backend_common.config.app_config import LlmConfig
from pydantic import Field


class AppConfig(LlmConfig):
    client_url: str = Field(description="The URL for the client application")
    hmac_secret: str = Field(description="The secret key for HMAC authentication")
    whisper_url: str = Field(description="The URL for the Whisper API")

    @classmethod
    def from_env(cls) -> "AppConfig":
        llm_base_url: str = get_env_or_throw("LLM_BASE_URL")
        api_key: str = get_env_or_throw("API_KEY")
        llm_model: str = get_env_or_throw("LLM_MODEL")
        client_url: str = get_env_or_throw("CLIENT_URL")
        hmac_secret: str = get_env_or_throw("HMAC_SECRET")
        whisper_url: str = get_env_or_throw("WHISPER_URL")

        return cls(
            llm_url=llm_base_url,
            llm_api_key=api_key,
            llm_model=llm_model,
            client_url=client_url,
            hmac_secret=hmac_secret,
            whisper_url=whisper_url,
        )

    def __str__(self) -> str:
        return f"""
        AppConfig(
            client_url={self.client_url},
            llm_url={self.llm_url},
            llm_api_key={log_secret(self.llm_api_key)},
            llm_model={self.llm_model},
            hmac_secret={log_secret(self.hmac_secret)},
            whisper_url={self.whisper_url}
        )
        """
