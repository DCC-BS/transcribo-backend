"""Shared test configuration.

Importing the routes pulls in the DI container, which builds ``AppConfig.from_env()`` at
import time. Provide dummy values so unit tests can import the app without a real
environment. ``setdefault`` means a real environment (e.g. for integration runs) still
takes precedence.
"""

import os

import pytest

_DUMMY_ENV = {
    "LLM_URL": "http://llm.test",
    "LLM_HEALTH_CHECK_URL": "http://llm.test/health",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-model",
    "CLIENT_URL": "http://client.test",
    "HMAC_SECRET": "test-secret",
    "WHISPER_URL": "http://whisper.test",
    "WHISPER_HEALTH_CHECK_URL": "http://whisper.test/health",
}

for _key, _value in _DUMMY_ENV.items():
    os.environ.setdefault(_key, _value)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
