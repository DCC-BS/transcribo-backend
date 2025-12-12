"""
API routers for the Transcribo backend.

This package contains organized API route handlers grouped by functionality.
"""

from transcribo_backend.routers.summarization import router as summarization_router
from transcribo_backend.routers.tasks import router as tasks_router
from transcribo_backend.routers.transcription import router as transcription_router

__all__ = [
    "summarization_router",
    "tasks_router",
    "transcription_router",
]
