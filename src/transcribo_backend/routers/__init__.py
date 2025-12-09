"""
API routers for the Transcribo backend.

This package contains organized API route handlers grouped by functionality.
"""

from transcribo_backend.routers.tasks import router as tasks_router
from transcribo_backend.routers.transcription import router as transcription_router
from transcribo_backend.routers.summarization import router as summarization_router

__all__ = [
    "tasks_router",
    "transcription_router",
    "summarization_router",
]
