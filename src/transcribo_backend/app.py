from backend_common.fastapi_error_handling import inject_api_error_handler
from backend_common.fastapi_health_probes import health_probe_router
from backend_common.fastapi_health_probes.router import ServiceDependency
from fastapi import FastAPI

from transcribo_backend.config import settings
from transcribo_backend.routers import (
    summarization_router,
    tasks_router,
    transcription_router,
)
from transcribo_backend.utils.logger import init_logger

init_logger()

probe_dependencies: list[ServiceDependency] = [
    {
        "name": "llm_api",
        "health_check_url": settings.llm_health_check,
        "api_key": settings.api_key,
    },
    {
        "name": "whisper_api",
        "health_check_url": settings.whisper_health_check,
        "api_key": settings.api_key,
    },
]

# Initialize FastAPI app
app = FastAPI()

# Include health probes
app.include_router(health_probe_router(probe_dependencies))

# Include API routers
app.include_router(tasks_router)
app.include_router(transcription_router)
app.include_router(summarization_router)

# Inject error handler
inject_api_error_handler(app)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("transcribo_backend.app:app", host="127.0.0.1", port=8000, reload=True)
