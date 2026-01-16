from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dcc_backend_common.fastapi_error_handling import inject_api_error_handler
from dcc_backend_common.fastapi_health_probes import health_probe_router
from dcc_backend_common.fastapi_health_probes.router import ServiceDependency
from dcc_backend_common.logger import get_logger, init_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog.stdlib import BoundLogger

from transcribo_backend.container import Container
from transcribo_backend.routes import summarize_route, transcribe_route
from transcribo_backend.utils.app_config import AppConfig


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan events.
    """
    logger = get_logger("app.lifespan")
    logger.info("Checking container dependencies before startup")
    app.state.container.check_dependencies()
    logger.info("Container dependencies are healthy")

    yield


def _build_fastapi_app() -> FastAPI:
    """
    Instantiate the FastAPI application with metadata and lifespan.
    """
    app = FastAPI(
        title="Transcribo",
        description="API for transcription service with AI-powered language processing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    return app


def _register_health_routes(app: FastAPI, config: AppConfig) -> None:
    """
    Register health routes for the application.
    """
    whisper_base_url = config.whisper_url.rstrip("v1")
    llm_base_url = config.llm_base_url.rstrip("v1")
    service_dependencies: list[ServiceDependency] = [
        ServiceDependency(
            name="whisper",
            health_check_url=f"{whisper_base_url}readyz",
            api_key=config.api_key,
        ),
        ServiceDependency(
            name="llm",
            health_check_url=f"{llm_base_url}health",
            api_key=config.api_key,
        ),
    ]
    app.include_router(health_probe_router(service_dependencies=service_dependencies))


def _configure_container(app: FastAPI, logger: BoundLogger) -> Container:
    """
    Configure the dependency injection container and attach it to app state.
    """
    logger.debug("Configuring dependency injection container")
    container = Container()
    logger.info("Dependency injection configured")
    app.state.container = container
    return container


def _register_routes(app: FastAPI, logger: BoundLogger) -> None:
    """
    Register API routers.
    """
    logger.debug("Registering API routers")
    app.include_router(summarize_route.create_router())
    app.include_router(transcribe_route.create_router())
    logger.info("All routers registered")


def _configure_cors(app: FastAPI, client_url: str, logger: BoundLogger) -> None:
    """
    Apply CORS middleware configuration.
    """
    logger.debug("Setting up CORS middleware")

    app.add_middleware(
        CORSMiddleware,  # ty:ignore[invalid-argument-type]
        allow_origins=[client_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS configured with origin: {client_url}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This function initializes the FastAPI application with:
    - Environment variables loading
    - Logging configuration
    - Dependency injection container setup
    - CORS middleware configuration
    - API route registration

    Returns:
        FastAPI: Configured FastAPI application instance
    """

    init_logger()

    logger = get_logger("app")
    logger.info("Starting Transcribo API")

    app = _build_fastapi_app()

    inject_api_error_handler(app)

    container = _configure_container(app=app, logger=logger)
    config = container.app_config()
    logger.info(f"AppConfig loaded: {config}")

    _register_health_routes(app=app, config=config)

    _configure_cors(app=app, client_url=config.client_url, logger=logger)
    _register_routes(app=app, logger=logger)

    logger.info("API setup complete")
    return app


app = create_app()
