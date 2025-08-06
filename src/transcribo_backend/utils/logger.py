import logging
import os
import time
import uuid
from typing import Any, Mapping, Optional

import structlog
import structlog.processors
from structlog.processors import CallsiteParameter
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, Processor


# Standard library logging setup
def setup_stdlib_logging() -> None:
    """Configure standard library logging to work with structlog."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    # Create a handler for console output
    handler = logging.StreamHandler()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Disable propagation for libraries that are too verbose
    for logger_name in ["uvicorn.access"]:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.propagate = False


def add_request_id(logger: BoundLogger, method_name: str, event_dict: EventDict) -> Mapping[str, Any]:
    """
    Add a request ID to the log context if it doesn't exist.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary

    Returns:
        The updated event dictionary
    """
    if "request_id" not in event_dict:
        event_dict["request_id"] = str(uuid.uuid4())
    return event_dict


def add_timestamp(logger: BoundLogger, method_name: str, event_dict: EventDict) -> Mapping[str, Any]:
    """
    Add an ISO-8601 timestamp to the log entry.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary

    Returns:
        The updated event dictionary
    """
    event_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return event_dict


def init_logger() -> None:
    """
    Initialize the logger configuration based on environment.
    Uses JSON renderer in production environment for compatibility with fluentbit.
    Adds the module name as context to the logger.
    """
    # Set up standard library logging first
    setup_stdlib_logging()

    # Define processors list for structlog
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,  # Filter logs by configured level
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        add_timestamp,
        add_request_id,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                CallsiteParameter.MODULE,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]  # Using enum values for structlog 25.2.0
        ),
        # structlog.processors.format_exc_info,  # Format exception info if present
        structlog.processors.UnicodeDecoder(),  # Handle non-unicode characters
    ]

    # Use different renderers for development vs production
    if os.getenv("PROD"):
        # JSON renderer for production to be fluentbit compatible
        processors.append(structlog.processors.JSONRenderer())
    else:
        # For development, use a colored console renderer
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional name for the logger, typically the module name

    Returns:
        A bound logger instance for structured logging
    """
    if name:
        return structlog.get_logger(name)  # type: ignore
    return structlog.get_logger()  # type: ignore
