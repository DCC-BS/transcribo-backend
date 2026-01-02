# Stage 1: Builder
FROM python:3.14-alpine AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache gcc musl-dev

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --locked: Sync with lockfile
# --no-dev: Exclude development dependencies
# --no-install-project: Install dependencies only (caching layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy application code
COPY . /app

# Sync project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Stage 2: Runtime
FROM python:3.14-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install packages with
RUN apk add --no-cache --allow-untrusted ffmpeg bash || \
    (apk update && apk add --no-cache ffmpeg bash)

# Create non-root user (Alpine syntax)
RUN addgroup -S app && adduser -S app -G app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app /app

RUN chmod +x /app/run.sh
RUN chown app:app /app/run.sh

# Enable virtual environment
ENV PATH="/app/.venv/bin:$PATH"

USER app

ENV ENVIRONMENT=production

ENTRYPOINT ["/app/run.sh"]
