# check=skip=SecretsUsedInArgOrEnv
# The build stage runs on the shared mise base image (ghcr.io/dcc-bs/dcc-docker-images/mise)
# which provides the toolchain and the `assemble-runtime` script. Python is managed by
# uv itself (pinned by `requires-python` in pyproject.toml). The runtime image only carries
# python + varlock, so the python version lives in exactly one place: pyproject.toml.

# Stage 1: Build the application
FROM ghcr.io/dcc-bs/dcc-docker-images/mise:13-slim AS build

# Optional GitHub token to raise the API rate limit when mise resolves the
# `github:dmno-dev/varlock` release. Unauthenticated builds are limited to 60
# requests/hour per IP, which can break fresh builds/CI. Pass it with:
#   docker build --build-arg GITHUB_TOKEN=<token> .
# A fine-grained token with no scopes is sufficient. Omit for local builds.
ARG GITHUB_TOKEN=""
ENV GITHUB_TOKEN="${GITHUB_TOKEN}"

ENV APP_MODE=build
ENV DOCKER_BUILD=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=120
# uv installs its own python here; a stable path so the runtime copy below
# never hardcodes the python version or architecture.
ENV UV_PYTHON_INSTALL_DIR="/uv-python"

# Set the working directory
WORKDIR /app

# Copy source code (the `install` task's `uv sync` installs the project
# editable, so source must be present before `mise install` runs)
COPY . .

# Install the pinned toolchain (uv, varlock) from mise.toml. The postinstall
# hook runs the `install` task, which does `uv sync --locked --no-dev` (because
# DOCKER_BUILD=1) and auto-installs the python pinned by `requires-python`.
RUN mise trust -a && mise install

# Assemble a minimal runtime: only python + varlock (drop mise, uv, pass-cli,
# usage, rust and python headers/tcl-tk/terminfo). Shared logic from the base image.
RUN assemble-runtime python

# Stage 2: Run the application
# ------------------------------------------------
FROM debian:13-slim

# Set the working directory
WORKDIR /app

# Security: Create and switch to a non-root user
RUN useradd --create-home --uid 1000 app

# Environment
ENV APP_MODE=prod
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Runtime python is the one assembled from uv in the build stage
ENV PATH="/runtime/varlock:/app/.venv/bin:$PATH"

# Copy the built application and the minimal runtime (python + varlock) from
# the build stage
COPY --from=build --chown=app:app /app /app
COPY --from=build --chown=app:app /runtime /runtime
COPY --chown=app:app .env*.schema /app/

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# Start the application: load env via varlock, then run uvicorn with the runtime python
ENTRYPOINT ["/bin/sh", "-c", "varlock load && varlock run -- uvicorn transcribo_backend.app:app --host 0.0.0.0 --port \"${PORT:-8090}\" --no-access-log"]
