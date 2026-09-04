# syntax=docker/dockerfile:1.7

FROM python:3.11.15-slim-bookworm

# -----------------------------------------------------------------------------
# Runtime environment
# -----------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# -----------------------------------------------------------------------------
# uv
# -----------------------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------

WORKDIR /workspace

# Install dependencies first.
# Keeping this layer separate allows Docker to reuse the dependency cache
# when only source code changes.
COPY pyproject.toml uv.lock .python-version README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Install the project itself.
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["bash"]
