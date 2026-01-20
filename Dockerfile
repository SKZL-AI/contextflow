# ContextFlow AI - Docker Image
# Multi-stage build for optimal image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies)
RUN poetry install --only=main --no-root

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install the package
RUN poetry install --only=main

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash contextflow

# Set working directory
WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

# Copy additional files
COPY README.md ./
COPY .env.example ./.env.example

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs \
    && chown -R contextflow:contextflow /app

# Switch to non-root user
USER contextflow

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Default command: Start API server
CMD ["python", "-m", "uvicorn", "contextflow.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Labels
# =============================================================================
LABEL org.opencontainers.image.title="ContextFlow AI" \
      org.opencontainers.image.description="Intelligent LLM Context Orchestration Framework" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="ContextFlow" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/contextflow/contextflow"
