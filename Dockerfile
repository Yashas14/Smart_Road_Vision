# ============================================================================
# SmartRoadVision — Multi-stage Dockerfile
# Stage 1: builder  -> resolve & install Python deps into a venv
# Stage 2: runtime  -> slim image carrying only the venv + source
# ============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps required to build wheels (opencv, torch, psycopg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

# Install the project (runtime deps only) into the venv
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Minimal runtime system libraries for OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Bring the prepared virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Application source
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# Default: production API server (gunicorn managing uvicorn workers)
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
