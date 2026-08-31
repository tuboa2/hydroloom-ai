# ==============================================================================
# Multi-Stage Production Dockerfile for Hydroloom (Render Free-Tier Optimized)
# ==============================================================================

# --- Stage 1: Build Dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv for fast deterministic builds
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

# Generate virtual environment with production dependencies using system Python
RUN python -m venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache fastapi uvicorn pydantic psutil numpy polars scikit-learn huggingface-hub joblib

# --- Stage 2: Production Runtime ---
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    ENVIRONMENT=production \
    HYDROLOOM_CACHE_DIR=/tmp/hydroloom_cache

# Create unprivileged user for security hardening
RUN addgroup --gid 10001 appgroup && \
    adduser --uid 10001 --gid 10001 --disabled-password --gecos "" appuser && \
    mkdir -p /tmp/hydroloom_cache && \
    chown -R appuser:appgroup /app /tmp/hydroloom_cache

# Copy virtualenv from builder with correct permissions
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
RUN chmod -R a+rx /opt/venv

# Copy application source code
COPY --chown=appuser:appgroup services /app/services
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml

# Switch to non-root user
USER 10001

EXPOSE 8000

# Health check using Python stdlib to avoid needing curl/wget
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Launch FastAPI application dynamically binding to Render's $PORT
CMD ["sh", "-c", "uvicorn services.wqi_predictor.deployment.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
