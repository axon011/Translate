# ==============================================================================
# Stage 1: Builder - install Python dependencies
# ==============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /opt/conda

# Copy application code (includes static/ for web dashboard)
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .
COPY requirements.txt .

# HuggingFace cache directory (mount as volume for persistence)
ENV HF_HOME=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Create cache directory
RUN mkdir -p /app/hf_cache /app/models

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
