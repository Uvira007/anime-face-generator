# Anime Face Generator API - Docker Configuration
# 
# This Dockerfile creates a lightweight container for serving
# the DCGAN model via FastAPI.
#
# Build: docker build -t anime-face-api .
# Run:   docker run -p 8000:8000 -v ./models:/app/models anime-face-api

# Use Python slim image for smaller footprint
FROM python:3.10-slim@sha256:a1f1e26a3a27caadd8f4c3a4b75f6b28b02f07e2f1e8c1a8a8b9c9d9e9f9a9a

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/generator.onnx \
    LATENT_DIM=100 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY inference.py .
COPY main.py .

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

