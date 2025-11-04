# Felix Framework - Air-Gapped Multi-Agent AI System
# Production-ready Docker image

FROM python:3.10-slim

LABEL maintainer="Felix Framework"
LABEL description="Multi-agent AI framework for air-gapped environments"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    data \
    logs \
    knowledge_sources \
    results

# Set environment variables
ENV PYTHONPATH=/app
ENV FELIX_DATA_DIR=/app/data
ENV FELIX_LOG_DIR=/app/logs

# Expose ports
EXPOSE 8000  

# API server
EXPOSE 5000  # GUI (if running web version)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (API server)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
