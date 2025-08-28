FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the sync script
COPY model_sync.py .

# Create non-root user for security
RUN groupadd -r syncuser && useradd -r -g syncuser syncuser
RUN chown -R syncuser:syncuser /app

# Switch to non-root user
USER syncuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the sync script
CMD ["python", "model_sync.py"]
