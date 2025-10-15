FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client script and class names file
COPY client.py .
COPY class_names.json .

# Copy the web application
COPY webapp/ webapp/

# Set environment variables with defaults
ENV MODEL_SERVER_URL=http://host.docker.internal:8000
ENV MAX_STARTUP_WAIT=300
ENV HEALTH_CHECK_INTERVAL=10

# Expose port for web app
EXPOSE 8080

# Run the web application by default
CMD ["python", "webapp/app.py"]
