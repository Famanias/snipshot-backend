FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

# Create models directory
RUN mkdir -p /app/models

# Copy application code (excluding models/)
COPY snipshot_engine/ ./snipshot_engine/
COPY fonts/ ./fonts/
COPY main.py .
COPY startup.sh .

# Make startup script executable
RUN chmod +x /app/startup.sh

# Expose port
EXPOSE 8001

# Run startup script
CMD ["/app/startup.sh"]
