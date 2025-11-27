# Dockerfile for Urban Sound Classification API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose ports
EXPOSE 8000 8001

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.prediction:app", "--host", "0.0.0.0", "--port", "8000"]
