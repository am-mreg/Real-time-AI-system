# Dockerfile - minimal image for running the FastAPI app
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by OpenCV / ffmpeg and common libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libsm6 \
      libxext6 \
      build-essential \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the app. Use the module path so uvicorn reload works in dev.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
