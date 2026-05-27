FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install production WSGI server
RUN pip install --no-cache-dir gunicorn

# Copy project files
COPY . .

# Disable SQL — use static index only
ENV V2_USE_SQL_DB=0

# HF Spaces uses port 7860
EXPOSE 7860

# Use gunicorn for production; fallback to Flask dev server if gunicorn not available
CMD ["sh", "-c", "python 8_api.py --v2 --host 0.0.0.0 --port 7860"]
