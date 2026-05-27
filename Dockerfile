FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (no MySQL needed)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    transformers>=4.41.0 \
    sentencepiece>=0.1.99 \
    sentence-transformers>=2.2.2 \
    faiss-cpu>=1.7.4 \
    rank-bm25>=0.2.2 \
    rapidfuzz>=3.0.0 \
    lightgbm>=4.0.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    flask>=3.0.0 \
    requests>=2.31.0 \
    indic-transliteration>=2.3.0

# Copy project files
COPY . .

# Disable SQL — use static index only
ENV V2_USE_SQL_DB=0

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["python", "8_api.py", "--v2", "--host", "0.0.0.0", "--port", "7860"]
