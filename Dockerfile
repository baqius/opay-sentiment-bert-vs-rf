FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data needed by preprocessor()
RUN python -m nltk.downloader stopwords wordnet -d /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

# Copy application code and model artifacts
COPY fast_app.py preprocessing.py ./
COPY rf_sentiment_model.pkl tfidf_vectorizer.pkl label_encoder.pkl ./

# Run as a non-root user for better container security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "fast_app:app", "--host", "0.0.0.0", "--port", "8000"]
