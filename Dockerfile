FROM python:3.11-slim

# Install system dependencies for Tesseract OCR and PDF handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application source
COPY . .

# Set environment defaults
ENV PORT=5000 \
    FLASK_DEBUG=false \
    PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=tesseract \
    OCR_LANG=eng

EXPOSE 5000

# Run with Gunicorn production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "wsgi:app"]
