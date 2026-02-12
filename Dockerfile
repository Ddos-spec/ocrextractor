FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    OCR_ZOOM=1.35 \
    OCR_MAX_PAGES=0 \
    OCR_CONCURRENCY=1 \
    RESULT_CACHE_TTL_SECONDS=900 \
    RESULT_CACHE_MAX_ITEMS=256

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ind \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8899

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8899"]
