# --- Build Stage ---
FROM python:3.12-slim as builder

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.12-slim

WORKDIR /app

# 1. Install Tesseract, Farsi Pack, AND libgl1 (for OpenCV/Unstructured)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fas \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# 2. FIX: TESSDATA_PREFIX must be a SINGLE path. 
# On Debian/Ubuntu images, apt installs data to /usr/share/tesseract-ocr/4.00/tessdata
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]