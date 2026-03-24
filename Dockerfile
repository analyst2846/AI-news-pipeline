# ============================================================
# Dockerfile for Quebec City News Intelligence Pipeline
# Designed for Cloud Run Jobs (not Cloud Run services)
# ============================================================
FROM python:3.11-slim

# Install system dependencies for Playwright + Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    libxshmfence1 \
    libglu1-mesa \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# 1) CPU-only torch first (saves ~1.8GB vs full CUDA bundle)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# 2) Everything else from requirements.txt (torch already satisfied, won't re-download)
RUN pip install --no-cache-dir -r requirements.txt
# 3) Explicit pyarrow install as safety net (in case of silent failure above)
RUN pip install --no-cache-dir pyarrow

# Install Playwright Chromium browser
RUN playwright install chromium
RUN playwright install-deps chromium

# Pre-download HuggingFace models at build time (cached in image)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline; \
AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment'); \
AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment'); \
pipeline('text-classification', model='classla/multilingual-IPTC-news-topic-classifier'); \
AutoTokenizer.from_pretrained('knowledgator/gliclass-x-base'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3'); \
"

RUN python -c "\
from gliclass import GLiClassModel; \
GLiClassModel.from_pretrained('knowledgator/gliclass-x-base'); \
"

# Force offline mode so transformers never calls HuggingFace API at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application code LAST (so code changes don't bust model cache)
COPY main.py .

# Environment variables (override at deployment)
ENV GCP_PROJECT_ID="---"
ENV BQ_DATASET_ID="---"
ENV BQ_TABLE_ID="---"
ENV GCP_LOCATION="us-central1"
ENV GEMINI_LOCATION="global"
ENV GEMINI_MODEL="gemini-3-flash-preview"

# Run the pipeline
CMD ["python", "main.py"]
