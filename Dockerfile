#Base Image with Python
FROM python:3.12-slim AS base

WORKDIR /app

# Set environment variables
# PYTHONUNBUFFERED: Force Python to print output immediately (important for logs)
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files (saves space)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies Installation
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Training Image
# ---------------------------
FROM base AS training

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/runs \
    /app/src

COPY . .

CMD ["python", "./src/training/train.py"]

# ---------------------------
# Inference Image
# ---------------------------
FROM base AS inference

EXPOSE 8000

COPY runs /app/runs/
COPY src /app/src/
COPY api /app/api/


CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]