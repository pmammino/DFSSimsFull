# Worker service image — light numeric API.
# Build context is the REPO ROOT (so the service can import the repo modules):
#   docker build -f service/Dockerfile -t dfs-worker .
FROM python:3.11-slim

WORKDIR /app

# System libs some scientific wheels expect at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install light API deps first for layer caching.
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

# Copy the repo (numeric modules + service). .dockerignore keeps the large
# .npy/.csv artifacts OUT of the image — they are pulled from R2 at runtime
# (or bind-mounted in dev via docker-compose).
COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn service.main:app --host 0.0.0.0 --port ${PORT}"]
