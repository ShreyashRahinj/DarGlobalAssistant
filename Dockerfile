FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build so startup is fast and offline.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" && \
    chmod -R a+rX /tmp/fastembed_cache

COPY app ./app
COPY scraper ./scraper
COPY data ./data

# Small free-tier instances: keep ONNX from spawning a thread per host core
# (spinning threads inflated startup to >1 GB RAM and ~17x CPU).
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Bake the retrieval index into the image so a restart/cold start loads it
# instead of re-embedding 303 chunks on a fractional CPU.
RUN python -c "from pathlib import Path; from app.rag import Retriever; r = Retriever(index_dir=Path('data/index')); n = r.build_from_jsonl(Path('data/pages.jsonl')); assert n > 0, 'index build produced no chunks'"

RUN groupadd -r appuser -g 1000 && useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app /tmp/fastembed_cache
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
