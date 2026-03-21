FROM python:3.12-slim

# Build dependencies for faiss-cpu, numpy, chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libopenblas-dev curl \
    && rm -rf /var/lib/apt/lists/*

# UTF-8 locale for Thai text
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY config/ config/
COPY src/ src/
COPY data/documents/ data/documents/
RUN pip install --no-cache-dir -e .

# Pre-download BGE-M3 embeddings to avoid cold-start
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

# Create storage directories
RUN mkdir -p storage/chroma data

# Non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
