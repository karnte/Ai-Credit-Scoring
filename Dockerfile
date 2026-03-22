FROM python:3.12-slim

# UTF-8 locale for Thai text
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install build deps in smaller steps to reduce peak memory
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids 4GB+ NVIDIA CUDA packages)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==1.26.4 \
    && pip install --no-cache-dir faiss-cpu==1.8.0 \
    && pip install --no-cache-dir chromadb \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY config/ config/
COPY src/ src/
COPY data/documents/ data/documents/
RUN pip install --no-cache-dir --no-deps -e .

# BGE-M3 downloads at runtime into the huggingface_cache volume (saves ~2.4GB image size)

# Create storage directories
RUN mkdir -p storage/chroma data

# Non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
