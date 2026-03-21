from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from src.api.routes import scoring, rag
from src.db.database import engine
from src.db import models

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Credit Scoring API Gateway",
    description="Data Pipeline entry point for the Credit Scoring ML Models.",
    version="1.0.0"
)

# Custom validation exception handler (Dead Letter Queue entry point for malformed payloads)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Catches 422 errors and logs them. 
    In production, push `request.body` to Dead Letter Queue (DLQ).
    """
    body = await request.body()
    # Log only safe metadata — never the raw body which may contain PII (PDPA/GDPR).
    logger.error(
        "[DLQ APPEND] Malformed payload | path=%s content_length=%d errors=%s",
        request.url.path,
        len(body),
        exc.errors(),
    )
    
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Failed. See logs or DLQ.", "errors": exc.errors()},
    )

# Include the main scoring router
app.include_router(scoring.router, prefix="/api/v1", tags=["Decisioning"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Credit Scoring API Gateway"}


@app.get("/health/deep")
async def health_deep():
    """Deep health check — verifies external dependencies."""
    from config.settings import Settings
    checks: dict = {}

    # ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=Settings.CHROMA_PERSIST_DIR)
        client.get_collection(Settings.CHROMA_COLLECTION)
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"

    # SQLite
    try:
        from sqlalchemy import text
        from src.db.database import SessionLocal
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        checks["sqlite"] = "ok"
    except Exception as e:
        checks["sqlite"] = f"error: {e}"

    # Gemini API
    if Settings.USE_GEMINI:
        try:
            import httpx
            resp = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={Settings.GEMINI_API_KEY}",
                timeout=5.0,
            )
            checks["gemini"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
        except Exception as e:
            checks["gemini"] = f"error: {e}"

    # Ollama (only if enabled)
    if Settings.USE_OLLAMA:
        try:
            import httpx
            resp = httpx.get(f"{Settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            checks["ollama"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
        except Exception as e:
            checks["ollama"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )
