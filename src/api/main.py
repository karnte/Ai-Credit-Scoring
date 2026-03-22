from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from src.api.routes import scoring, rag, predict
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
app.include_router(predict.router, prefix="/api/v1", tags=["Frontend"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Credit Scoring API Gateway"}
