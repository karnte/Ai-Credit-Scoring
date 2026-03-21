"""
Simple in-memory rate limiting middleware for FastAPI.
No external dependencies required.

Usage in src/api/main.py:
    from deploy.rate_limit import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware, max_requests=30, window_seconds=60)
"""

import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _clean(self, key: str, now: float):
        cutoff = now - self.window_seconds
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Scoring endpoints get a stricter limit
        path = request.url.path
        if "/score" in path or "/plan" in path:
            limit = min(self.max_requests, 10)
            key = f"{client_ip}:scoring"
        else:
            limit = self.max_requests
            key = client_ip

        self._clean(key, now)

        if len(self._hits[key]) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        self._hits[key].append(now)
        return await call_next(request)
