from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_seconds = window_seconds
        self.clock = clock
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        with self.lock:
            events = self.events[key]
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if self.limit <= 0 or len(events) >= self.limit:
                retry_after = max(1, math.ceil(events[0] + self.window_seconds - now)) if events else max(1, math.ceil(self.window_seconds))
                return False, retry_after
            events.append(now)
            return True, 0


class WriteRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: SlidingWindowLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        protected = request.url.path.startswith("/api/v1/") and request.method in {"POST", "PUT", "PATCH", "DELETE"}
        if protected:
            client = request.client.host if request.client else "local"
            allowed, retry_after = self.limiter.check(client)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "写请求过于频繁，请稍后重试"},
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)
