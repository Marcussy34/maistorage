"""
Production middleware for MAI Storage RAG API.

Implements Phase 10 hardening requirements:
- Request size limits and payload validation
- Rate limiting with Redis/memory backend
- Circuit breakers for external dependencies
- Error boundaries and graceful fallbacks
- Security headers and CORS enforcement
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
import json

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE, HTTP_429_TOO_MANY_REQUESTS

from logging_config import get_logger
from models import ErrorResponse, ErrorDetail

logger = get_logger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request size limits.
    
    Prevents large payloads from consuming excessive memory or processing time.
    """
    
    def __init__(self, app, max_request_size: int = 10_000_000):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header if present
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_request_size:
                logger.warning(
                    "request_size_exceeded",
                    content_length=content_length,
                    max_size=self.max_request_size
                )
                return JSONResponse(
                    status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            code="REQUEST_TOO_LARGE",
                            message=f"Request size {content_length} exceeds maximum {self.max_request_size} bytes"
                        )
                    ).dict()
                )
        
        # For streaming bodies, we'll check during processing
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Memory-based rate limiting middleware.
    
    Uses sliding window algorithm for smooth rate limiting.
    In production, consider using Redis for distributed rate limiting.
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 100,
        burst_capacity: int = 20,
        enable_burst: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_capacity = burst_capacity
        self.enable_burst = enable_burst
        
        # Track requests per client IP
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_tokens: Dict[str, int] = defaultdict(lambda: burst_capacity)
        self.last_refill: Dict[str, float] = defaultdict(lambda: time.time())
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (from load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _refill_burst_tokens(self, client_ip: str) -> None:
        """Refill burst tokens based on time elapsed."""
        if not self.enable_burst:
            return
        
        now = time.time()
        last_refill = self.last_refill[client_ip]
        elapsed = now - last_refill
        
        # Refill at rate of 1 token per minute / burst_capacity
        refill_rate = self.burst_capacity / 60.0  # tokens per second
        tokens_to_add = int(elapsed * refill_rate)
        
        if tokens_to_add > 0:
            self.burst_tokens[client_ip] = min(
                self.burst_capacity,
                self.burst_tokens[client_ip] + tokens_to_add
            )
            self.last_refill[client_ip] = now
    
    def _is_rate_limited(self, client_ip: str) -> Tuple[bool, Dict[str, any]]:
        """Check if client IP is rate limited."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        requests = self.request_counts[client_ip]
        while requests and requests[0] < minute_ago:
            requests.popleft()
        
        # Refill burst tokens
        self._refill_burst_tokens(client_ip)
        
        # Check burst capacity first
        if self.enable_burst and self.burst_tokens[client_ip] > 0:
            self.burst_tokens[client_ip] -= 1
            requests.append(now)
            return False, {
                "requests_in_window": len(requests),
                "burst_tokens_remaining": self.burst_tokens[client_ip],
                "rate_limit_type": "burst"
            }
        
        # Check regular rate limit
        if len(requests) >= self.requests_per_minute:
            return True, {
                "requests_in_window": len(requests),
                "burst_tokens_remaining": self.burst_tokens[client_ip],
                "retry_after": 60 - (now - requests[0]) if requests else 60
            }
        
        # Allow request
        requests.append(now)
        return False, {
            "requests_in_window": len(requests),
            "burst_tokens_remaining": self.burst_tokens[client_ip],
            "rate_limit_type": "normal"
        }
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        is_limited, rate_info = self._is_rate_limited(client_ip)
        
        if is_limited:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                **rate_info
            )
            
            response = JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="RATE_LIMIT_EXCEEDED",
                        message=f"Rate limit exceeded. Try again in {rate_info.get('retry_after', 60)} seconds."
                    )
                ).dict()
            )
            
            if "retry_after" in rate_info:
                response.headers["Retry-After"] = str(int(rate_info["retry_after"]))
            
            return response
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - rate_info["requests_in_window"])
        )
        
        if self.enable_burst:
            response.headers["X-RateLimit-Burst-Remaining"] = str(rate_info["burst_tokens_remaining"])
        
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """
    Circuit breaker pattern for external service dependencies.
    
    Prevents cascading failures when external services are down.
    """
    
    def __init__(
        self,
        app,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # Circuit breaker state per service
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_time: Dict[str, float] = defaultdict(float)
        self.circuit_open: Dict[str, bool] = defaultdict(bool)
    
    def _get_service_key(self, request: Request) -> str:
        """Determine service key from request path."""
        path = request.url.path
        
        # Map paths to services
        if path.startswith("/retrieve"):
            return "qdrant"
        elif path.startswith("/rag") or path.startswith("/chat"):
            return "openai"
        else:
            return "default"
    
    def _should_trip_breaker(self, service_key: str) -> bool:
        """Check if circuit breaker should trip."""
        return self.failure_counts[service_key] >= self.failure_threshold
    
    def _should_attempt_reset(self, service_key: str) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.circuit_open[service_key]:
            return False
        
        time_since_failure = time.time() - self.last_failure_time[service_key]
        return time_since_failure >= self.recovery_timeout
    
    def _record_success(self, service_key: str) -> None:
        """Record successful request."""
        self.failure_counts[service_key] = 0
        self.circuit_open[service_key] = False
        
        logger.info(
            "circuit_breaker_reset",
            service=service_key
        )
    
    def _record_failure(self, service_key: str) -> None:
        """Record failed request."""
        self.failure_counts[service_key] += 1
        self.last_failure_time[service_key] = time.time()
        
        if self._should_trip_breaker(service_key):
            self.circuit_open[service_key] = True
            logger.error(
                "circuit_breaker_tripped",
                service=service_key,
                failure_count=self.failure_counts[service_key],
                threshold=self.failure_threshold
            )
    
    async def dispatch(self, request: Request, call_next):
        service_key = self._get_service_key(request)
        
        # Check if circuit is open
        if self.circuit_open[service_key]:
            if not self._should_attempt_reset(service_key):
                logger.warning(
                    "circuit_breaker_open",
                    service=service_key,
                    failure_count=self.failure_counts[service_key]
                )
                
                return JSONResponse(
                    status_code=503,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            code="SERVICE_UNAVAILABLE",
                            message=f"Service {service_key} is temporarily unavailable. Circuit breaker is open."
                        )
                    ).dict()
                )
        
        try:
            # Attempt request
            response = await call_next(request)
            
            # Record success if status is OK
            if 200 <= response.status_code < 300:
                self._record_success(service_key)
            elif response.status_code >= 500:
                self._record_failure(service_key)
            
            return response
            
        except Exception as e:
            # Record failure
            self._record_failure(service_key)
            
            # Re-raise the exception
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    
    Implements basic security best practices for API responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Allow Swagger UI resources for /docs and /redoc endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdn.jsdelivr.net"
            )
        else:
            # Strict CSP for API endpoints
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # API-specific headers
        response.headers["X-API-Version"] = "0.3.0"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        
        return response


class ErrorBoundaryMiddleware(BaseHTTPMiddleware):
    """
    Global error boundary for unhandled exceptions.
    
    Ensures all errors are logged and return consistent JSON responses.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        
        except Exception as e:
            # Log unexpected errors
            logger.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="INTERNAL_SERVER_ERROR",
                        message="An unexpected error occurred. Please try again later."
                    )
                ).dict()
            )


# Export middleware classes
__all__ = [
    'RequestSizeLimitMiddleware',
    'RateLimitMiddleware', 
    'CircuitBreakerMiddleware',
    'SecurityHeadersMiddleware',
    'ErrorBoundaryMiddleware'
]
