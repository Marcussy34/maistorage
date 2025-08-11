"""
Advanced logging configuration for MAI Storage RAG API.

Implements Phase 10 structured logging requirements:
- Request correlation IDs
- Performance metrics integration
- Structured JSON logging for production
- Multiple log levels and handlers
- Integration with tracing systems
"""

import json
import logging
import logging.config
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject request context into logging.
    
    Adds request ID, trace ID, and timing information to all logs
    within the request scope.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Extract trace ID from headers if present
        trace_id = request.headers.get('x-trace-id') or str(uuid.uuid4())
        trace_id_var.set(trace_id)
        
        # Extract user ID from headers if present
        user_id = request.headers.get('x-user-id')
        user_id_var.set(user_id)
        
        # Record start time
        start_time = time.time()
        
        # Log request start
        logger = structlog.get_logger()
        logger.info(
            "request_started",
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get('user-agent'),
            client_ip=request.client.host if request.client else None
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2)
            )
            
            # Add headers for client debugging
            response.headers["x-request-id"] = request_id
            response.headers["x-trace-id"] = trace_id
            response.headers["x-response-time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            # Calculate duration for error case
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request error
            logger.error(
                "request_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=round(duration_ms, 2),
                exc_info=True
            )
            
            raise


def add_request_context(logger, method_name, event_dict):
    """
    Add request context to all log entries.
    
    This processor adds request ID, trace ID, and user context
    to every log message within a request scope.
    """
    event_dict["request_id"] = request_id_var.get()
    event_dict["trace_id"] = trace_id_var.get()
    event_dict["user_id"] = user_id_var.get()
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    
    return event_dict


def add_service_context(logger, method_name, event_dict):
    """Add service-level context to all log entries."""
    event_dict["service"] = "rag_api"
    event_dict["version"] = "0.3.0"  # Phase 10 version
    event_dict["environment"] = "development"  # Override in production
    
    return event_dict


class PerformanceLogger:
    """
    Context manager for performance logging.
    
    Usage:
        with PerformanceLogger("embedding_generation"):
            result = await generate_embeddings(text)
    """
    
    def __init__(self, operation_name: str, **kwargs):
        self.operation_name = operation_name
        self.context = kwargs
        self.start_time = None
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            "operation_started",
            operation=self.operation_name,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                "operation_completed",
                operation=self.operation_name,
                duration_ms=round(duration_ms, 2),
                **self.context
            )
        else:
            self.logger.error(
                "operation_failed",
                operation=self.operation_name,
                duration_ms=round(duration_ms, 2),
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = False,
    enable_tracing: bool = False
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output logs in JSON format
        enable_tracing: Whether to enable distributed tracing
    """
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        add_service_context,
        add_request_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_logs:
        # JSON logging for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console logging for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    # Enable debug logging for our modules in development
    if log_level.upper() == "DEBUG":
        logging.getLogger("retrieval").setLevel(logging.DEBUG)
        logging.getLogger("graph").setLevel(logging.DEBUG)
        logging.getLogger("cache").setLevel(logging.DEBUG)


class TracingAdapter:
    """
    Adapter for integrating with external tracing systems.
    
    Supports Langfuse, Phoenix, and other OpenTelemetry-compatible systems.
    """
    
    def __init__(self, enable_langfuse: bool = False, enable_phoenix: bool = False):
        self.enable_langfuse = enable_langfuse
        self.enable_phoenix = enable_phoenix
        self.logger = structlog.get_logger()
        
        if enable_langfuse:
            self._setup_langfuse()
        
        if enable_phoenix:
            self._setup_phoenix()
    
    def _setup_langfuse(self):
        """Setup Langfuse tracing integration."""
        try:
            # Import will fail if langfuse is not installed
            import langfuse
            
            self.langfuse = langfuse.Langfuse()
            self.logger.info("langfuse_tracing_enabled")
            
        except ImportError:
            self.logger.warning("langfuse_not_available", msg="Install langfuse for tracing")
            self.enable_langfuse = False
    
    def _setup_phoenix(self):
        """Setup Phoenix tracing integration."""
        try:
            # Import will fail if phoenix is not installed
            import phoenix as px
            
            self.phoenix_session = px.launch_app()
            self.logger.info("phoenix_tracing_enabled", session_url=self.phoenix_session.url)
            
        except ImportError:
            self.logger.warning("phoenix_not_available", msg="Install phoenix for tracing")
            self.enable_phoenix = False
    
    def trace_operation(self, operation_name: str, **kwargs):
        """Create a traced operation context."""
        if self.enable_langfuse:
            return self._langfuse_trace(operation_name, **kwargs)
        else:
            return PerformanceLogger(operation_name, **kwargs)
    
    def _langfuse_trace(self, operation_name: str, **kwargs):
        """Create a Langfuse trace context."""
        if hasattr(self, 'langfuse'):
            trace = self.langfuse.trace(name=operation_name, metadata=kwargs)
            return trace
        else:
            return PerformanceLogger(operation_name, **kwargs)


# Global tracing adapter instance
tracing_adapter: Optional[TracingAdapter] = None


def get_tracing_adapter() -> TracingAdapter:
    """Get the global tracing adapter instance."""
    global tracing_adapter
    if tracing_adapter is None:
        tracing_adapter = TracingAdapter()
    return tracing_adapter


def setup_production_logging():
    """Setup logging configuration optimized for production deployment."""
    configure_logging(
        log_level="INFO",
        json_logs=True,
        enable_tracing=True
    )


def setup_development_logging():
    """Setup logging configuration optimized for development."""
    configure_logging(
        log_level="DEBUG",
        json_logs=False,
        enable_tracing=False
    )


# Convenience function for getting a logger
def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Export key classes and functions
__all__ = [
    'RequestContextMiddleware',
    'PerformanceLogger', 
    'TracingAdapter',
    'configure_logging',
    'setup_production_logging',
    'setup_development_logging',
    'get_logger',
    'get_tracing_adapter'
]
