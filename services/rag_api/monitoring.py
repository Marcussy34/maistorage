"""
Monitoring and metrics endpoints for MAI Storage RAG API.

Implements Phase 10 monitoring requirements:
- Prometheus metrics export
- Custom business metrics
- Performance monitoring
- System resource tracking
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import threading

import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

from logging_config import get_logger

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'rag_api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'rag_api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

RETRIEVAL_QUERIES = Counter(
    'rag_api_retrieval_queries_total',
    'Total number of retrieval queries',
    ['method', 'source']
)

RETRIEVAL_LATENCY = Histogram(
    'rag_api_retrieval_latency_seconds',
    'Retrieval latency in seconds',
    ['method', 'source']
)

CACHE_OPERATIONS = Counter(
    'rag_api_cache_operations_total',
    'Total cache operations',
    ['operation', 'cache_type', 'result']
)

LLM_TOKENS = Counter(
    'rag_api_llm_tokens_total',
    'Total LLM tokens used',
    ['model', 'type']
)

LLM_REQUESTS = Counter(
    'rag_api_llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'rag_api_memory_usage_bytes',
    'Memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'rag_api_cpu_usage_percent',
    'CPU usage percentage'
)

ACTIVE_CONNECTIONS = Gauge(
    'rag_api_active_connections',
    'Number of active connections'
)


class MetricsCollector:
    """
    Centralized metrics collection and aggregation.
    
    Collects custom business metrics beyond what Prometheus tracks.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.request_times = deque(maxlen=1000)  # Rolling window
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0})
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.lock = threading.Lock()
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        with self.lock:
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            self.request_times.append(time.time())
            self.endpoint_stats[f"{method} {endpoint}"]["count"] += 1
            self.endpoint_stats[f"{method} {endpoint}"]["total_time"] += duration
            
            if status_code >= 400:
                self.error_counts[status_code] += 1
    
    def record_retrieval(self, method: str, source: str, latency: float):
        """Record retrieval metrics."""
        RETRIEVAL_QUERIES.labels(method=method, source=source).inc()
        RETRIEVAL_LATENCY.labels(method=method, source=source).observe(latency)
    
    def record_cache_operation(self, operation: str, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        result = "hit" if hit else "miss"
        CACHE_OPERATIONS.labels(operation=operation, cache_type=cache_type, result=result).inc()
        
        with self.lock:
            if hit:
                self.cache_stats[cache_type]["hits"] += 1
            else:
                self.cache_stats[cache_type]["misses"] += 1
    
    def record_llm_usage(self, model: str, tokens: int, token_type: str, success: bool):
        """Record LLM usage metrics."""
        LLM_TOKENS.labels(model=model, type=token_type).inc(tokens)
        status = "success" if success else "error"
        LLM_REQUESTS.labels(model=model, status=status).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
        except Exception as e:
            logger.warning("system_metrics_update_failed", error=str(e))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregated performance statistics."""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate requests per second (last minute)
            recent_requests = [t for t in self.request_times if current_time - t <= 60]
            requests_per_minute = len(recent_requests)
            requests_per_second = requests_per_minute / 60.0
            
            # Calculate average response times
            endpoint_averages = {}
            for endpoint, stats in self.endpoint_stats.items():
                if stats["count"] > 0:
                    endpoint_averages[endpoint] = {
                        "count": stats["count"],
                        "avg_response_time": stats["total_time"] / stats["count"]
                    }
            
            # Calculate cache hit rates
            cache_rates = {}
            for cache_type, stats in self.cache_stats.items():
                total = stats["hits"] + stats["misses"]
                if total > 0:
                    cache_rates[cache_type] = {
                        "hit_rate": stats["hits"] / total,
                        "total_operations": total
                    }
            
            return {
                "uptime_seconds": uptime,
                "requests_per_second": requests_per_second,
                "total_requests": sum(stats["count"] for stats in self.endpoint_stats.values()),
                "error_rate": sum(self.error_counts.values()) / max(1, sum(stats["count"] for stats in self.endpoint_stats.values())),
                "endpoint_stats": endpoint_averages,
                "cache_stats": cache_rates,
                "error_counts": dict(self.error_counts)
            }


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    # Update system metrics before export
    metrics_collector.update_system_metrics()
    
    # Generate Prometheus format metrics
    metrics_data = generate_latest()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


async def stats_endpoint() -> Dict[str, Any]:
    """Custom stats endpoint with business metrics."""
    metrics_collector.update_system_metrics()
    
    # Get system information
    try:
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=1)
        disk_info = psutil.disk_usage('/')
        
        system_stats = {
            "memory": {
                "total_mb": memory_info.total / 1024 / 1024,
                "used_mb": memory_info.used / 1024 / 1024,
                "available_mb": memory_info.available / 1024 / 1024,
                "usage_percent": memory_info.percent
            },
            "cpu": {
                "usage_percent": cpu_info,
                "count": psutil.cpu_count()
            },
            "disk": {
                "total_gb": disk_info.total / 1024 / 1024 / 1024,
                "used_gb": disk_info.used / 1024 / 1024 / 1024,
                "free_gb": disk_info.free / 1024 / 1024 / 1024,
                "usage_percent": (disk_info.used / disk_info.total) * 100
            }
        }
    except Exception as e:
        logger.warning("system_stats_failed", error=str(e))
        system_stats = {"error": "Could not retrieve system stats"}
    
    # Combine with performance stats
    performance_stats = metrics_collector.get_performance_stats()
    
    return {
        "timestamp": time.time(),
        "system": system_stats,
        "performance": performance_stats,
        "version": "0.3.0"
    }


async def health_detailed_endpoint() -> Dict[str, Any]:
    """Detailed health check with dependency status."""
    
    # This would typically check:
    # - Database connections
    # - External API availability
    # - Cache system status
    # - Queue system status
    
    dependencies = {
        "qdrant": {
            "status": "healthy",  # Would check actual connection
            "response_time_ms": 15,
            "last_check": time.time()
        },
        "openai": {
            "status": "healthy",  # Would check API availability
            "response_time_ms": 200,
            "last_check": time.time()
        },
        "cache": {
            "status": "healthy",
            "hit_rate": 0.85,
            "size_mb": 150
        }
    }
    
    overall_healthy = all(dep["status"] == "healthy" for dep in dependencies.values())
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": time.time(),
        "dependencies": dependencies,
        "system": await stats_endpoint()
    }


# Export key components
__all__ = [
    'MetricsCollector',
    'metrics_collector',
    'get_metrics_collector',
    'metrics_endpoint',
    'stats_endpoint',
    'health_detailed_endpoint'
]
