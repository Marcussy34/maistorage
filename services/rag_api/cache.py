"""
Phase 9: Comprehensive Caching System for Performance Optimization

This module implements multi-layer caching for:
- Query embeddings (to reduce OpenAI API calls)
- Candidate IDs (for popular queries)
- Rerank features (to speed up cross-encoder inference)
- Prompt templates (for faster prompt assembly)
- BM25 indices (for faster lexical search)
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Base cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_removals": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self.stats["expired_removals"] += 1
                self.stats["misses"] += 1
                return None
            
            entry.touch()
            self.stats["hits"] += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            now = time.time()
            ttl = ttl or self.default_ttl
            
            # Remove expired entries first
            await self._cleanup_expired()
            
            # If at capacity, remove LRU entry
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl_seconds=ttl
            )
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "expired_removals": 0}
    
    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]
            self.stats["expired_removals"] += 1
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]
        self.stats["evictions"] += 1


class EmbeddingCache:
    """Specialized cache for query embeddings."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size=max_size, default_ttl=ttl_seconds)
        self.embedding_dimension = None
    
    def _hash_query(self, query: str, model: str) -> str:
        """Create hash key for query and model combination."""
        content = f"{model}:{query.strip().lower()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get_embedding(self, query: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding for query."""
        key = self._hash_query(query, model)
        cached_embedding = await self.cache.get(key)
        
        if cached_embedding is not None:
            return np.array(cached_embedding)
        return None
    
    async def set_embedding(self, query: str, model: str, embedding: np.ndarray) -> None:
        """Cache embedding for query."""
        key = self._hash_query(query, model)
        
        # Store embedding dimension for validation
        if self.embedding_dimension is None:
            self.embedding_dimension = len(embedding)
        elif len(embedding) != self.embedding_dimension:
            logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
            return
        
        # Convert to list for JSON serialization
        await self.cache.set(key, embedding.tolist())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = await self.cache.hit_rate()
        size = await self.cache.size()
        
        return {
            "type": "embedding_cache",
            "size": size,
            "max_size": self.cache.max_size,
            "hit_rate": hit_rate,
            "embedding_dimension": self.embedding_dimension,
            **self.cache.stats
        }


class CandidateCache:
    """Cache for retrieval candidate IDs."""
    
    def __init__(self, max_size: int = 5000, ttl_seconds: int = 1800):
        self.cache = LRUCache(max_size=max_size, default_ttl=ttl_seconds)
    
    def _hash_request(self, query: str, retrieval_params: Dict[str, Any]) -> str:
        """Create hash key for retrieval request."""
        # Create reproducible hash from query and parameters
        content = {
            "query": query.strip().lower(),
            "params": sorted(retrieval_params.items())
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def get_candidates(self, 
                           query: str, 
                           retrieval_params: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
        """Get cached candidate IDs for query and parameters."""
        key = self._hash_request(query, retrieval_params)
        return await self.cache.get(key)
    
    async def set_candidates(self, 
                           query: str, 
                           retrieval_params: Dict[str, Any],
                           candidates: Dict[str, List[str]]) -> None:
        """Cache candidate IDs for query and parameters."""
        key = self._hash_request(query, retrieval_params)
        await self.cache.set(key, candidates)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = await self.cache.hit_rate()
        size = await self.cache.size()
        
        return {
            "type": "candidate_cache",
            "size": size,
            "max_size": self.cache.max_size,
            "hit_rate": hit_rate,
            **self.cache.stats
        }


class RerankCache:
    """Cache for reranking features and scores."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 7200):
        self.cache = LRUCache(max_size=max_size, default_ttl=ttl_seconds)
    
    def _hash_rerank_input(self, query: str, documents: List[str], model: str) -> str:
        """Create hash key for reranking input."""
        # Create reproducible hash from query, documents, and model
        content = {
            "query": query.strip().lower(),
            "documents": [doc.strip() for doc in documents],
            "model": model
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def get_rerank_scores(self, 
                              query: str, 
                              documents: List[str], 
                              model: str) -> Optional[List[float]]:
        """Get cached rerank scores."""
        key = self._hash_rerank_input(query, documents, model)
        return await self.cache.get(key)
    
    async def set_rerank_scores(self, 
                              query: str, 
                              documents: List[str], 
                              model: str,
                              scores: List[float]) -> None:
        """Cache rerank scores."""
        key = self._hash_rerank_input(query, documents, model)
        await self.cache.set(key, scores)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = await self.cache.hit_rate()
        size = await self.cache.size()
        
        return {
            "type": "rerank_cache",
            "size": size,
            "max_size": self.cache.max_size,
            "hit_rate": hit_rate,
            **self.cache.stats
        }


class PromptCache:
    """Cache for compiled prompt templates."""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 86400):
        self.cache = LRUCache(max_size=max_size, default_ttl=ttl_seconds)
    
    def _hash_prompt_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Create hash key for prompt template and variables."""
        content = {
            "template": template_name,
            "variables": sorted(variables.items())
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def get_prompt(self, 
                        template_name: str, 
                        variables: Dict[str, Any]) -> Optional[str]:
        """Get cached compiled prompt."""
        key = self._hash_prompt_template(template_name, variables)
        return await self.cache.get(key)
    
    async def set_prompt(self, 
                        template_name: str, 
                        variables: Dict[str, Any],
                        compiled_prompt: str) -> None:
        """Cache compiled prompt."""
        key = self._hash_prompt_template(template_name, variables)
        await self.cache.set(key, compiled_prompt)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = await self.cache.hit_rate()
        size = await self.cache.size()
        
        return {
            "type": "prompt_cache",
            "size": size,
            "max_size": self.cache.max_size,
            "hit_rate": hit_rate,
            **self.cache.stats
        }


class PersistentBM25Cache:
    """Persistent cache for BM25 indices."""
    
    def __init__(self, 
                 cache_dir: str = "cache/bm25",
                 rebuild_threshold: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rebuild_threshold = rebuild_threshold
        self._loaded_indices: Dict[str, Any] = {}
    
    def _get_cache_path(self, collection_name: str) -> Path:
        """Get cache file path for collection."""
        return self.cache_dir / f"{collection_name}_bm25.pkl"
    
    def _get_metadata_path(self, collection_name: str) -> Path:
        """Get metadata file path for collection."""
        return self.cache_dir / f"{collection_name}_bm25_meta.json"
    
    async def load_bm25_index(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Load BM25 index from cache."""
        if collection_name in self._loaded_indices:
            return self._loaded_indices[collection_name]
        
        cache_path = self._get_cache_path(collection_name)
        metadata_path = self._get_metadata_path(collection_name)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is stale
            cache_age = time.time() - metadata.get("created_at", 0)
            if cache_age > 86400:  # 24 hours
                logger.info(f"BM25 cache for {collection_name} is stale, will rebuild")
                return None
            
            # Load index
            with open(cache_path, 'rb') as f:
                bm25_index = pickle.load(f)
            
            # Cache in memory
            self._loaded_indices[collection_name] = {
                "index": bm25_index,
                "metadata": metadata
            }
            
            logger.info(f"Loaded BM25 index for {collection_name} from cache")
            return self._loaded_indices[collection_name]
        
        except Exception as e:
            logger.error(f"Failed to load BM25 cache for {collection_name}: {e}")
            return None
    
    async def save_bm25_index(self, 
                            collection_name: str, 
                            bm25_index: Any,
                            document_count: int) -> None:
        """Save BM25 index to cache."""
        cache_path = self._get_cache_path(collection_name)
        metadata_path = self._get_metadata_path(collection_name)
        
        try:
            # Save index
            with open(cache_path, 'wb') as f:
                pickle.dump(bm25_index, f)
            
            # Save metadata
            metadata = {
                "collection_name": collection_name,
                "document_count": document_count,
                "created_at": time.time(),
                "cache_version": "1.0"
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Cache in memory
            self._loaded_indices[collection_name] = {
                "index": bm25_index,
                "metadata": metadata
            }
            
            logger.info(f"Saved BM25 index for {collection_name} to cache")
        
        except Exception as e:
            logger.error(f"Failed to save BM25 cache for {collection_name}: {e}")
    
    async def should_rebuild(self, collection_name: str, current_doc_count: int) -> bool:
        """Check if BM25 index should be rebuilt."""
        metadata_path = self._get_metadata_path(collection_name)
        
        if not metadata_path.exists():
            return True
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_doc_count = metadata.get("document_count", 0)
            return current_doc_count - cached_doc_count >= self.rebuild_threshold
        
        except Exception:
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*_bm25.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "type": "bm25_cache",
            "cached_collections": len(cache_files),
            "memory_cached": len(self._loaded_indices),
            "total_size_bytes": total_size,
            "cache_directory": str(self.cache_dir)
        }


class CacheManager:
    """Central cache manager for all cache types."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize caches based on configuration
        cache_config = self.config.get("caching", {})
        
        self.embedding_cache = EmbeddingCache(
            max_size=cache_config.get("query_embeddings", {}).get("max_size", 10000),
            ttl_seconds=cache_config.get("query_embeddings", {}).get("ttl_seconds", 3600)
        ) if cache_config.get("query_embeddings", {}).get("enabled", True) else None
        
        self.candidate_cache = CandidateCache(
            max_size=cache_config.get("candidate_ids", {}).get("max_size", 5000),
            ttl_seconds=cache_config.get("candidate_ids", {}).get("ttl_seconds", 1800)
        ) if cache_config.get("candidate_ids", {}).get("enabled", True) else None
        
        self.rerank_cache = RerankCache(
            max_size=cache_config.get("rerank_features", {}).get("max_size", 1000),
            ttl_seconds=cache_config.get("rerank_features", {}).get("ttl_seconds", 7200)
        ) if cache_config.get("rerank_features", {}).get("enabled", True) else None
        
        self.prompt_cache = PromptCache(
            max_size=cache_config.get("prompt_cache", {}).get("max_size", 500),
            ttl_seconds=cache_config.get("prompt_cache", {}).get("ttl_seconds", 86400)
        ) if cache_config.get("prompt_cache", {}).get("enabled", True) else None
        
        self.bm25_cache = PersistentBM25Cache(
            rebuild_threshold=cache_config.get("bm25_index", {}).get("rebuild_threshold", 1000)
        ) if cache_config.get("bm25_index", {}).get("enabled", True) else None
        
        logger.info("CacheManager initialized with configuration")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "retrieval_tuning.yaml"
        
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        stats = {
            "cache_manager": {
                "config_loaded": bool(self.config),
                "enabled_caches": []
            }
        }
        
        if self.embedding_cache:
            stats["embedding_cache"] = await self.embedding_cache.get_stats()
            stats["cache_manager"]["enabled_caches"].append("embedding")
        
        if self.candidate_cache:
            stats["candidate_cache"] = await self.candidate_cache.get_stats()
            stats["cache_manager"]["enabled_caches"].append("candidate")
        
        if self.rerank_cache:
            stats["rerank_cache"] = await self.rerank_cache.get_stats()
            stats["cache_manager"]["enabled_caches"].append("rerank")
        
        if self.prompt_cache:
            stats["prompt_cache"] = await self.prompt_cache.get_stats()
            stats["cache_manager"]["enabled_caches"].append("prompt")
        
        if self.bm25_cache:
            stats["bm25_cache"] = await self.bm25_cache.get_stats()
            stats["cache_manager"]["enabled_caches"].append("bm25")
        
        return stats
    
    async def clear_all_caches(self) -> None:
        """Clear all caches."""
        if self.embedding_cache:
            await self.embedding_cache.cache.clear()
        
        if self.candidate_cache:
            await self.candidate_cache.cache.clear()
        
        if self.rerank_cache:
            await self.rerank_cache.cache.clear()
        
        if self.prompt_cache:
            await self.prompt_cache.cache.clear()
        
        logger.info("All caches cleared")
    
    async def get_overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all caches."""
        total_hits = 0
        total_requests = 0
        
        for cache in [self.embedding_cache, self.candidate_cache, 
                     self.rerank_cache, self.prompt_cache]:
            if cache:
                stats = cache.cache.stats
                total_hits += stats["hits"]
                total_requests += stats["hits"] + stats["misses"]
        
        return total_hits / total_requests if total_requests > 0 else 0.0


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
