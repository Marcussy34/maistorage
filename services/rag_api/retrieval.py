"""
Phase 9 Optimized Hybrid Retrieval System for MAI Storage RAG API.

This module implements the core retrieval logic combining:
- Dense vector search (semantic similarity) with caching
- BM25 lexical search (keyword matching) with persistent indices
- Reciprocal Rank Fusion (RRF) for result combination
- Cross-encoder reranking with feature caching
- Maximal Marginal Relevance (MMR) for diversity
- Context condensation for optimized token usage
- Multi-layer caching for performance optimization
"""

import asyncio
import logging
import os
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime

# Core dependencies
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

# Local imports
from models import (
    Document, RetrievalRequest, RetrievalResponse, RetrievalResult,
    RetrievalMethod, RerankMethod, CollectionInfo
)
from tools import (
    Timer, preprocess_text, tokenize_text, calculate_bm25_scores,
    reciprocal_rank_fusion, maximal_marginal_relevance, 
    normalize_scores, combine_scores, deduplicate_results,
    validate_query, create_debug_info
)
from cache import CacheManager, get_cache_manager
from context_condenser import ContextCondenser

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Main retrieval class implementing hybrid search with dense vectors and BM25.
    
    This class combines multiple retrieval strategies:
    1. Dense vector search using embeddings
    2. BM25 lexical search using term frequency
    3. Result fusion using RRF
    4. Cross-encoder reranking
    5. MMR for diversity
    """
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 embedding_model: str = "text-embedding-3-small",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 openai_api_key: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the optimized hybrid retriever with caching and performance tuning.
        
        Args:
            qdrant_url: Qdrant server URL
            embedding_model: OpenAI embedding model name
            reranker_model: Sentence-transformers reranker model
            openai_api_key: OpenAI API key
            config_path: Path to retrieval_tuning.yaml configuration file
        """
        self.qdrant_url = qdrant_url
        self.embedding_model = embedding_model
        self.reranker_model_name = reranker_model
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        
        # Initialize context condenser
        self.context_condenser = ContextCondenser(
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            config=self.config
        )
        
        # Initialize clients with optimized parameters
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Set up OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize reranker (lazy loading)
        self._reranker_model = None
        
        # Legacy BM25 document cache (will be replaced by persistent cache)
        self._bm25_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking with enhanced metrics
        self.stats = {
            "total_queries": 0,
            "dense_queries": 0,
            "bm25_queries": 0,
            "hybrid_queries": 0,
            "total_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embedding_api_calls": 0,
            "rerank_operations": 0,
            "context_condensations": 0
        }
        
        # Apply HNSW optimizations if configured
        self._apply_hnsw_optimizations()
        
        logger.info(f"Phase 9 Optimized HybridRetriever initialized with Qdrant at {qdrant_url}")
        logger.info(f"Configuration loaded: {bool(self.config)}")
        logger.info(f"Cache manager enabled: {self.cache_manager is not None}")
        logger.info(f"Context condenser enabled: {bool(self.context_condenser)}")
    
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
    
    def _apply_hnsw_optimizations(self) -> None:
        """Apply HNSW parameter optimizations to Qdrant collections."""
        hnsw_config = self.config.get("hnsw", {})
        if not hnsw_config:
            logger.info("No HNSW configuration found, using defaults")
            return
        
        # Log the HNSW parameters that will be used
        ef_construct = hnsw_config.get("ef_construct", 256)
        m = hnsw_config.get("m", 16)
        ef_search = hnsw_config.get("ef_search", 64)
        
        logger.info(f"HNSW parameters - ef_construct: {ef_construct}, m: {m}, ef_search: {ef_search}")
        
        # Note: These parameters are applied during collection creation in the indexer
        # Here we just store them for use when querying
        self.hnsw_ef_search = ef_search
    
    @property
    def reranker_model(self) -> CrossEncoder:
        """Lazy load the reranker model."""
        if self._reranker_model is None:
            logger.info(f"Loading reranker model: {self.reranker_model_name}")
            self._reranker_model = CrossEncoder(self.reranker_model_name)
        return self._reranker_model
    
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get information about a Qdrant collection."""
        try:
            collection = self.qdrant_client.get_collection(collection_name)
            
            from models import DistanceMetric
            return CollectionInfo(
                name=collection_name,
                vectors_count=collection.points_count,
                indexed_documents=collection.points_count,  # Approximate
                vector_size=collection.config.params.vectors.size,
                distance_metric=DistanceMetric.from_qdrant(collection.config.params.vectors.distance.value),
                index_type="HNSW",
                index_params={
                    "m": collection.config.params.vectors.hnsw_config.m if collection.config.params.vectors.hnsw_config else 16,
                    "ef_construct": collection.config.params.vectors.hnsw_config.ef_construct if collection.config.params.vectors.hnsw_config else 256
                }
            )
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query using OpenAI with caching."""
        try:
            # Check cache first
            if self.cache_manager and self.cache_manager.embedding_cache:
                cached_embedding = await self.cache_manager.embedding_cache.get_embedding(
                    query, self.embedding_model
                )
                if cached_embedding is not None:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Query embedding retrieved from cache")
                    return cached_embedding
                else:
                    self.stats["cache_misses"] += 1
            
            # Generate new embedding
            with Timer("query_embedding") as timer:
                response = self.openai_client.embeddings.create(
                    input=query,
                    model=self.embedding_model
                )
                embedding = np.array(response.data[0].embedding)
                self.stats["embedding_api_calls"] += 1
            
            # Cache the embedding
            if self.cache_manager and self.cache_manager.embedding_cache:
                await self.cache_manager.embedding_cache.set_embedding(
                    query, self.embedding_model, embedding
                )
            
            logger.debug(f"Query embedding generated in {timer.elapsed_ms:.2f}ms")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def dense_search(self, 
                          query: str,
                          collection_name: str,
                          top_k: int = 50,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform dense vector search using Qdrant.
        
        Args:
            query: Search query
            collection_name: Qdrant collection name
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            with Timer("dense_search") as timer:
                # Generate query embedding
                query_embedding = await self.embed_query(query)
                
                # Prepare Qdrant filter
                qdrant_filter = None
                if filters:
                    # Convert filters to Qdrant format (simplified)
                    conditions = []
                    for key, value in filters.items():
                        conditions.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchValue(value=value)
                            )
                        )
                    if conditions:
                        qdrant_filter = qdrant_models.Filter(must=conditions)
                
                # Perform search with optimized HNSW parameters
                search_params = qdrant_models.SearchParams(
                    hnsw_ef=getattr(self, 'hnsw_ef_search', 64)  # Use optimized ef_search
                )
                
                search_results = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding.tolist(),
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,  # We don't need vectors in results
                    search_params=search_params
                )
            
            # Convert to our format
            results = []
            for hit in search_results:
                result = {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {},
                    "text": hit.payload.get("text", "") if hit.payload else ""
                }
                results.append(result)
            
            logger.debug(f"Dense search returned {len(results)} results in {timer.elapsed_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise
    
    async def bm25_search(self,
                         query: str,
                         collection_name: str,
                         top_k: int = 50,
                         k1: float = 1.2,
                         b: float = 0.75) -> List[Dict[str, Any]]:
        """
        Perform BM25 lexical search.
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results to return
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            List of search results with BM25 scores
        """
        try:
            with Timer("bm25_search") as timer:
                # Get or build BM25 index for collection
                bm25_data = await self._get_bm25_index(collection_name)
                
                if not bm25_data:
                    logger.warning(f"No BM25 data available for collection {collection_name}")
                    return []
                
                # Tokenize query
                query_tokens = tokenize_text(preprocess_text(query))
                
                if not query_tokens:
                    logger.warning("No valid tokens in query for BM25 search")
                    return []
                
                # Calculate BM25 scores
                doc_tokens_list = bm25_data["doc_tokens"]
                scores = calculate_bm25_scores(query_tokens, doc_tokens_list, k1=k1, b=b)
                
                # Combine scores with document info
                results = []
                for i, score in enumerate(scores):
                    if score > 0:  # Only include documents with positive scores
                        doc_info = bm25_data["documents"][i]
                        result = {
                            "id": doc_info["id"],
                            "score": score,
                            "payload": doc_info.get("payload", {}),
                            "text": doc_info.get("text", "")
                        }
                        results.append(result)
                
                # Sort by score and limit
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:top_k]
            
            logger.debug(f"BM25 search returned {len(results)} results in {timer.elapsed_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise
    
    async def _get_bm25_index(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get or build BM25 index for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            BM25 index data or None if failed
        """
        # Check cache first
        if collection_name in self._bm25_cache:
            logger.debug(f"Using cached BM25 index for {collection_name}")
            return self._bm25_cache[collection_name]
        
        try:
            logger.info(f"Building BM25 index for collection {collection_name}")
            
            # Scroll through all documents in collection
            documents = []
            doc_tokens = []
            
            # Get all points from collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on your collection size
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            
            for point in points:
                doc_info = {
                    "id": str(point.id),
                    "payload": point.payload or {},
                    "text": point.payload.get("text", "") if point.payload else ""
                }
                documents.append(doc_info)
                
                # Tokenize document text
                text = preprocess_text(doc_info["text"])
                tokens = tokenize_text(text)
                doc_tokens.append(tokens)
            
            # Cache the index
            bm25_data = {
                "documents": documents,
                "doc_tokens": doc_tokens,
                "created_at": datetime.utcnow()
            }
            
            self._bm25_cache[collection_name] = bm25_data
            
            logger.info(f"BM25 index built for {collection_name} with {len(documents)} documents")
            return bm25_data
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index for {collection_name}: {e}")
            return None
    
    async def rerank_results(self,
                           query: str,
                           results: List[Dict[str, Any]],
                           method: RerankMethod = RerankMethod.BGE_RERANKER_V2,
                           top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: Original search query
            results: List of retrieval results
            method: Reranking method to use
            top_k: Number of results to rerank
            
        Returns:
            Reranked results with updated scores
        """
        if method == RerankMethod.NONE or not results:
            return results
        
        try:
            with Timer("reranking") as timer:
                # Limit to top_k results for reranking
                results_to_rerank = results[:top_k]
                
                if method == RerankMethod.BGE_RERANKER_V2:
                    # Prepare query-document pairs
                    pairs = []
                    for result in results_to_rerank:
                        pairs.append([query, result["text"]])
                    
                    # Get reranking scores
                    rerank_scores = self.reranker_model.predict(pairs)
                    
                    # Update results with rerank scores
                    for i, result in enumerate(results_to_rerank):
                        result["rerank_score"] = float(rerank_scores[i])
                        result["original_score"] = result.get("score", 0.0)
                        result["score"] = result["rerank_score"]  # Use rerank score as primary
                    
                    # Sort by rerank score
                    results_to_rerank.sort(key=lambda x: x["rerank_score"], reverse=True)
                
                else:
                    logger.warning(f"Unsupported reranking method: {method}")
                    return results
            
            logger.debug(f"Reranked {len(results_to_rerank)} results in {timer.elapsed_ms:.2f}ms")
            
            # Return reranked results plus any remaining results
            remaining_results = results[top_k:] if len(results) > top_k else []
            return results_to_rerank + remaining_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results if reranking fails
            return results
    
    async def apply_mmr(self,
                       query: str,
                       results: List[Dict[str, Any]],
                       lambda_param: float = 0.5,
                       top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        Args:
            query: Original search query
            results: List of retrieval results
            lambda_param: MMR lambda parameter (0=diversity, 1=relevance)
            top_k: Number of diverse results to select
            
        Returns:
            Diversified results
        """
        if not results or len(results) <= top_k:
            return results
        
        try:
            with Timer("mmr") as timer:
                # Generate query embedding
                query_embedding = await self.embed_query(query)
                
                # Get embeddings for all result documents
                doc_texts = [result["text"] for result in results]
                doc_embeddings_response = self.openai_client.embeddings.create(
                    input=doc_texts,
                    model=self.embedding_model
                )
                
                doc_embeddings = np.array([
                    item.embedding for item in doc_embeddings_response.data
                ])
                
                # Prepare data for MMR
                doc_ids = [result["id"] for result in results]
                relevance_scores = [result.get("score", 0.0) for result in results]
                
                # Apply MMR
                mmr_results = maximal_marginal_relevance(
                    query_embedding=query_embedding,
                    doc_embeddings=doc_embeddings,
                    doc_ids=doc_ids,
                    relevance_scores=relevance_scores,
                    lambda_param=lambda_param,
                    top_k=top_k
                )
                
                # Reconstruct results in MMR order
                result_map = {result["id"]: result for result in results}
                mmr_ordered_results = []
                
                for doc_id, mmr_score in mmr_results:
                    if doc_id in result_map:
                        result = result_map[doc_id].copy()
                        result["mmr_score"] = mmr_score
                        mmr_ordered_results.append(result)
            
            logger.debug(f"MMR applied to {len(results)} results, selected {len(mmr_ordered_results)} in {timer.elapsed_ms:.2f}ms")
            return mmr_ordered_results
            
        except Exception as e:
            logger.error(f"MMR application failed: {e}")
            # Return top results if MMR fails
            return results[:top_k]
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Main retrieval method implementing the full hybrid pipeline.
        
        Args:
            request: Retrieval request with parameters
            
        Returns:
            Complete retrieval response with results and metadata
        """
        start_time = datetime.utcnow()
        timing_info = {}
        
        try:
            # Validate query
            is_valid, error_msg = validate_query(request.query)
            if not is_valid:
                raise ValueError(f"Invalid query: {error_msg}")
            
            # Update stats
            self.stats["total_queries"] += 1
            
            logger.info(f"Processing retrieval request: method={request.method}, query='{request.query[:100]}...'")
            
            # Initialize result containers
            dense_results = []
            bm25_results = []
            final_results = []
            
            # Step 1: Retrieve using specified method(s)
            if request.method == RetrievalMethod.DENSE:
                with Timer("dense_retrieval") as timer:
                    dense_results = await self.dense_search(
                        query=request.query,
                        collection_name=request.collection_name,
                        top_k=request.top_k,
                        filters=request.filters
                    )
                    final_results = dense_results
                timing_info["dense_time_ms"] = timer.elapsed_ms
                self.stats["dense_queries"] += 1
                
            elif request.method == RetrievalMethod.BM25:
                with Timer("bm25_retrieval") as timer:
                    bm25_results = await self.bm25_search(
                        query=request.query,
                        collection_name=request.collection_name,
                        top_k=request.top_k
                    )
                    final_results = bm25_results
                timing_info["bm25_time_ms"] = timer.elapsed_ms
                self.stats["bm25_queries"] += 1
                
            elif request.method == RetrievalMethod.HYBRID:
                # Perform both dense and BM25 search
                with Timer("dense_retrieval") as timer:
                    dense_results = await self.dense_search(
                        query=request.query,
                        collection_name=request.collection_name,
                        top_k=request.top_k_dense,
                        filters=request.filters
                    )
                timing_info["dense_time_ms"] = timer.elapsed_ms
                
                with Timer("bm25_retrieval") as timer:
                    bm25_results = await self.bm25_search(
                        query=request.query,
                        collection_name=request.collection_name,
                        top_k=request.top_k_bm25
                    )
                timing_info["bm25_time_ms"] = timer.elapsed_ms
                
                # Step 2: Fusion using RRF
                with Timer("fusion") as timer:
                    final_results = await self._fuse_results(
                        dense_results=dense_results,
                        bm25_results=bm25_results,
                        rrf_k=request.rrf_k,
                        dense_weight=request.dense_weight,
                        bm25_weight=request.bm25_weight,
                        top_k=request.top_k
                    )
                timing_info["fusion_time_ms"] = timer.elapsed_ms
                self.stats["hybrid_queries"] += 1
            
            else:
                raise ValueError(f"Unsupported retrieval method: {request.method}")
            
            # Step 3: Reranking
            if request.rerank_method != RerankMethod.NONE and final_results:
                with Timer("reranking") as timer:
                    final_results = await self.rerank_results(
                        query=request.query,
                        results=final_results,
                        method=request.rerank_method,
                        top_k=request.rerank_top_k
                    )
                timing_info["rerank_time_ms"] = timer.elapsed_ms
            
            # Step 4: Apply MMR for diversity
            if request.enable_mmr and final_results:
                with Timer("mmr") as timer:
                    final_results = await self.apply_mmr(
                        query=request.query,
                        results=final_results,
                        lambda_param=request.mmr_lambda,
                        top_k=request.top_k
                    )
                timing_info["mmr_time_ms"] = timer.elapsed_ms
            
            # Step 5: Deduplicate results
            result_tuples = [(r["id"], r.get("score", 0.0)) for r in final_results]
            deduplicated_tuples = deduplicate_results(result_tuples)
            
            # Reconstruct final results
            result_map = {r["id"]: r for r in final_results}
            deduplicated_results = []
            for doc_id, score in deduplicated_tuples:
                if doc_id in result_map:
                    result = result_map[doc_id]
                    result["final_score"] = score
                    deduplicated_results.append(result)
            
            final_results = deduplicated_results[:request.top_k]
            
            # Step 6: Convert to response format
            retrieval_results = []
            for i, result in enumerate(final_results):
                # Create document
                document = Document(
                    id=result["id"],
                    text=result["text"],
                    metadata=result.get("payload", {}),
                    doc_name=result.get("payload", {}).get("doc_name"),
                    chunk_index=result.get("payload", {}).get("chunk_index"),
                    total_chunks=result.get("payload", {}).get("total_chunks"),
                    file_type=result.get("payload", {}).get("file_type"),
                    char_count=result.get("payload", {}).get("char_count"),
                    timestamp=result.get("payload", {}).get("timestamp"),
                    start_index=result.get("payload", {}).get("start_index")
                )
                
                # Create retrieval result
                retrieval_result = RetrievalResult(
                    document=document,
                    dense_score=result.get("dense_score"),
                    bm25_score=result.get("bm25_score"),
                    hybrid_score=result.get("hybrid_score"),
                    rerank_score=result.get("rerank_score"),
                    final_score=result.get("final_score", result.get("score", 0.0)),
                    final_rank=i + 1
                )
                
                retrieval_results.append(retrieval_result)
            
            # Calculate total time
            end_time = datetime.utcnow()
            total_time_ms = (end_time - start_time).total_seconds() * 1000
            self.stats["total_time_ms"] += total_time_ms
            
            # Get collection info
            collection_info = await self.get_collection_info(request.collection_name)
            
            # Create debug info
            debug_info = None
            if request.include_scores:
                debug_info = create_debug_info(
                    dense_results=dense_results if dense_results else None,
                    bm25_results=bm25_results if bm25_results else None,
                    fusion_method="RRF" if request.method == RetrievalMethod.HYBRID else None,
                    rerank_method=request.rerank_method.value if request.rerank_method != RerankMethod.NONE else None,
                    mmr_applied=request.enable_mmr,
                    timing_info=timing_info
                )
            
            # Build response
            response = RetrievalResponse(
                query=request.query,
                results=retrieval_results,
                total_results=len(retrieval_results),
                method_used=request.method,
                rerank_method=request.rerank_method if request.rerank_method != RerankMethod.NONE else None,
                retrieval_time_ms=total_time_ms,
                dense_time_ms=timing_info.get("dense_time_ms"),
                bm25_time_ms=timing_info.get("bm25_time_ms"),
                fusion_time_ms=timing_info.get("fusion_time_ms"),
                rerank_time_ms=timing_info.get("rerank_time_ms"),
                mmr_time_ms=timing_info.get("mmr_time_ms"),
                collection_name=request.collection_name,
                collection_size=collection_info.vectors_count if collection_info else None,
                debug_info=debug_info
            )
            
            logger.info(f"Retrieval completed: {len(retrieval_results)} results in {total_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    async def _fuse_results(self,
                           dense_results: List[Dict[str, Any]],
                           bm25_results: List[Dict[str, Any]],
                           rrf_k: int = 60,
                           dense_weight: float = 0.5,
                           bm25_weight: float = 0.5,
                           top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Fuse dense and BM25 results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            bm25_results: Results from BM25 retrieval  
            rrf_k: RRF k parameter
            dense_weight: Weight for dense results
            bm25_weight: Weight for BM25 results
            top_k: Number of fused results to return
            
        Returns:
            Fused and ranked results
        """
        # Create rankings for RRF
        dense_ranking = [result["id"] for result in dense_results]
        bm25_ranking = [result["id"] for result in bm25_results]
        
        rankings = [dense_ranking, bm25_ranking]
        weights = [dense_weight, bm25_weight]
        
        # Apply RRF
        rrf_results = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
        
        # Create result map for quick lookup
        all_results = {}
        
        # Add dense results
        for i, result in enumerate(dense_results):
            result_copy = result.copy()
            result_copy["dense_score"] = result["score"]
            result_copy["dense_rank"] = i + 1
            all_results[result["id"]] = result_copy
        
        # Add BM25 results (merge if already exists)
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            if doc_id in all_results:
                all_results[doc_id]["bm25_score"] = result["score"]
                all_results[doc_id]["bm25_rank"] = i + 1
            else:
                result_copy = result.copy()
                result_copy["bm25_score"] = result["score"]
                result_copy["bm25_rank"] = i + 1
                all_results[doc_id] = result_copy
        
        # Build final fused results
        fused_results = []
        for doc_id, rrf_score in rrf_results[:top_k]:
            if doc_id in all_results:
                result = all_results[doc_id]
                result["hybrid_score"] = rrf_score
                result["score"] = rrf_score  # Use RRF score as primary score
                fused_results.append(result)
        
        return fused_results
    
    def clear_cache(self):
        """Clear BM25 cache."""
        self._bm25_cache.clear()
        logger.info("BM25 cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval and caching statistics."""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["total_queries"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_queries"]
        else:
            stats["avg_time_ms"] = 0.0
        
        # Calculate cache hit rate
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        # Add cache manager statistics
        if self.cache_manager:
            cache_stats = await self.cache_manager.get_all_stats()
            stats["cache_details"] = cache_stats
            
            # Overall hit rate from cache manager
            overall_hit_rate = await self.cache_manager.get_overall_hit_rate()
            stats["overall_cache_hit_rate"] = overall_hit_rate
        
        # Add configuration info
        stats["configuration"] = {
            "config_loaded": bool(self.config),
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model_name,
            "hnsw_ef_search": getattr(self, 'hnsw_ef_search', 64),
            "context_condenser_enabled": bool(self.context_condenser)
        }
        
        return stats
