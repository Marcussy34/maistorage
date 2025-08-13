"""
LangGraph agentic RAG workflow implementation for Phase 5.

This module implements the multi-step agentic RAG loop with planner, retriever,
synthesizer, and verifier nodes using LangGraph's StateGraph architecture.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

from llm_client import LLMClient, LLMConfig
from retrieval import HybridRetriever, RetrievalRequest
from models import RetrievalMethod, RerankMethod, CitationEngineConfig
from prompts.planner import format_planner_prompt
from prompts.baseline import format_baseline_prompt, format_context_from_results
from prompts.verifier import format_verifier_prompt, format_faithfulness_check
from citer import SentenceCitationEngine, create_citation_engine

logger = logging.getLogger(__name__)


def safe_text_for_json(text: str, max_length: int = 500) -> str:
    """
    Safely prepare text content for JSON serialization.
    
    Args:
        text: Raw text that may contain problematic characters
        max_length: Maximum length to truncate to
        
    Returns:
        Safe text that won't break JSON parsing
    """
    if not text:
        return ""
    
    # Remove control characters that can break JSON
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    # Replace problematic quotes and backslashes
    text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


class AgentStep(str, Enum):
    """Steps in the agentic RAG workflow."""
    PLANNER = "planner"
    RETRIEVER = "retriever"
    SYNTHESIZER = "synthesizer"
    VERIFIER = "verifier"
    DONE = "done"


class TraceEventType(str, Enum):
    """Types of trace events emitted during execution."""
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    TOKEN = "token"
    SOURCES = "sources"
    METRICS = "metrics"
    VERIFICATION = "verification"
    DONE = "done"


class TraceEvent(BaseModel):
    """A single trace event in the agentic workflow."""
    event_type: TraceEventType
    timestamp: datetime
    step: Optional[AgentStep] = None
    data: Dict[str, Any] = {}
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class AgentState(TypedDict):
    """State maintained throughout the agentic RAG workflow."""
    # Input
    query: str
    original_query: str
    
    # Planning
    plan: Optional[str]
    sub_queries: List[str]
    key_concepts: List[str]
    
    # Retrieval
    retrieval_results: List[Dict[str, Any]]
    context: str
    
    # Synthesis
    answer: str
    citations: List[Dict[str, Any]]
    
    # Verification
    verification_result: Optional[Dict[str, Any]]
    needs_refinement: bool
    refinement_count: int
    
    # Workflow control
    current_step: AgentStep
    max_refinements: int
    
    # Performance tracking
    start_time: float
    step_times: Dict[str, float]
    trace_events: List[TraceEvent]
    
    # Configuration
    top_k: int
    enable_verification: bool
    enable_sentence_citations: bool
    
    # Sentence-level citations (Phase 6)
    sentence_attribution: Optional[Dict[str, Any]]
    
    # Token usage tracking
    tokens_used: Optional[Dict[str, Any]]
    total_tokens: Optional[int]


class AgenticRAG:
    """
    Agentic RAG system using LangGraph for multi-step reasoning workflow.
    
    Implements the Phase 5 architecture:
    - Planner: Analyzes query and creates retrieval strategy
    - Retriever: Executes hybrid search based on plan
    - Synthesizer: Generates answer from retrieved context
    - Verifier: Validates answer quality and determines if refinement needed
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_config: Optional[LLMConfig] = None,
        max_refinements: int = 2,
        citation_engine: Optional[SentenceCitationEngine] = None
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            retriever: Hybrid retriever for document search
            llm_config: Configuration for LLM client
            max_refinements: Maximum number of refinement iterations
            citation_engine: Optional sentence-level citation engine (Phase 6)
        """
        self.retriever = retriever
        self.max_refinements = max_refinements
        self.citation_engine = citation_engine
        
        # Initialize LLM client for all agent components
        self.llm = LLMClient(llm_config or LLMConfig())
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("Agentic RAG workflow initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for agentic RAG."""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("verifier", self._verifier_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "synthesizer")
        workflow.add_edge("synthesizer", "verifier")
        
        # Conditional edge from verifier
        workflow.add_conditional_edges(
            "verifier",
            self._should_refine,
            {
                "refine": "planner",  # Go back to planning for refinement
                "done": END           # Complete the workflow
            }
        )
        
        return workflow.compile()
    
    async def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Planner node: Analyze query and create retrieval strategy.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with planning results
        """
        start_time = time.time()
        
        # Emit step start event
        trace_event = TraceEvent(
            event_type=TraceEventType.STEP_START,
            step=AgentStep.PLANNER,
            data={"query": state["query"]}
        )
        
        try:
            logger.info(f"Planner analyzing query: {state['query']}")
            
            # Generate plan using LLM
            planner_messages = format_planner_prompt(state["query"])
            response = await self.llm.achat_completion(planner_messages)
            
            # Parse planning response (simplified - in production might use structured output)
            plan_content = response.content
            
            # Extract key concepts and sub-queries from plan
            # For now, using simple heuristics - could be enhanced with structured prompting
            key_concepts = self._extract_key_concepts(plan_content)
            sub_queries = self._extract_sub_queries(state["query"], plan_content)
            
            # Calculate step time
            step_time = (time.time() - start_time) * 1000
            
            # Emit completion event with safely handled plan content
            completion_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.PLANNER,
                data={
                    "plan": safe_text_for_json(plan_content),
                    "plan_length": len(plan_content),
                    "key_concepts": key_concepts,
                    "sub_queries": sub_queries,
                    "time_ms": step_time
                }
            )
            
            # Add trace events to existing list
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, completion_event])
            
            return {
                "plan": plan_content,
                "key_concepts": key_concepts,
                "sub_queries": sub_queries,
                "current_step": AgentStep.RETRIEVER,
                "step_times": {**state.get("step_times", {}), "planner": step_time},
                "trace_events": new_trace_events
            }
            
        except Exception as e:
            logger.error(f"Planner node failed: {e}")
            error_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.PLANNER,
                data={"error": safe_text_for_json(str(e)), "time_ms": (time.time() - start_time) * 1000}
            )
            
            # Fallback: use original query without planning
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, error_event])
            
            return {
                "plan": f"Simple retrieval for: {state['query']}",
                "key_concepts": [state["query"]],
                "sub_queries": [state["query"]],
                "current_step": AgentStep.RETRIEVER,
                "trace_events": new_trace_events
            }
    
    async def _retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retriever node: Execute hybrid search based on planning results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieval results
        """
        start_time = time.time()
        
        # Emit step start event
        trace_event = TraceEvent(
            event_type=TraceEventType.STEP_START,
            step=AgentStep.RETRIEVER,
            data={"queries": state.get("sub_queries", [state["query"]])}
        )
        
        try:
            logger.info(f"Retriever executing search for {len(state.get('sub_queries', []))} queries")
            
            # Determine which queries to use for retrieval
            queries_to_search = state.get("sub_queries", []) or [state["query"]]
            
            all_results = []
            
            # Execute retrieval for each sub-query (or main query)
            for query in queries_to_search:
                retrieval_request = RetrievalRequest(
                    query=query,
                    method=RetrievalMethod.HYBRID,
                    top_k=state.get("top_k", 10),
                    rerank_method=RerankMethod.BGE_RERANKER_V2,
                    enable_mmr=True
                )
                
                results = await self.retriever.retrieve(retrieval_request)
                all_results.extend(results.results)
            
            # Deduplicate and rerank combined results
            unique_results = self._deduplicate_results(all_results)
            top_results = unique_results[:state.get("top_k", 10)]
            
            # Format context for synthesis
            context = format_context_from_results(top_results)
            
            # Prepare citations data
            citations_data = [
                {
                    "doc_name": result.document.doc_name,
                    "chunk_index": result.document.chunk_index,
                    "text_snippet": result.document.text[:200] + "..." if len(result.document.text) > 200 else result.document.text,
                    "relevance_score": result.final_score or result.hybrid_score or 0.0
                }
                for result in top_results
            ]
            
            # Calculate step time
            step_time = (time.time() - start_time) * 1000
            
            # Emit sources event
            sources_event = TraceEvent(
                event_type=TraceEventType.SOURCES,
                step=AgentStep.RETRIEVER,
                data={"sources": citations_data, "total_results": len(top_results)}
            )
            
            # Emit completion event
            completion_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.RETRIEVER,
                data={
                    "results_count": len(top_results),
                    "time_ms": step_time
                }
            )
            
            # Add trace events to existing list
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, sources_event, completion_event])
            
            return {
                "retrieval_results": [result.dict() for result in top_results],
                "context": context,
                "citations": citations_data,
                "current_step": AgentStep.SYNTHESIZER,
                "step_times": {**state.get("step_times", {}), "retriever": step_time},
                "trace_events": new_trace_events
            }
            
        except Exception as e:
            logger.error(f"Retriever node failed: {e}")
            error_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.RETRIEVER,
                data={"error": safe_text_for_json(str(e)), "time_ms": (time.time() - start_time) * 1000}
            )
            
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, error_event])
            
            return {
                "retrieval_results": [],
                "context": "No context available due to retrieval error.",
                "citations": [],
                "current_step": AgentStep.SYNTHESIZER,
                "trace_events": new_trace_events
            }
    
    async def _synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesizer node: Generate answer from retrieved context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with synthesized answer
        """
        start_time = time.time()
        
        # Emit step start event
        trace_event = TraceEvent(
            event_type=TraceEventType.STEP_START,
            step=AgentStep.SYNTHESIZER,
            data={"context_length": len(state.get("context", ""))}
        )
        
        try:
            logger.info("Synthesizer generating answer from context")
            
            # Use baseline prompt for synthesis (could be enhanced with agentic-specific prompts)
            synthesis_messages = format_baseline_prompt(
                query=state["query"],
                context=state.get("context", "")
            )
            
            # Generate answer with streaming support (for future streaming implementation)
            response = await self.llm.achat_completion(synthesis_messages)
            answer = response.content
            

            
            # Generate sentence-level attributions if enabled
            sentence_attribution = None
            if state.get("enable_sentence_citations", False) and self.citation_engine:
                try:
                    logger.info("Generating sentence-level citations for agentic answer")
                    
                    # Convert retrieval results back to RetrievalResult objects
                    from models import RetrievalResult, Document
                    retrieval_results = []
                    for result_dict in state.get("retrieval_results", []):
                        # Reconstruct RetrievalResult from dictionary
                        doc_data = result_dict.get("document", {})
                        document = Document(
                            id=doc_data.get("id", "unknown"),
                            text=doc_data.get("text", ""),
                            metadata=doc_data.get("metadata", {}),
                            doc_name=doc_data.get("doc_name"),
                            chunk_index=doc_data.get("chunk_index"),
                            total_chunks=doc_data.get("total_chunks"),
                            file_type=doc_data.get("file_type"),
                            char_count=doc_data.get("char_count"),
                            start_index=doc_data.get("start_index")
                        )
                        
                        retrieval_result = RetrievalResult(
                            document=document,
                            scores=result_dict.get("scores", {}),
                            dense_score=result_dict.get("dense_score"),
                            bm25_score=result_dict.get("bm25_score"),
                            hybrid_score=result_dict.get("hybrid_score"),
                            rerank_score=result_dict.get("rerank_score"),
                            final_score=result_dict.get("final_score")
                        )
                        retrieval_results.append(retrieval_result)
                    
                    # Generate sentence attribution
                    attribution_result = await self.citation_engine.generate_sentence_citations(
                        response_text=answer,
                        retrieval_results=retrieval_results
                    )
                    
                    sentence_attribution = attribution_result.dict()
                    logger.info(f"Sentence attribution completed: {attribution_result.attribution_coverage:.2%} coverage")
                    
                except Exception as e:
                    logger.warning(f"Sentence attribution failed in agentic workflow: {e}")
            
            # Calculate step time
            step_time = (time.time() - start_time) * 1000
            
            # Emit completion event
            completion_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.SYNTHESIZER,
                data={
                    "answer_length": len(answer),
                    "time_ms": step_time,
                    "tokens_used": response.usage.get("total_tokens", 0),
                    "sentence_attribution_enabled": bool(sentence_attribution)
                }
            )
            
            # Add trace events to existing list
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, completion_event])
            
            return {
                "answer": answer,
                "sentence_attribution": sentence_attribution,
                "current_step": AgentStep.VERIFIER if state.get("enable_verification", True) else AgentStep.DONE,
                "step_times": {**state.get("step_times", {}), "synthesizer": step_time},
                "trace_events": new_trace_events,
                "tokens_used": response.usage,
                "total_tokens": response.usage.get("total_tokens", 0)
            }
            
        except Exception as e:
            logger.error(f"Synthesizer node failed: {e}")
            error_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.SYNTHESIZER,
                data={"error": safe_text_for_json(str(e)), "time_ms": (time.time() - start_time) * 1000}
            )
            
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, error_event])
            
            return {
                "answer": f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                "current_step": AgentStep.DONE,
                "trace_events": new_trace_events
            }
    
    async def _verifier_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Verifier node: Validate answer quality and determine if refinement needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with verification results
        """
        start_time = time.time()
        
        # Emit step start event
        trace_event = TraceEvent(
            event_type=TraceEventType.STEP_START,
            step=AgentStep.VERIFIER,
            data={"answer_length": len(state.get("answer", ""))}
        )
        
        try:
            logger.info("Verifier assessing answer quality")
            
            # Use simplified faithfulness check for now
            verification_messages = format_faithfulness_check(
                answer=state.get("answer", ""),
                context=state.get("context", "")
            )
            
            response = await self.llm.achat_completion(verification_messages)
            verification_content = response.content
            
            # Parse verification result (simplified)
            needs_refinement = self._parse_verification_result(verification_content)
            refinement_count = state.get("refinement_count", 0)
            
            # Check if we should refine (not if at max refinements)
            should_refine = (
                needs_refinement and 
                refinement_count < state.get("max_refinements", self.max_refinements)
            )
            
            # Calculate step time
            step_time = (time.time() - start_time) * 1000
            
            # Emit verification event
            verification_event = TraceEvent(
                event_type=TraceEventType.VERIFICATION,
                step=AgentStep.VERIFIER,
                data={
                    "verification": safe_text_for_json(verification_content),
                    "needs_refinement": needs_refinement,
                    "will_refine": should_refine,
                    "refinement_count": refinement_count
                }
            )
            
            # Emit completion event
            completion_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.VERIFIER,
                data={
                    "passed": not needs_refinement,
                    "time_ms": step_time
                }
            )
            
            # Add trace events to existing list
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, verification_event, completion_event])
            
            return {
                "verification_result": {
                    "content": safe_text_for_json(verification_content),
                    "needs_refinement": needs_refinement,
                    "passed": not needs_refinement
                },
                "needs_refinement": should_refine,
                "refinement_count": refinement_count + (1 if should_refine else 0),
                "current_step": AgentStep.DONE,
                "step_times": {**state.get("step_times", {}), "verifier": step_time},
                "trace_events": new_trace_events
            }
            
        except Exception as e:
            logger.error(f"Verifier node failed: {e}")
            error_event = TraceEvent(
                event_type=TraceEventType.STEP_COMPLETE,
                step=AgentStep.VERIFIER,
                data={"error": safe_text_for_json(str(e)), "time_ms": (time.time() - start_time) * 1000}
            )
            
            new_trace_events = state.get("trace_events", [])
            new_trace_events.extend([trace_event, error_event])
            
            return {
                "verification_result": {"passed": True, "error": safe_text_for_json(str(e))},
                "needs_refinement": False,
                "current_step": AgentStep.DONE,
                "trace_events": new_trace_events
            }
    
    def _should_refine(self, state: AgentState) -> str:
        """
        Conditional edge function to determine if refinement is needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            "refine" if refinement needed, "done" otherwise
        """
        return "refine" if state.get("needs_refinement", False) else "done"
    
    def _extract_key_concepts(self, plan_content: str) -> List[str]:
        """Extract key concepts from planner output."""
        # Simple heuristic - look for lines with "Key Concepts:" or similar
        concepts = []
        lines = plan_content.split('\n')
        
        for line in lines:
            if 'key concept' in line.lower() or 'important term' in line.lower():
                # Extract terms after colon or dash
                if ':' in line:
                    terms = line.split(':', 1)[1].strip()
                elif '-' in line:
                    terms = line.split('-', 1)[1].strip()
                else:
                    terms = line.strip()
                
                # Split on commas and clean up
                for term in terms.split(','):
                    clean_term = term.strip().strip('*').strip('-').strip()
                    if clean_term and len(clean_term) > 2:
                        concepts.append(clean_term)
        
        # Fallback: use original query terms
        if not concepts:
            concepts = [word.strip() for word in self._extract_important_words(plan_content)]
        
        return concepts[:5]  # Limit to top 5 concepts
    
    def _extract_sub_queries(self, original_query: str, plan_content: str) -> List[str]:
        """Extract sub-queries from planner output."""
        sub_queries = []
        lines = plan_content.split('\n')
        
        for line in lines:
            if 'sub-quer' in line.lower() or 'question' in line.lower():
                # Look for numbered lists or bullet points
                if any(char in line for char in ['1.', '2.', '3.', '-', '*']):
                    # Extract the question part
                    for sep in ['1.', '2.', '3.', '-', '*', ':']:
                        if sep in line:
                            query = line.split(sep, 1)[-1].strip()
                            if query and len(query) > 5 and '?' in query:
                                sub_queries.append(query)
                            break
        
        # If no sub-queries found, use the original query
        if not sub_queries:
            sub_queries = [original_query]
        
        return sub_queries[:3]  # Limit to 3 sub-queries
    
    def _extract_important_words(self, text: str) -> List[str]:
        """Extract important words from text (simple heuristic)."""
        # Simple word extraction - in production might use NLP libraries
        words = text.lower().split()
        important_words = []
        
        skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        for word in words:
            word = word.strip('.,?!";:')
            if len(word) > 3 and word not in skip_words:
                important_words.append(word)
        
        return important_words[:10]
    
    def _deduplicate_results(self, results: List[Any]) -> List[Any]:
        """Remove duplicate results based on document ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            doc_id = result.document.id
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        return unique_results
    
    def _parse_verification_result(self, verification_content: str) -> bool:
        """Parse verification result to determine if refinement is needed."""
        content_lower = verification_content.lower()
        
        # Look for negative indicators
        negative_indicators = [
            'unfaithful', 'partially_faithful', 'not_enough_context',
            'needs improvement', 'refinement needed', 'incorrect',
            'incomplete', 'unclear'
        ]
        
        positive_indicators = [
            'faithful', 'accurate', 'complete', 'correct',
            'well-supported', 'clear', 'satisfactory'
        ]
        
        # Check for explicit negative indicators
        for indicator in negative_indicators:
            if indicator in content_lower:
                return True  # Needs refinement
        
        # Check for explicit positive indicators
        for indicator in positive_indicators:
            if indicator in content_lower:
                return False  # No refinement needed
        
        # Default to no refinement if unclear
        return False
    
    async def run(
        self,
        query: str,
        top_k: int = 10,
        enable_verification: bool = True,
        max_refinements: int = None
    ) -> Dict[str, Any]:
        """
        Run the complete agentic RAG workflow.
        
        Args:
            query: User query to process
            top_k: Number of documents to retrieve
            enable_verification: Whether to run verification step
            max_refinements: Override default max refinements
            
        Returns:
            Final workflow state with answer and metadata
        """
        start_time = time.time()
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            original_query=query,
            plan=None,
            sub_queries=[],
            key_concepts=[],
            retrieval_results=[],
            context="",
            answer="",
            citations=[],
            verification_result=None,
            needs_refinement=False,
            refinement_count=0,
            current_step=AgentStep.PLANNER,
            max_refinements=max_refinements or self.max_refinements,
            start_time=start_time,
            step_times={},
            trace_events=[],
            top_k=top_k,
            enable_verification=enable_verification,
            enable_sentence_citations=bool(self.citation_engine),
            sentence_attribution=None
        )
        
        logger.info(f"Starting agentic RAG workflow for query: {query}")
        
        try:
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            
            # Add final metrics event
            metrics_event = TraceEvent(
                event_type=TraceEventType.METRICS,
                data={
                    "total_time_ms": total_time,
                    "step_times": final_state.get("step_times", {}),
                    "refinement_count": final_state.get("refinement_count", 0),
                    "verification_passed": not final_state.get("needs_refinement", False)
                }
            )
            
            # Add done event
            done_event = TraceEvent(
                event_type=TraceEventType.DONE,
                data={"success": True}
            )
            
            if "trace_events" not in final_state:
                final_state["trace_events"] = []
            final_state["trace_events"].extend([metrics_event, done_event])
            final_state["total_time_ms"] = total_time
            
            logger.info(f"Agentic RAG completed in {total_time:.2f}ms with {final_state.get('refinement_count', 0)} refinements")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Agentic RAG workflow failed: {e}")
            
            # Return error state
            return {
                **initial_state,
                "answer": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "trace_events": [
                    TraceEvent(
                        event_type=TraceEventType.DONE,
                        data={"success": False, "error": safe_text_for_json(str(e))}
                    )
                ],
                "total_time_ms": (time.time() - start_time) * 1000
            }


# Factory function for easy instantiation
def create_agentic_rag(
    retriever: HybridRetriever,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    enable_sentence_citations: bool = False,
    **kwargs
) -> AgenticRAG:
    """
    Factory function to create an agentic RAG instance.
    
    Args:
        retriever: Hybrid retriever instance
        model: LLM model name
        api_key: OpenAI API key
        enable_sentence_citations: Whether to enable sentence-level citations (Phase 6)
        **kwargs: Additional configuration
        
    Returns:
        Configured AgenticRAG instance
    """
    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    # Create citation engine if sentence citations are enabled
    citation_engine = None
    if enable_sentence_citations:
        try:
            # Create LLM client for citation engine
            llm_client = LLMClient(llm_config)
            
            # Create citation engine directly (synchronous creation)
            citation_engine = SentenceCitationEngine(
                retriever=retriever,
                llm_client=llm_client,
                config=CitationEngineConfig()
            )
            logger.info("Sentence citation engine enabled for agentic RAG")
        except Exception as e:
            logger.warning(f"Failed to create citation engine: {e}")
    
    return AgenticRAG(
        retriever=retriever, 
        llm_config=llm_config,
        citation_engine=citation_engine
    )
