#!/usr/bin/env python3
"""
Phase 8: Evaluation Harness with RAGAS and Retrieval Metrics

This module implements comprehensive evaluation of both Traditional and Agentic RAG systems
using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision/Recall) and 
retrieval-specific metrics (Recall@k, nDCG, MRR).

Usage:
    python run_ragas.py --golden_qa_path ../golden_qa.json --output_dir ./results
    python run_ragas.py --run_traditional --run_agentic --compare
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# RAGAS and evaluation imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from sklearn.metrics import ndcg_score

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag_baseline import BaselineRAG, RAGRequest
from graph import AgenticRAG
from retrieval import HybridRetriever
from llm_client import LLMClient
from models import RetrievalRequest


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question_id: int
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    response_time_ms: float
    token_usage: int
    retrieval_time_ms: float
    
    # RAGAS metrics
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    
    # Retrieval metrics
    recall_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    mrr_score: Optional[float] = None
    
    # Additional metadata
    mode: str = "traditional"  # "traditional" or "agentic"
    evaluation_timestamp: str = ""
    error: Optional[str] = None


@dataclass
class RetrievalMetrics:
    """Container for retrieval-specific metrics."""
    recall_at_k: float
    ndcg_at_k: float
    mrr_score: float
    precision_at_k: float
    map_score: float  # Mean Average Precision


class RAGEvaluator:
    """Comprehensive RAG evaluation using RAGAS and retrieval metrics."""
    
    def __init__(self, 
                 baseline_rag: Optional[BaselineRAG] = None,
                 agentic_rag: Optional[AgenticRAG] = None,
                 retriever: Optional[HybridRetriever] = None):
        """Initialize evaluator with RAG systems."""
        self.baseline_rag = baseline_rag
        self.agentic_rag = agentic_rag
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def evaluate_traditional_rag(self, 
                                     questions: List[Dict[str, Any]],
                                     top_k: int = 5) -> List[EvaluationResult]:
        """Evaluate traditional baseline RAG system."""
        results = []
        
        for question_data in questions:
            question_id = question_data["id"]
            question = question_data["question"]
            
            self.logger.info(f"Evaluating Traditional RAG - Question {question_id}: {question[:50]}...")
            
            try:
                start_time = time.time()
                
                # Create RAG request
                rag_request = RAGRequest(
                    query=question,
                    top_k=top_k,
                    enable_citations=True,
                    retrieval_method="hybrid"
                )
                
                # Generate response
                rag_response = await self.baseline_rag.generate(rag_request)
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Extract sources for RAGAS
                sources = []
                for citation in rag_response.citations:
                    sources.append({
                        "content": citation.snippet,
                        "doc_id": citation.doc_id,
                        "score": citation.score,
                        "title": citation.title or "Unknown"
                    })
                
                result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    answer=rag_response.answer,
                    sources=sources,
                    response_time_ms=response_time_ms,
                    token_usage=rag_response.total_tokens or 0,
                    retrieval_time_ms=rag_response.retrieval_time_ms or 0,
                    mode="traditional",
                    evaluation_timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating question {question_id}: {e}")
                error_result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    answer="",
                    sources=[],
                    response_time_ms=0,
                    token_usage=0,
                    retrieval_time_ms=0,
                    mode="traditional",
                    evaluation_timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def evaluate_agentic_rag(self, 
                                 questions: List[Dict[str, Any]],
                                 top_k: int = 5) -> List[EvaluationResult]:
        """Evaluate agentic RAG system."""
        results = []
        
        for question_data in questions:
            question_id = question_data["id"]
            question = question_data["question"]
            
            self.logger.info(f"Evaluating Agentic RAG - Question {question_id}: {question[:50]}...")
            
            try:
                start_time = time.time()
                
                # Run agentic workflow
                final_state = await self.agentic_rag.run(
                    query=question,
                    top_k=top_k,
                    enable_verification=True,
                    max_refinements=2
                )
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Extract answer and sources from final state
                answer = final_state.get("final_answer", "")
                retrieved_docs = final_state.get("retrieved_docs", [])
                
                # Convert retrieved docs to sources format
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "content": doc.get("content", ""),
                        "doc_id": doc.get("doc_id", ""),
                        "score": doc.get("score", 0.0),
                        "title": doc.get("metadata", {}).get("title", "Unknown")
                    })
                
                result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    sources=sources,
                    response_time_ms=response_time_ms,
                    token_usage=final_state.get("total_tokens", 0),
                    retrieval_time_ms=final_state.get("retrieval_time_ms", 0),
                    mode="agentic",
                    evaluation_timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating agentic question {question_id}: {e}")
                error_result = EvaluationResult(
                    question_id=question_id,
                    question=question,
                    answer="",
                    sources=[],
                    response_time_ms=0,
                    token_usage=0,
                    retrieval_time_ms=0,
                    mode="agentic",
                    evaluation_timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    def calculate_retrieval_metrics(self, 
                                  results: List[EvaluationResult], 
                                  golden_qa: List[Dict[str, Any]],
                                  k: int = 5) -> Dict[str, float]:
        """Calculate retrieval-specific metrics."""
        total_recall = 0.0
        total_ndcg = 0.0
        total_mrr = 0.0
        total_precision = 0.0
        valid_questions = 0
        
        for result in results:
            if result.error:
                continue
                
            # Find corresponding golden QA entry
            golden_entry = next(
                (q for q in golden_qa if q["id"] == result.question_id), 
                None
            )
            
            if not golden_entry:
                continue
                
            expected_topics = golden_entry.get("expected_topics", [])
            if not expected_topics:
                continue
            
            # Calculate metrics based on retrieved sources
            retrieved_docs = result.sources[:k]  # Top-k results
            
            if not retrieved_docs:
                continue
            
            # Recall@k: fraction of relevant docs retrieved
            relevant_retrieved = 0
            for doc in retrieved_docs:
                doc_content = doc.get("content", "").lower()
                # Check if any expected topic appears in the content
                for topic in expected_topics:
                    if topic.lower() in doc_content:
                        relevant_retrieved += 1
                        break
            
            recall_at_k = relevant_retrieved / min(len(expected_topics), k)
            total_recall += recall_at_k
            
            # Precision@k: fraction of retrieved docs that are relevant
            precision_at_k = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            total_precision += precision_at_k
            
            # MRR: Mean Reciprocal Rank - position of first relevant document
            mrr = 0.0
            for i, doc in enumerate(retrieved_docs):
                doc_content = doc.get("content", "").lower()
                for topic in expected_topics:
                    if topic.lower() in doc_content:
                        mrr = 1.0 / (i + 1)
                        break
                if mrr > 0:
                    break
            total_mrr += mrr
            
            # nDCG@k: Normalized Discounted Cumulative Gain
            # Create relevance scores based on topic matching
            relevance_scores = []
            for doc in retrieved_docs:
                doc_content = doc.get("content", "").lower()
                relevance = 0
                for topic in expected_topics:
                    if topic.lower() in doc_content:
                        relevance = 1
                        break
                relevance_scores.append(relevance)
            
            if any(relevance_scores):
                # Create ideal relevance (all 1s for available relevant docs)
                ideal_relevance = [1] * min(relevant_retrieved, k) + [0] * max(0, k - relevant_retrieved)
                if len(ideal_relevance) > len(relevance_scores):
                    ideal_relevance = ideal_relevance[:len(relevance_scores)]
                
                try:
                    ndcg = ndcg_score([ideal_relevance], [relevance_scores], k=k)
                    total_ndcg += ndcg
                except:
                    # Handle edge cases in NDCG calculation
                    total_ndcg += 0.0
            
            valid_questions += 1
        
        # Calculate averages
        if valid_questions == 0:
            return {
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "mrr_score": 0.0,
                "precision_at_k": 0.0,
                "map_score": 0.0
            }
        
        return {
            "recall_at_k": total_recall / valid_questions,
            "ndcg_at_k": total_ndcg / valid_questions,
            "mrr_score": total_mrr / valid_questions,
            "precision_at_k": total_precision / valid_questions,
            "map_score": total_precision / valid_questions  # MAP approximation
        }
    
    async def run_ragas_evaluation(self, 
                                 results: List[EvaluationResult]) -> List[EvaluationResult]:
        """Run RAGAS evaluation on the results."""
        if not results:
            return results
        
        # Filter out error results
        valid_results = [r for r in results if not r.error and r.answer and r.sources]
        
        if not valid_results:
            self.logger.warning("No valid results for RAGAS evaluation")
            return results
        
        # Prepare data for RAGAS
        questions = [r.question for r in valid_results]
        answers = [r.answer for r in valid_results]
        contexts = [[source["content"] for source in r.sources] for r in valid_results]
        
        # Create dataset for RAGAS
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts
        })
        
        try:
            self.logger.info("Running RAGAS evaluation...")
            
            # Run RAGAS evaluation
            ragas_result = evaluate(
                dataset=ragas_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy, 
                    context_precision,
                    context_recall
                ]
            )
            
            # Update results with RAGAS scores
            for i, result in enumerate(valid_results):
                if i < len(ragas_result):
                    result.faithfulness_score = ragas_result["faithfulness"][i] if i < len(ragas_result.get("faithfulness", [])) else None
                    result.answer_relevancy_score = ragas_result["answer_relevancy"][i] if i < len(ragas_result.get("answer_relevancy", [])) else None
                    result.context_precision_score = ragas_result["context_precision"][i] if i < len(ragas_result.get("context_precision", [])) else None
                    result.context_recall_score = ragas_result["context_recall"][i] if i < len(ragas_result.get("context_recall", [])) else None
            
            self.logger.info("RAGAS evaluation completed successfully")
            
        except Exception as e:
            self.logger.error(f"RAGAS evaluation failed: {e}")
            # Continue without RAGAS scores
        
        return results
    
    def save_results(self, 
                    results: List[EvaluationResult], 
                    retrieval_metrics: Dict[str, float],
                    output_path: Path):
        """Save evaluation results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to dictionaries
        results_data = [asdict(result) for result in results]
        
        # Create comprehensive results object
        full_results = {
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_questions": len(results),
                "successful_evaluations": len([r for r in results if not r.error]),
                "failed_evaluations": len([r for r in results if r.error]),
                "mode": results[0].mode if results else "unknown"
            },
            "retrieval_metrics": retrieval_metrics,
            "ragas_summary": self._calculate_ragas_summary(results),
            "performance_summary": self._calculate_performance_summary(results),
            "detailed_results": results_data
        }
        
        # Save JSON
        json_path = output_path / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        csv_path = output_path / f"evaluation_results_{timestamp}.csv"
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to {json_path} and {csv_path}")
        
        return json_path, csv_path
    
    def _calculate_ragas_summary(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate summary statistics for RAGAS metrics."""
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            return {}
        
        summary = {}
        
        # Calculate averages for each metric
        metrics = ["faithfulness_score", "answer_relevancy_score", 
                  "context_precision_score", "context_recall_score"]
        
        for metric in metrics:
            scores = [getattr(r, metric) for r in valid_results 
                     if getattr(r, metric) is not None]
            if scores:
                summary[f"avg_{metric}"] = np.mean(scores)
                summary[f"std_{metric}"] = np.std(scores)
                summary[f"min_{metric}"] = np.min(scores)
                summary[f"max_{metric}"] = np.max(scores)
        
        return summary
    
    def _calculate_performance_summary(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate performance summary statistics."""
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            return {}
        
        response_times = [r.response_time_ms for r in valid_results]
        token_usage = [r.token_usage for r in valid_results]
        retrieval_times = [r.retrieval_time_ms for r in valid_results]
        
        return {
            "avg_response_time_ms": np.mean(response_times),
            "p50_response_time_ms": np.percentile(response_times, 50),
            "p95_response_time_ms": np.percentile(response_times, 95),
            "avg_token_usage": np.mean(token_usage),
            "total_token_usage": np.sum(token_usage),
            "avg_retrieval_time_ms": np.mean(retrieval_times),
            "success_rate": len(valid_results) / len(results) if results else 0
        }


async def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation with RAGAS metrics")
    parser.add_argument("--golden_qa_path", type=str, default="../golden_qa.json",
                       help="Path to golden QA dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--run_traditional", action="store_true",
                       help="Run traditional RAG evaluation")
    parser.add_argument("--run_agentic", action="store_true", 
                       help="Run agentic RAG evaluation")
    parser.add_argument("--compare", action="store_true",
                       help="Generate comparison report")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load golden QA dataset
    with open(args.golden_qa_path, 'r') as f:
        golden_qa_data = json.load(f)
    
    questions = golden_qa_data["questions"]
    
    # Initialize RAG systems
    try:
        from rag_baseline import BaselineRAG
        from graph import AgenticRAG
        from retrieval import HybridRetriever
        
        retriever = HybridRetriever()
        baseline_rag = BaselineRAG(retriever=retriever)
        agentic_rag = AgenticRAG(retriever=retriever)
        
        evaluator = RAGEvaluator(
            baseline_rag=baseline_rag,
            agentic_rag=agentic_rag,
            retriever=retriever
        )
        
    except Exception as e:
        print(f"Failed to initialize RAG systems: {e}")
        return
    
    # Run evaluations
    if args.run_traditional or (not args.run_traditional and not args.run_agentic):
        print("Evaluating Traditional RAG...")
        traditional_results = await evaluator.evaluate_traditional_rag(questions, args.top_k)
        traditional_results = await evaluator.run_ragas_evaluation(traditional_results)
        traditional_retrieval_metrics = evaluator.calculate_retrieval_metrics(
            traditional_results, questions, args.top_k
        )
        
        traditional_json, traditional_csv = evaluator.save_results(
            traditional_results, traditional_retrieval_metrics, output_dir
        )
        print(f"Traditional RAG results saved to {traditional_json}")
    
    if args.run_agentic:
        print("Evaluating Agentic RAG...")
        agentic_results = await evaluator.evaluate_agentic_rag(questions, args.top_k)
        agentic_results = await evaluator.run_ragas_evaluation(agentic_results)
        agentic_retrieval_metrics = evaluator.calculate_retrieval_metrics(
            agentic_results, questions, args.top_k
        )
        
        agentic_json, agentic_csv = evaluator.save_results(
            agentic_results, agentic_retrieval_metrics, output_dir
        )
        print(f"Agentic RAG results saved to {agentic_json}")
    
    if args.compare and args.run_traditional and args.run_agentic:
        print("Generating comparison report...")
        # This would be implemented to compare the two result sets
        print("Comparison functionality to be implemented in frontend")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
