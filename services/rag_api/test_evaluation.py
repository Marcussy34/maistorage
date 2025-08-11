#!/usr/bin/env python3
"""
Quick test script for Phase 8 evaluation system

This script verifies that the evaluation harness works correctly
by running a minimal test with a subset of questions.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from eval.run_ragas import RAGEvaluator
from rag_baseline import BaselineRAG
from graph import AgenticRAG
from retrieval import HybridRetriever


async def test_evaluation_system():
    """Test the evaluation system with a minimal example."""
    print("🧪 Testing Phase 8 Evaluation System...")
    
    try:
        # Load golden QA dataset
        with open("golden_qa.json", 'r') as f:
            golden_qa_data = json.load(f)
        
        # Use only first 2 questions for quick test
        test_questions = golden_qa_data["questions"][:2]
        print(f"📝 Testing with {len(test_questions)} questions")
        
        # Initialize RAG systems
        print("🔧 Initializing RAG systems...")
        retriever = HybridRetriever()
        baseline_rag = BaselineRAG(retriever=retriever)
        agentic_rag = AgenticRAG(retriever=retriever)
        
        evaluator = RAGEvaluator(
            baseline_rag=baseline_rag,
            agentic_rag=agentic_rag,
            retriever=retriever
        )
        
        # Test Traditional RAG evaluation
        print("\n🔍 Testing Traditional RAG evaluation...")
        traditional_results = await evaluator.evaluate_traditional_rag(test_questions, top_k=3)
        
        print(f"✅ Traditional RAG: {len(traditional_results)} results")
        for result in traditional_results:
            if result.error:
                print(f"❌ Question {result.question_id}: {result.error}")
            else:
                print(f"✅ Question {result.question_id}: {len(result.sources)} sources, {result.response_time_ms:.0f}ms")
        
        # Test retrieval metrics calculation
        print("\n📊 Testing retrieval metrics...")
        retrieval_metrics = evaluator.calculate_retrieval_metrics(
            traditional_results, test_questions, k=3
        )
        
        print("📈 Retrieval Metrics:")
        for metric, value in retrieval_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Test RAGAS evaluation (commented out to avoid slow LLM calls)
        # print("\n🎯 Testing RAGAS evaluation...")
        # ragas_results = await evaluator.run_ragas_evaluation(traditional_results)
        # print(f"✅ RAGAS evaluation completed")
        
        print("\n🎉 Evaluation system test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_endpoints():
    """Test that evaluation API endpoints are responsive."""
    print("\n🌐 Testing API endpoints...")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test evaluation results endpoint
            response = await client.get(f"{base_url}/eval/results")
            if response.status_code == 200:
                print("✅ Evaluation results endpoint working")
            else:
                print(f"❌ Evaluation results endpoint failed: {response.status_code}")
                return False
            
            print("🎉 API endpoints test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("💡 Make sure FastAPI server is running: uvicorn main:app --reload")
        return False


if __name__ == "__main__":
    print("🚀 Phase 8 Evaluation System Test Suite")
    print("=" * 50)
    
    # Test evaluation system
    success = asyncio.run(test_evaluation_system())
    
    if success:
        # Test API endpoints
        api_success = asyncio.run(test_api_endpoints())
        
        if api_success:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED - Phase 8 evaluation system is working correctly!")
            print("\n📚 Next steps:")
            print("  1. Visit http://localhost:3000/eval to see the dashboard")
            print("  2. Run full evaluation: POST /eval/run with mode='both'")
            print("  3. Proceed to Phase 9: Performance & Cost Tuning")
        else:
            print("\n⚠️  Evaluation core works, but API endpoints need attention")
            sys.exit(1)
    else:
        print("\n❌ EVALUATION SYSTEM TEST FAILED")
        sys.exit(1)
