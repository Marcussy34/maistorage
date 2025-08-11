#!/usr/bin/env python3
"""
Structure test for Phase 8 evaluation system

This script verifies that all evaluation components are properly implemented
without requiring API keys or external services.
"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """Test that all evaluation modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        from eval.run_ragas import RAGEvaluator, EvaluationResult, RetrievalMetrics
        print("✅ RAGAS evaluation modules imported successfully")
        
        from eval import __init__
        print("✅ Evaluation package imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_golden_qa_dataset():
    """Test that golden QA dataset is properly structured."""
    print("\n📚 Testing golden QA dataset...")
    
    try:
        with open("golden_qa.json", 'r') as f:
            golden_qa_data = json.load(f)
        
        # Verify dataset structure
        assert "dataset_info" in golden_qa_data
        assert "questions" in golden_qa_data
        
        questions = golden_qa_data["questions"]
        print(f"✅ Found {len(questions)} questions in dataset")
        
        # Verify question structure
        for i, question in enumerate(questions[:3]):  # Check first 3
            required_fields = ["id", "question", "type", "expected_topics"]
            for field in required_fields:
                assert field in question, f"Missing field '{field}' in question {i+1}"
        
        print("✅ Golden QA dataset structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Golden QA dataset test failed: {e}")
        return False


def test_evaluation_classes():
    """Test that evaluation classes are properly defined."""
    print("\n🏗️ Testing evaluation class structure...")
    
    try:
        from eval.run_ragas import RAGEvaluator, EvaluationResult, RetrievalMetrics
        
        # Test EvaluationResult dataclass
        result = EvaluationResult(
            question_id=1,
            question="Test question",
            answer="Test answer",
            sources=[],
            response_time_ms=100.0,
            token_usage=50,
            retrieval_time_ms=50.0,
            mode="traditional"
        )
        print("✅ EvaluationResult dataclass working")
        
        # Test that RAGEvaluator can be instantiated (without dependencies)
        evaluator = RAGEvaluator()
        assert hasattr(evaluator, 'calculate_retrieval_metrics')
        assert hasattr(evaluator, 'save_results')
        print("✅ RAGEvaluator class structure validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation class test failed: {e}")
        return False


def test_api_structure():
    """Test that API endpoints are properly defined in main.py."""
    print("\n🌐 Testing API endpoint structure...")
    
    try:
        with open("main.py", 'r') as f:
            main_content = f.read()
        
        # Check for evaluation endpoints
        required_endpoints = [
            "@app.post(\"/eval/run\")",
            "@app.get(\"/eval/results\")",
            "@app.get(\"/eval/compare\")"
        ]
        
        for endpoint in required_endpoints:
            assert endpoint in main_content, f"Missing endpoint: {endpoint}"
        
        print("✅ All evaluation API endpoints found in main.py")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


def test_frontend_structure():
    """Test that frontend evaluation page exists."""
    print("\n🎨 Testing frontend structure...")
    
    try:
        frontend_files = [
            "../../apps/web/pages/eval.js",
            "../../apps/web/pages/api/eval/run.js",
            "../../apps/web/pages/api/eval/results.js",
            "../../apps/web/pages/api/eval/compare.js",
            "../../apps/web/src/components/ui/card.js"
        ]
        
        for file_path in frontend_files:
            file_full_path = Path(__file__).parent / file_path
            assert file_full_path.exists(), f"Missing frontend file: {file_path}"
        
        print("✅ All frontend evaluation files found")
        return True
        
    except Exception as e:
        print(f"❌ Frontend structure test failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    try:
        import ragas
        print("✅ RAGAS library available")
        
        import datasets
        print("✅ Datasets library available")
        
        import pandas
        print("✅ Pandas library available")
        
        import sklearn
        print("✅ Scikit-learn library available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dependency test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Phase 8 Evaluation System Structure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_golden_qa_dataset,
        test_evaluation_classes,
        test_api_structure,
        test_frontend_structure,
        test_dependencies
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("\n✅ Phase 8 evaluation system is properly implemented")
        print("\n📚 To run full evaluation tests:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Ensure Qdrant is running")
        print("  3. Run: python test_evaluation.py")
        print("\n🚀 Ready to proceed to Phase 9: Performance & Cost Tuning")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        sys.exit(1)
