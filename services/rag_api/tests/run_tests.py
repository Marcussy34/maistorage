"""
Test runner for Phase 11 comprehensive test suite.

This script provides different test execution modes and reporting capabilities
for the MAI Storage RAG API test suite.
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TestRunner:
    """Comprehensive test runner for Phase 11 test suite."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.results = {}
    
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all unit tests."""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir / "unit"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.results["unit_tests"] = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
        else:
            print("❌ Unit tests failed")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results["unit_tests"]
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir / "integration"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.results["integration_tests"] = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results["integration_tests"]
    
    def run_edge_case_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all edge case tests."""
        print("🔺 Running Edge Case Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir / "edge_cases"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.results["edge_case_tests"] = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Edge case tests passed")
        else:
            print("❌ Edge case tests failed")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results["edge_case_tests"]
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all performance tests."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir / "performance"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            "-m", "not slow"  # Skip slow tests by default
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.results["performance_tests"] = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Performance tests passed")
        else:
            print("❌ Performance tests failed")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return self.results["performance_tests"]
    
    def run_all_tests(self, verbose: bool = False, include_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        print("🚀 Running Complete Test Suite...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all test categories
        self.run_unit_tests(verbose)
        self.run_integration_tests(verbose)
        self.run_edge_case_tests(verbose)
        self.run_performance_tests(verbose)
        
        total_time = time.time() - start_time
        
        # Calculate overall results
        all_passed = all(
            result.get("success", False) 
            for result in self.results.values()
        )
        
        print("\n" + "=" * 50)
        print("📊 Test Suite Summary")
        print("=" * 50)
        
        for test_type, result in self.results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"{test_type.replace('_', ' ').title()}: {status}")
        
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Phase 11 test suite complete.")
        else:
            print("\n💥 SOME TESTS FAILED! Please review the failures above.")
        
        return {
            "overall_success": all_passed,
            "total_time": total_time,
            "results": self.results
        }
    
    def run_coverage_report(self) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        print("📈 Running Tests with Coverage...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir),
            "--cov=../",  # Cover parent directory
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=70"  # Require 70% coverage
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        coverage_result = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Coverage requirements met")
        else:
            print("❌ Coverage requirements not met")
        
        print(result.stdout)
        
        return coverage_result
    
    def generate_test_report(self, output_file: str = "test_report.html"):
        """Generate HTML test report."""
        print(f"📄 Generating test report: {output_file}")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.base_dir),
            "--html=" + output_file,
            "--self-contained-html"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Test report generated: {output_file}")
        else:
            print("❌ Failed to generate test report")
        
        return result.returncode == 0


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Phase 11 Test Suite Runner")
    
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "edge_cases", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        help="Generate HTML report with specified filename"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow-running tests"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print("🧪 MAI Storage RAG API - Phase 11 Test Suite")
    print("=" * 50)
    
    try:
        if args.coverage:
            runner.run_coverage_report()
        elif args.report:
            runner.generate_test_report(args.report)
        else:
            if args.test_type == "unit":
                runner.run_unit_tests(args.verbose)
            elif args.test_type == "integration":
                runner.run_integration_tests(args.verbose)
            elif args.test_type == "edge_cases":
                runner.run_edge_case_tests(args.verbose)
            elif args.test_type == "performance":
                runner.run_performance_tests(args.verbose)
            else:  # all
                runner.run_all_tests(args.verbose, args.include_slow)
        
        # Print final summary
        if runner.results:
            failed_tests = [
                test_type for test_type, result in runner.results.items()
                if not result.get("success", False)
            ]
            
            if failed_tests:
                print(f"\n❌ Failed test categories: {', '.join(failed_tests)}")
                sys.exit(1)
            else:
                print("\n✅ All tests passed successfully!")
                sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
