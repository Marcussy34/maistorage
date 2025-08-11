# Phase 11 Test Suite - MAI Storage RAG API

This directory contains the comprehensive test suite for Phase 11 of the MAI Storage RAG API project. The test suite is designed to prevent regressions, cover edge cases, and ensure high code quality across all components.

## 📁 Test Organization

```
tests/
├── README.md                 # This file
├── conftest.py              # Global fixtures and configuration
├── run_tests.py             # Test runner script
├── unit/                    # Unit tests for individual components
│   ├── __init__.py
│   ├── test_text_splitters.py        # Text chunking and splitting tests
│   ├── test_citation_engine.py       # Sentence-level citation tests  
│   └── test_retrieval_algorithms.py  # RRF/MMR/BM25 algorithm tests
├── integration/             # End-to-end workflow tests
│   ├── __init__.py
│   └── test_rag_workflows.py         # Traditional vs Agentic RAG tests
├── edge_cases/              # Edge case and error condition tests
│   ├── __init__.py
│   └── test_edge_cases.py            # Ambiguous acronyms, typos, etc.
└── performance/             # Performance and load tests
    ├── __init__.py
    └── test_performance.py           # Latency, memory, concurrency tests
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

Test individual components and algorithms in isolation:

- **Text Splitters**: Document chunking, semantic splitting, edge cases
- **Citation Engine**: Sentence-level attribution, confidence scoring, text spans
- **Retrieval Algorithms**: RRF, MMR, BM25 scoring, utility functions

**Run with**: `make test-unit` or `pytest tests/unit/`

### Integration Tests (`tests/integration/`)

Test complete workflows and component interactions:

- **Traditional RAG**: Baseline query processing, happy path, error handling
- **Agentic RAG**: Multi-step workflows, refinement loops, trace generation
- **Comparison**: Traditional vs Agentic behavior differences
- **OOD Queries**: Out-of-distribution query handling

**Run with**: `make test-integration` or `pytest tests/integration/`

### Edge Case Tests (`tests/edge_cases/`)

Test challenging scenarios and error conditions:

- **Ambiguous Acronyms**: Context resolution, multiple meanings
- **Typos and Misspellings**: Fuzzy matching, correction handling
- **Conflicting Documents**: Information synthesis, uncertainty handling
- **Malformed Inputs**: Security, validation, graceful degradation
- **Unicode/International**: Multi-language support, special characters

**Run with**: `make test-edge-cases` or `pytest tests/edge_cases/`

### Performance Tests (`tests/performance/`)

Test performance characteristics and scalability:

- **Latency Testing**: P50/P95 response times, algorithm performance
- **Memory Usage**: Resource consumption under load
- **Concurrency**: Thread safety, concurrent request handling
- **Load Testing**: High-volume query processing
- **Stress Testing**: System limits and failure modes

**Run with**: `make test-performance` or `pytest tests/performance/`

## 🚀 Running Tests

### Quick Commands

```bash
# Run all tests
make test-comprehensive

# Run specific test categories
make test-unit
make test-integration  
make test-edge-cases
make test-performance

# Run with coverage
make test-coverage

# Run only fast tests (excludes slow performance tests)
make test-fast

# Generate HTML test report
make test-report
```

### Advanced Usage

```bash
# Use the test runner directly
cd services/rag_api
python tests/run_tests.py --help

# Run specific test types
python tests/run_tests.py --test-type unit --verbose
python tests/run_tests.py --test-type integration
python tests/run_tests.py --test-type all --coverage

# Run pytest directly with custom options
pytest tests/unit/ -v --tb=short --durations=10
pytest tests/ -k "test_rrf" --verbose
pytest tests/ -m "not slow" --tb=line
```

## 📊 Coverage and Quality Gates

### Coverage Requirements

- **Minimum Coverage**: 70% overall
- **Unit Tests**: 85%+ for core algorithms
- **Integration**: 60%+ for workflow paths
- **Critical Components**: 90%+ (RRF, MMR, citation engine)

### Quality Checks

The test suite includes automated quality checks:

- **Linting**: ruff for Python code quality
- **Formatting**: black for consistent code style
- **Type Checking**: mypy for type safety (optional)
- **Security**: bandit for security scanning (optional)

### Performance Thresholds

- **P50 Latency**: < 200ms (Traditional RAG), < 800ms (Agentic RAG)
- **P95 Latency**: < 500ms (Traditional RAG), < 1500ms (Agentic RAG)
- **Memory Usage**: < 200MB increase under normal load
- **Throughput**: > 10 requests/second for concurrent load

## 🔧 Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
export TESTING=true
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4o-mini
export EMBEDDING_MODEL=text-embedding-3-small
export QDRANT_URL=http://localhost:6333
```

### Pytest Configuration

Key pytest configuration options:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --tb=short
    --durations=10
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    edge_case: marks tests as edge case tests
```

### Mock Strategy

Tests use mocking extensively to ensure:

- **Isolation**: Unit tests don't depend on external services
- **Speed**: Fast execution without network calls
- **Reliability**: Consistent results regardless of external state
- **Coverage**: Test error conditions and edge cases

## 🎯 Test Fixtures and Data

### Global Fixtures (`conftest.py`)

- **sample_documents**: Realistic document examples
- **sample_retrieval_results**: Mock retrieval responses
- **sample_queries**: Various query types for testing
- **mock_embeddings**: Pre-generated embedding vectors
- **sample_citations**: Citation examples with confidence scores

### Temporary Resources

Tests automatically create and clean up:

- Temporary directories and files
- Mock databases and collections
- In-memory caches and indexes
- Isolated test environments

## 🏗️ CI/CD Integration

### GitHub Actions

The test suite is integrated with GitHub Actions (`.github/workflows/phase11-test-suite.yml`):

- **Parallel Execution**: Different test categories run in parallel
- **Service Dependencies**: Automatic Qdrant setup for integration tests
- **Artifact Collection**: Test results and coverage reports
- **Quality Gates**: Fail builds on test failures or coverage drops
- **Performance Monitoring**: Track performance regressions

### Test Stages

1. **Unit Tests**: Fast, isolated component tests
2. **Integration Tests**: Workflow and component interaction tests
3. **Edge Case Tests**: Error conditions and challenging scenarios
4. **Performance Tests**: Latency and resource usage validation
5. **Code Quality**: Linting, formatting, and coverage checks
6. **E2E Tests**: Full system integration (on push/schedule)

## 📈 Monitoring and Reporting

### Test Reports

- **JUnit XML**: Machine-readable test results
- **HTML Reports**: Human-readable test summaries
- **Coverage Reports**: Line and branch coverage analysis
- **Performance Metrics**: Latency and throughput measurements

### Continuous Monitoring

- **Regression Detection**: Catch performance and quality regressions
- **Trend Analysis**: Track test execution time and coverage trends
- **Failure Analysis**: Detailed logs and stack traces for debugging

## 🛠️ Development Workflow

### Adding New Tests

1. **Choose Category**: Determine if test is unit, integration, edge case, or performance
2. **Use Fixtures**: Leverage existing fixtures from `conftest.py`
3. **Follow Naming**: Use descriptive test names starting with `test_`
4. **Add Documentation**: Include docstrings explaining test purpose
5. **Mock Dependencies**: Mock external services and slow operations

### Test-Driven Development

1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
4. Add edge cases and error conditions
5. Ensure good test coverage

### Debugging Failed Tests

```bash
# Run single test with verbose output
pytest tests/unit/test_citation_engine.py::TestSentenceExtraction::test_simple_sentence_extraction -vv

# Run with pdb debugger
pytest tests/unit/test_citation_engine.py --pdb

# Run with detailed failure information
pytest tests/unit/test_citation_engine.py --tb=long

# Run with coverage to see untested lines
pytest tests/unit/test_citation_engine.py --cov=citer --cov-report=term-missing
```

## 📚 References

- **Phase 11 Requirements**: See `PLAN.md` lines 334-352
- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **GitHub Actions**: https://docs.github.com/en/actions

---

**Phase 11 Acceptance Criteria**: ✅ Complete

- ✅ Unit tests (splitters, RRF/MMR, attribution mapping)
- ✅ Integration/E2E tests (happy path, agent refine path, OOD)  
- ✅ CI pipeline (GitHub Actions)
- ✅ Edge cases (ambiguous acronyms, typos, conflicting docs)
- ✅ Load smoke tests (p50/p95 latency, memory)
- ✅ Coverage on critical paths
- ✅ Reproducible failures and test organization
