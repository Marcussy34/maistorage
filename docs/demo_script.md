# MAI Storage - Demo Script

**Version**: Phase 12 Complete  
**Last Updated**: January 2025  
**Demo Duration**: 15-20 minutes  
**Audience**: Technical stakeholders, decision makers, end users

## Demo Overview

This script provides a comprehensive demonstration of the MAI Storage Agentic RAG system, showcasing key differentiators between Traditional and Agentic RAG approaches, real-time trace visualization, sentence-level citations, and evaluation capabilities.

## Pre-Demo Setup Checklist

### Technical Prerequisites
- [ ] System running: `make dev` (all services healthy)
- [ ] Web app accessible: http://localhost:3000
- [ ] Sample documents ingested: `make ingest`
- [ ] Demo queries prepared (see sections below)
- [ ] Backup plan: screenshots/recordings ready

### Environment Verification
```bash
# Quick health check
make health

# Expected output:
# ✅ Web App (http://localhost:3000)
# ✅ API Health (http://localhost:8000/health)  
# ✅ Qdrant (http://localhost:6333)
# ✅ Sample data indexed (documents: 5+)
```

### Demo Data Validation
```bash
# Verify sample documents are indexed
curl http://localhost:8000/stats

# Expected response should show:
# - indexed_documents: > 0
# - collections: ["mai_storage_vectors"]
# - total_vectors: > 0
```

---

## Demo Script

### Introduction (2 minutes)

#### Opening Statement
"Welcome to the MAI Storage demonstration. Today I'll show you an advanced **Agentic RAG system** that goes beyond traditional retrieval-augmented generation to provide intelligent, multi-step reasoning for complex queries."

#### System Overview
"MAI Storage combines:
- **Traditional RAG**: Fast, single-pass query processing
- **Agentic RAG**: Multi-step reasoning with planning, verification, and refinement
- **Real-time Visualization**: Complete workflow transparency
- **Sentence-level Citations**: Precise source attribution
- **Comprehensive Evaluation**: Quality metrics and performance benchmarks"

#### Demo Agenda
1. **Traditional RAG Baseline** (3 minutes)
2. **Agentic RAG Multi-step Workflow** (5 minutes)  
3. **Citation System & Source Verification** (3 minutes)
4. **Agent Trace & Workflow Transparency** (3 minutes)
5. **Evaluation & Quality Metrics** (3 minutes)
6. **Q&A and Technical Discussion** (5 minutes)

---

### Part 1: Traditional RAG Baseline (3 minutes)

#### Navigate to Chat Interface
1. Open: http://localhost:3000/chat
2. **Point out**: Clean, modern interface with dark mode support
3. **Show**: Mode toggle currently set to "Traditional"

#### Simple Query Demonstration
**Query**: `"What is machine learning?"`

**Narration**: "Let's start with a straightforward question using Traditional RAG. This represents the standard approach most RAG systems use today."

**Expected Response**: ~30-45 seconds
- Direct, comprehensive answer
- Chunk-level citations (numbered references)
- Basic source information

**Key Points to Highlight**:
- **Speed**: Fast response time (< 500ms typical)
- **Simplicity**: Single-pass retrieval and generation
- **Citations**: Document-level source attribution
- **Use Case**: Excellent for straightforward factual questions

#### Show Citation Functionality
1. **Hover** over numbered citation (¹ ² ³)
2. **Demonstrate**: Citation hover cards with source preview
3. **Click** to open Context Panel showing full source documents

**Narration**: "Notice the numbered citations that provide immediate source verification. This builds trust and allows users to validate information directly."

---

### Part 2: Agentic RAG Multi-step Workflow (5 minutes)

#### Mode Switch
1. **Toggle** to "Agentic" mode
2. **Explain**: "Now we'll switch to our Agentic system that uses multi-step reasoning for complex queries."

#### Complex Query Demonstration
**Query**: `"Compare the advantages and disadvantages of different machine learning approaches for natural language processing tasks"`

**Narration**: "This is exactly the type of complex, multi-faceted question where Agentic RAG excels. Watch how the system breaks this down into manageable steps."

**Expected Workflow** (~60-90 seconds):
1. **Planner Phase**: Query decomposition and strategy
2. **Retrieval Phase**: Multi-source information gathering  
3. **Synthesis Phase**: Comprehensive response generation
4. **Verification Phase**: Quality validation and potential refinement

#### Real-time Trace Visualization
**Key Points to Highlight**:

##### 1. Planner Node
- **Show**: "Planning strategy..." indicator
- **Explain**: "The planner analyzes the query complexity and creates a retrieval strategy"
- **Point out**: Sub-queries being generated (visible in trace panel)

##### 2. Retrieval Node  
- **Show**: "Retrieving information..." with progress
- **Explain**: "Hybrid search combining dense vectors and BM25 sparse retrieval"
- **Highlight**: Multiple search strategies for comprehensive coverage

##### 3. Synthesis Node
- **Show**: Streaming response generation
- **Explain**: "Synthesizing information from multiple sources into a coherent response"
- **Point out**: Higher quality, more comprehensive answers

##### 4. Verification Node
- **Show**: "Verifying quality..." step
- **Explain**: "AI verifier checks faithfulness, coverage, and quality"
- **Demonstrate**: Either approval or refinement trigger

#### Quality Comparison
**Narration**: "Notice how the Agentic response is more comprehensive, better structured, and provides deeper analysis compared to what a Traditional RAG would produce for the same complex query."

---

### Part 3: Citation System & Source Verification (3 minutes)

#### Sentence-Level Citations
**Highlight**: "MAI Storage provides sentence-level citations, not just document-level references."

**Demonstration**:
1. **Point to** specific sentences with citations
2. **Hover** over citations to show precise source spans
3. **Explain**: Each claim is attributed to specific text passages

#### Citation Quality Features
**Show**:
- **High Confidence**: Regular numbered citations (¹ ² ³)
- **Low Confidence**: Warning indicators (⚠️) when certainty is low
- **Source Diversity**: Multiple supporting documents per claim

#### Source Verification Workflow
**Demonstrate**:
1. **Click** citation to open source document
2. **Show** highlighted text spans supporting the claim
3. **Navigate** between multiple supporting sources
4. **Explain** metadata: document titles, pages, relevance scores

**Narration**: "This level of citation granularity is crucial for enterprise applications where fact verification and source transparency are essential."

---

### Part 4: Agent Trace & Workflow Transparency (3 minutes)

#### Trace Panel Deep Dive
**Open** the Agent Trace panel (if not already visible)

**Demonstrate**:
1. **Timeline View**: Step-by-step execution sequence
2. **Performance Metrics**: Timing for each workflow step
3. **Token Usage**: Detailed cost breakdown per step
4. **Verification Results**: Quality scores and decisions

#### Step-by-Step Analysis
**Go through each trace event**:

##### Planner Step
- **Show**: Query analysis and decomposition
- **Highlight**: Sub-queries and key concepts identified
- **Timing**: ~500ms typical

##### Retrieval Step  
- **Show**: Search strategy execution
- **Highlight**: Number of candidates and final results
- **Timing**: ~800ms typical

##### Synthesis Step
- **Show**: Response generation progress
- **Highlight**: Token streaming and source integration
- **Timing**: ~1500ms typical

##### Verification Step
- **Show**: Quality assessment scores
- **Highlight**: Refinement decision (passed/needs improvement)
- **Timing**: ~400ms typical

#### Transparency Benefits
**Narration**: "This complete workflow visibility enables:
- **Debugging**: Understand exactly what the system is doing
- **Optimization**: Identify bottlenecks and improve performance  
- **Trust**: Full transparency builds user confidence
- **Compliance**: Audit trail for regulated industries"

---

### Part 5: Evaluation & Quality Metrics (3 minutes)

#### Navigate to Evaluation Dashboard
1. **Open**: http://localhost:3000/eval
2. **Show**: Comprehensive evaluation interface

#### Metrics Overview
**Demonstrate** key metrics categories:

##### RAGAS Quality Metrics
- **Faithfulness**: ~0.87 (factual accuracy)
- **Answer Relevancy**: ~0.82 (question alignment)
- **Context Precision**: ~0.78 (relevant sources)
- **Context Recall**: ~0.74 (comprehensive coverage)

##### Retrieval Performance
- **Recall@10**: ~0.86 (relevant document retrieval)
- **nDCG@10**: ~0.76 (ranking quality)
- **MRR**: ~0.73 (first relevant result position)

##### Performance Benchmarks
- **Traditional RAG**: P95 < 800ms
- **Agentic RAG**: P95 < 2500ms
- **Memory Usage**: < 2GB sustained
- **Cache Hit Rate**: ~85%

#### Comparison Analysis
**Show** Traditional vs Agentic comparison:

**Quality Improvements**:
- **Faithfulness**: +4.6% (agentic better fact-checking)
- **Answer Relevancy**: +7.3% (better question understanding)
- **Context Precision**: +5.1% (better source selection)
- **Context Recall**: +6.8% (more comprehensive)

**Performance Trade-offs**:
- **Response Time**: +221% (more thorough but slower)
- **Token Usage**: +74% (more comprehensive processing)
- **Cost**: +73% (higher quality comes with cost)

#### Evaluation Insights
**Narration**: "The evaluation framework shows clear trade-offs:
- **Agentic RAG** excels at complex, analytical questions
- **Traditional RAG** is better for simple, speed-critical queries
- **Cost-effectiveness** depends on use case requirements"

---

### Part 6: Advanced Features Demo (Optional - if time permits)

#### Custom Query Types
**Demonstrate** different query complexities:

##### Ambiguous Query
**Query**: `"What does AI mean in different contexts?"`
**Show**: How the system handles disambiguation

##### Multi-hop Question  
**Query**: `"How does climate change affect machine learning model training and deployment?"`
**Show**: Complex reasoning across domains

##### Out-of-Domain Query
**Query**: `"What is the recipe for chocolate cake?"`
**Show**: Graceful handling of unsupported queries

#### Performance Monitoring
**Show** real-time performance metrics:
- Request latency distributions
- Token usage trends
- Cache hit rates
- Error rates and types

---

## Q&A Session Preparation

### Common Questions & Answers

#### Technical Questions

**Q: "How does this compare to ChatGPT or other AI assistants?"**
**A**: "MAI Storage is designed for enterprise knowledge retrieval with several key differences:
- **Source Control**: All answers are grounded in your documents
- **Transparency**: Complete workflow visibility and citations
- **Customization**: Tuned for your specific domain and data
- **Privacy**: Runs on your infrastructure with your data"

**Q: "What happens if the knowledge base doesn't contain relevant information?"**
**A**: "The system handles this gracefully:
- **Traditional RAG**: Returns low-confidence response with warning
- **Agentic RAG**: Verification step catches insufficient coverage
- **Both modes**: Clear indicators when information is limited
- **Fallback**: Suggests query refinement or additional sources"

**Q: "How accurate are the citations and how can we trust them?"**
**A**: "Citation accuracy is measured through several mechanisms:
- **Confidence Scoring**: Each citation has a confidence score
- **Warning Indicators**: ⚠️ symbols for low-confidence claims
- **Source Verification**: Direct links to exact text passages
- **Evaluation Metrics**: Continuous monitoring of citation quality"

#### Business Questions

**Q: "What are the cost implications compared to traditional search?"**
**A**: "Cost structure includes:
- **Infrastructure**: Modest hosting costs for vector database
- **API Costs**: OpenAI tokens (~$0.05-0.08 per complex query)
- **Trade-off**: Higher per-query cost but dramatically reduced time-to-insight
- **ROI**: Faster research, better decisions, reduced manual effort"

**Q: "How long does implementation take?"**
**A**: "Implementation timeline:
- **Basic Setup**: 1-2 days for standard deployment
- **Data Ingestion**: Depends on corpus size (hours to days)
- **Customization**: 1-2 weeks for domain-specific optimization
- **Production**: 2-4 weeks including testing and integration"

**Q: "What about data security and privacy?"**
**A**: "Security features:
- **On-premise Deployment**: Data never leaves your infrastructure
- **API Key Management**: Secure OpenAI API usage
- **Access Controls**: User authentication and authorization
- **Audit Logging**: Complete request and response tracking"

#### Integration Questions

**Q: "How does this integrate with existing systems?"**
**A**: "Integration options:
- **REST API**: Standard HTTP endpoints for any application
- **Web Interface**: Ready-to-use chat interface
- **SDK Support**: Python and JavaScript client libraries
- **Webhook Support**: Real-time notifications and updates"

**Q: "Can this work with our existing documents and databases?"**
**A**: "Supported integrations:
- **Document Formats**: PDF, Word, Markdown, HTML, plain text
- **Data Sources**: File systems, SharePoint, Confluence, databases
- **Connectors**: Custom ingestion pipelines for any data source
- **Real-time Updates**: Incremental indexing for live data"

---

## Backup Plans

### Technical Issues

#### If System is Unresponsive
1. **Have screenshots** of all key features ready
2. **Pre-recorded demo video** as complete fallback
3. **Local examples** of typical responses and traces

#### If Network Issues
1. **Offline slide deck** with key screenshots
2. **Local data export** showing evaluation results
3. **Architecture diagrams** and system overview

#### If Performance Issues
1. **Simpler queries** that respond faster
2. **Pre-computed examples** showing typical results
3. **Focus on evaluation results** rather than live demo

### Content Adaptations

#### For Technical Audience
- **Deep dive** into architecture details
- **Performance optimization** discussions
- **Implementation specifics** and customization options

#### For Business Audience
- **Focus on ROI** and business value
- **Use case examples** relevant to their industry
- **Cost-benefit analysis** and competitive advantages

#### For Mixed Audience
- **Start with business value**, then dive into technical details
- **Interactive elements** to gauge interest level
- **Flexible pacing** based on audience engagement

---

## Post-Demo Follow-up

### Immediate Actions
1. **Share demo recording** and screenshots
2. **Provide access** to live demo environment
3. **Send technical documentation** and setup guides
4. **Schedule follow-up** for deeper technical discussions

### Technical Documentation Package
- **ARCHITECTURE.md**: Complete system design
- **USAGE.md**: Setup and configuration guide
- **EVAL.md**: Evaluation framework details
- **API Documentation**: Complete endpoint reference

### Next Steps Framework
1. **Proof of Concept**: 2-week evaluation with their data
2. **Pilot Implementation**: Department-specific deployment
3. **Production Rollout**: Full enterprise deployment
4. **Ongoing Support**: Training, optimization, and maintenance

---

This demo script provides a comprehensive showcase of MAI Storage's capabilities while maintaining flexibility for different audiences and technical environments.
