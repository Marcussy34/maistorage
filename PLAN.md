# PLAN.md — Phased Execution Plan (Agentic RAG + Next.js)

This plan is organized into **clear, sequential phases** with objectives, tasks, deliverables, acceptance criteria, inputs/dependencies, and estimated durations. Completing each phase and passing its exit criteria will fulfill the assignment requirements, including the demo-ready **Next.js** app, **Agentic RAG** loop, **citations**, **evaluation**, and **optimizations**.

---

## Models & Configuration (Explicit)

- **Answering LLM (default):** **OpenAI `gpt-5-mini`** (fast, cost-efficient; used for baseline answering, planner, verifier).  
  - Swappable via `OPENAI_MODEL=gpt-5-mini`.
- **Embeddings (retrieval):** **OpenAI `text-embedding-3-small`** for chunk/query embeddings.  
  - Swappable via `EMBEDDING_MODEL=text-embedding-3-small`.
- **Reranker (cross-encoder):** **`bge-reranker-v2`** (open-source) for reranking top dense+BM25 candidates.
- **Fallback LLM (offline/dev):** `Llama-3.1-8B-Instruct` via Ollama (optional).
- **Env vars:** `OPENAI_API_KEY`, `OPENAI_BASE_URL` (if proxying), `OPENAI_MODEL`, `EMBEDDING_MODEL`.

> Prompt templates (system / planner / verifier) live in `/services/rag_api/prompts/` and reference the model names above.

---

## Table of Contents

- [Phase 0 — Repo Bootstrap & Standards](#phase-0--repo-bootstrap--standards)
- [Phase 1 — Ingestion & Indexer MVP](#phase-1--ingestion--indexer-mvp)
- [Phase 2 — Retrieval Core (Hybrid + Rerank)](#phase-2--retrieval-core-hybrid--rerank)
- [Phase 2.5 — LLM Setup & Prompt Templates](#phase-25--llm-setup--prompt-templates)
- [Phase 3 — Baseline RAG (Traditional)](#phase-3--baseline-rag-traditional)
- [Phase 4 — Next.js Frontend (Streaming Shell)](#phase-4--nextjs-frontend-streaming-shell)
- [Phase 5 — Agentic Loop (LangGraph) + Streaming](#phase-5--agentic-loop-langgraph--streaming)
- [Phase 6 — Citations Engine (Per-Sentence Attribution)](#phase-6--citations-engine-per-sentence-attribution)
- [Phase 7 — Frontend Trace & Citations UX](#phase-7--frontend-trace--citations-ux)
- [Phase 8 — Evaluation Harness (RAGAS + Retrieval Metrics)](#phase-8--evaluation-harness-ragas--retrieval-metrics)
- [Phase 9 — Performance & Cost Tuning](#phase-9--performance--cost-tuning)
- [Phase 10 — Hardening, DX & Deployment](#phase-10--hardening-dx--deployment)
- [Phase 11 — Test Suite & Edge Cases](#phase-11--test-suite--edge-cases)
- [Phase 12 — Documentation & Demo](#phase-12--documentation--demo)
- [Requirement Coverage Map](#requirement-coverage-map)
- [Quick Start Commands](#quick-start-commands)

---

## Phase 0 — Repo Bootstrap & Standards

**Objective**  
Create a clean monorepo and shared conventions to accelerate all downstream work.

**Deliverables**
- Monorepo scaffold:
  ```text
  /apps/web (Next.js 14)
  /services/rag_api (FastAPI + LangGraph)
  /services/indexer
  /infrastructure (docker-compose, .env.example)
  ```
- Tooling: `ruff`, `black`, `isort`, `eslint`, `prettier`, `pre-commit`, Makefile.
- Basic README + `.env.example` with **`OPENAI_API_KEY`, `OPENAI_MODEL=gpt-5-mini`, `EMBEDDING_MODEL=text-embedding-3-small`**.

**Tasks**
- Initialize repos, package managers (`pnpm` for web, `uv`/`pip` for py).
- Wire `docker-compose` for **Qdrant** (and **Elasticsearch** optional).
- Add Git hooks and format/lint configs.

**Acceptance (Exit)**
- `pnpm dev` runs a placeholder Next.js page.
- `uvicorn main:app --reload` returns a hello from FastAPI.
- `docker compose up -d qdrant` healthy.
- README quickstart verified on a clean machine.

**Inputs/Dependencies**: None.  
**Est. Duration**: 0.5–1 day.

---

## Phase 1 — Ingestion & Indexer MVP

**Objective**  
Load raw docs → split into semantic chunks → embed → upsert to Qdrant.

**Deliverables**
- `/services/indexer/ingest.py`, `/services/indexer/embed.py`, `config.yaml`.
- A sample corpus indexed with coherent metadata (doc_id, title, url, section/page, timestamp).

**Tasks**
- Implement loaders (PDF/MD/HTML). Handle OCR or skip as needed.
- Semantic chunking (200–500 tokens, 15–20% overlap).
- **Embeddings:** call **OpenAI `text-embedding-3-small`** to embed chunks/queries; store full text + metadata.
- CLI: `python ingest.py --path ./data`.

**Acceptance (Exit)**
- Qdrant collection exists with expected vector count.
- Spot-check stored chunks for correct text & metadata.

**Inputs/Dependencies**: Phase 0 complete; sample data available.  
**Est. Duration**: 0.5–1 day.

---

## Phase 2 — Retrieval Core (Hybrid + Rerank)

**Objective**  
Return high-quality, diverse chunks via hybrid retrieval + reranking.

**Deliverables**
- `/services/rag_api/retrieval.py`, `/services/rag_api/models.py`, `/services/rag_api/tools.py`.
- Endpoint `POST /retrieve` → ranked chunks with scores.

**Tasks**
- Dense search (Qdrant vectors from **`text-embedding-3-small`**) + BM25 (Elasticsearch or in-proc BM25).
- **RRF** to fuse lists; **cross-encoder** rerank (`bge-reranker-v2`) top-100 → top-15.
- **MMR** to improve diversity; configurable top-k.

**Acceptance (Exit)**
- Manual queries return relevant/diverse chunks.
- Latency recorded (< 400–800 ms retrieval path on dev machine).
- Unit tests for RRF/MMR correctness.

**Inputs/Dependencies**: Phase 1 indexed corpus.  
**Est. Duration**: 1 day.

---

## Phase 2.5 — LLM Setup & Prompt Templates

**Objective**  
Explicitly configure the LLM and prompt scaffolding for all chains/agents.

**Deliverables**
- `/services/rag_api/models.py` with **`gpt-5-mini`** client and swappable interface.
- `/services/rag_api/prompts/` with **baseline**, **planner**, **verifier** templates.
- Smoke tests: model responds without retrieval.

**Tasks**
- Implement OpenAI client; read `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-5-mini` from env.
- Create standard prompts and variables (style, tone, citation placeholders).
- Add a tiny harness to test raw generation latency and token usage.

**Acceptance (Exit)**
- `gpt-5-mini` returns coherent text for a simple prompt.
- Prompts load and render without errors.
- Latency and token logs captured.

**Inputs/Dependencies**: Phase 0 complete.  
**Est. Duration**: 0.5 day.

---

## Phase 3 — Baseline RAG (Traditional)

**Objective**  
Establish a one-pass RAG baseline and a gold QA set.

**Deliverables**
- `/services/rag_api/rag_baseline.py` (`POST /rag` non-stream).
- Golden QA JSON (10–20 items) to drive evals later.

**Tasks**
- Query → retrieve top-k → pack context → **LLM (`gpt-5-mini`) generate**.
- Return answer + **chunk-level citations**.

**Acceptance (Exit)**
- Baseline answers coherent for straightforward questions.
- Citations list shows supporting chunks.
- Golden QA file created.

**Inputs/Dependencies**: Phase 2 retrieval, Phase 2.5 LLM.  
**Est. Duration**: 0.5–1 day.

---

## Phase 4 — Next.js Frontend (Streaming Shell, Pages Router, JS)

**Objective**  
A usable chat UI with streaming tokens and API proxy.

**Deliverables**  
- `/apps/web/pages/chat.js` (messages + input)  
- `/apps/web/pages/api/chat/stream.js` (proxy to FastAPI)  
- `components/ChatInput.js`, `components/ChatStream.js` (NDJSON parsing)  

**Tasks**  
- Tailwind + shadcn/ui setup; dark mode  
- Implement NDJSON client parser (fetch + ReadableStream)  
- Wire environment for API URL via API route proxy  

**Acceptance (Exit)**  
- User can send a message; assistant streams tokens from API  
- Basic error handling; resend works  

**Inputs/Dependencies**: Phase 3 API responses or mock stream  
**Est. Duration**: 1 day  

---

## Phase 5 — Agentic Loop (LangGraph) + Streaming

**Objective**  
Planner → Retrieve → Synthesize → **Verifier** → Refine → Finalize; stream `trace` events.

**Deliverables**
- `/services/rag_api/graph.py`, `/services/rag_api/verify.py`.
- `POST /chat/stream?agentic=1` emitting NDJSON events: `token`, `trace`, `sources`, `metrics`, `done`.

**Tasks**
- Define LangGraph state machine and nodes; **use `gpt-5-mini`** for **Planner**, **Synthesis**, **Verifier**.
- Add query decomposition, alias expansion when needed.
- Implement verifier (faithfulness/coverage) with retry loop.

**Acceptance (Exit)**
- Multi-hop/ambiguous queries trigger at least one refine pass.
- Stable `trace` event schema logged and viewable.

**Inputs/Dependencies**: Phase 2 retrieval; Phase 4 streaming UI; Phase 2.5 LLM.  
**Est. Duration**: 1–1.5 days.

---

## Phase 6 — Citations Engine (Per-Sentence Attribution)

**Objective**  
Upgrade from chunk-level to **sentence-level** citations with confidence.

**Deliverables**
- `/services/rag_api/citer.py`.
- Extended `sources` payload: `sentences[]` + `sources[]` with spans & scores.

**Tasks**
- Post-hoc attribution: sentence embeddings → best supporting chunk (cosine sim).
- (Optional) Use `gpt-5-mini` to rephrase sentences for better alignment (no new claims).
- Mark low-confidence with ⚠️; optional auto-refine hook.

**Acceptance (Exit)**
- Each response sentence maps to ≥1 citation or shows ⚠️.
- Snippet spans align with the sentence’s claim.

**Inputs/Dependencies**: Phase 5 agent outputs.  
**Est. Duration**: 0.5–1 day.

---

## Phase 7 — Frontend Trace & Citations UX

**Objective**  
Make reasoning and evidence **auditable** in the UI.

**Deliverables**
- `AgentTrace` panel (timeline of planner/retriever/verifier steps).
- `SourceBadge` (¹ ² ³) with HoverCard snippets & open-source links.
- Context panel with top-k chunks and highlights.
- Mode toggle: **Traditional vs Agentic**.

**Tasks**
- Render `trace` events; map sentences ↔ citations.
- Add metrics chips (retrieval ms, LLM ms, tokens).

**Acceptance (Exit)**
- Users can inspect reasoning and verify each claim via citations.
- Visible behavior difference between Traditional and Agentic modes.

**Inputs/Dependencies**: Phase 6 payload shape finalized.  
**Est. Duration**: 1 day.

---

## Phase 8 — Evaluation Harness (RAGAS + Retrieval Metrics)

**Objective**  
Quantify quality and create a repeatable evaluation loop.

**Deliverables**
- `/services/rag_api/eval/run_ragas.py` (stores results JSON/CSV).
- `/apps/web/app/eval/page.tsx` displays metrics & comparisons.

**Tasks**
- Compute RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall).
- Compute retrieval metrics (Recall@k, nDCG, MRR).
- Show Traditional vs Agentic side-by-side.

**Acceptance (Exit)**
- Reproducible eval run passes thresholds (configurable).
- Clear deltas and regression visibility.

**Inputs/Dependencies**: Phases 3 & 5 models; golden QA set.  
**Est. Duration**: 1 day.

---

## Phase 9 — Performance & Cost Tuning

**Objective**  
Meet latency & accuracy targets efficiently.

**Deliverables**
- `retrieval_tuning.yaml` with selected `efSearch`, `M`, top-k, rerank depth.
- Cached pipelines and context condenser.

**Tasks**
- Tune Qdrant HNSW (`M`, `efConstruction`, `efSearch`).
- Adjust fusion depth, rerank depth, apply **MMR**.
- Add caching (embeddings, candidate IDs, rerank features, prompts).
- Implement context condenser (sentence selection).

**Acceptance (Exit)**
- p50/p95 latency targets achieved; no quality regression (per Phase 8).

**Inputs/Dependencies**: Phases 2, 5, 8.  
**Est. Duration**: 1 day.

---

## Phase 10 — Hardening, DX & Deployment

**Objective**  
Make it easy to run, debug, and deploy.

**Deliverables**
- Tracing (Langfuse/Phoenix), health checks, structured logs.
- Dockerized API; Vercel config for Next.js.
- Ops notes in README.

**Tasks**
- Add health endpoints, request logging, error boundaries.
- Set up env validation, guards (payload size limits).
- Local: `docker compose up` for infra + API; Web on Vercel.

**Acceptance (Exit)**
- Local spin-up one-liner works end-to-end.
- Preview deployment is functional and stable.

**Inputs/Dependencies**: All prior phases.  
**Est. Duration**: 0.5–1 day.

---

## Phase 11 — Test Suite & Edge Cases

**Objective**  
Prevent regressions and cover tricky scenarios.

**Deliverables**
- Unit tests (splitters, RRF/MMR, attribution mapping).
- Integration/E2E tests (happy path, agent refine path, OOD).
- CI pipeline (GitHub Actions or equivalent).

**Tasks**
- Build tests for ambiguous acronyms, typos, conflicting docs.
- Add load smoke (p50/p95 latency, memory).

**Acceptance (Exit)**
- CI green; coverage on critical paths; reproducible failures.

**Inputs/Dependencies**: Prior phases stable.  
**Est. Duration**: 1 day.

---

## Phase 12 — Documentation & Demo

**Objective**  
Enable quick onboarding and a compelling demo.

**Deliverables**
- `/docs/ARCHITECTURE.md`, `/docs/USAGE.md`, `/docs/EVAL.md`, `/docs/demo_script.md`.
- Screenshots/GIFs of agent trace and citations.

**Tasks**
- Capture trace screenshots; document toggles & metrics.
- Summarize design choices and trade-offs.

**Acceptance (Exit)**
- New contributor can run and understand the system in < 30 minutes.
- Demo checklist reproducible without author assistance.

**Inputs/Dependencies**: Prior phases complete.  
**Est. Duration**: 0.5 day.

---

## Requirement Coverage Map

| Requirement | Phase(s) | Evidence |
|---|---|---|
| Agentic RAG that retrieves chunks correctly | 2, 5, 6, 9 | `retrieval.py`, `graph.py`, RRF+rerank+MMR; eval passes |
| Working prototype (Next.js/Gradio/Streamlit) | 4, 7 | Next.js chat with streaming + citations + trace |
| Thought process & implementation flow | 12 | `ARCHITECTURE.md` + diagrams + notes |
| Investigation of Agentic RAG as a whole | 5, 8, 12 | Trace screenshots, EVAL.md discussion |
| Differences: traditional vs agentic RAG | 3, 5, 8, 7 | Mode toggle + side-by-side metrics on `/eval` |
| Any open-source libraries | 0–12 | bge-reranker-v2, Qdrant, (optionally Llama) |
| Test cases for quality | 8, 11 | RAGAS suite + unit/E2E tests |
| Bonus: citations handling | 6, 7 | Sentence-level attribution + UI hovers |
| Bonus: optimized retrieval | 2, 9 | Fusion, rerank, MMR, HNSW tuning, caching |
| **Explicit model choices** | **0, 1, 2.5, 3, 5** | `gpt-5-mini` for generation & agents; `text-embedding-3-small` for embeddings |

---

## Quick Start Commands

```bash
# Infra
docker compose up -d qdrant elasticsearch

# Ingest
cd services/indexer
python ingest.py --path ./data

# API
cd ../rag_api
export OPENAI_MODEL=gpt-5-mini
export EMBEDDING_MODEL=text-embedding-3-small
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Web
cd ../../apps/web
pnpm i
pnpm dev
```

---

### Notes on Sequencing
- Build **retrieval quality first** (Phase 2) before agent loops (Phase 5).
- Finalize **citations payload** (Phase 6) before heavy UI polish (Phase 7).
- Use **evaluation** (Phase 8) to guide **tuning** (Phase 9); re-run after changes.
