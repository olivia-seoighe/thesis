## Overview

**Thesis:** *Hybrid Graph-Augmented Retrieval for Large Language Models in Distributed Event-Driven Architectures*

This research investigates retrieval strategies — vector, keyword, hybrid, and graph-based — for grounding LLM responses in large, interconnected codebases. The system summarises source code from production systems into structured, retrieval-optimised documents, which are then indexed into a vector store and a knowledge graph to support evaluation across retrieval strategies.

### Data Pipeline — Code Summarisation

Raw source code is too large and noisy to embed directly. The summarisation service fetches each configured repository from GitHub and uses an LLM to produce structured markdown summaries for each source file (purpose, business logic, service dependencies, data models). These are written to `summaries/<repo>/summaries.json`. A simplified GitHub webhook endpoint can refresh summaries after merged pull requests, showing how the production pattern keeps architecture documentation from going stale.

A knowledge graph is derived from the code summaries (entity and relationship extraction → `graph.json`) for use in the GraphRAG retrieval path.

### Indexing Pipeline — Chunking · Embedding · Storage

Summaries are split into overlapping word-window chunks (1500 words, 15% overlap), embedded using `text-embedding-3-large` (3072 dimensions), and stored in Postgres alongside a `tsvector` column for full-text search. HNSW and GIN indexes are built for fast vector and keyword retrieval respectively.

### Data Stores — Vector Store & Graph Store

- **Vector Store** — Postgres + pgvector. Stores embeddings, chunk text, source code, and document metadata.
- **Graph Store** — Apache AGE (graph extension on Postgres). Stores the knowledge graph for GraphRAG traversal.

### Retrieval Pipeline — Keyword · Vector · Hybrid · Graph

The retrieval service exposes keyword (BM25), vector (cosine/HNSW), hybrid (Reciprocal Rank Fusion over keyword, vector, and graph), and graph traversal modes. The default UI path is service-aware hybrid retrieval: query text is matched against `service_acronyms.json`, then retrieval runs globally, with a hard service filter only for explicit local-scope questions and service boosting otherwise. The structured-first graph router is retained only under `archive/` because it added template complexity and was not kept as part of the simplified active system.

### Generation & UI — LLM Q&A Interface

The generation service assembles retrieved context with citations and queries Claude to produce a grounded answer with multi-turn conversation support. The UI (`http://localhost:13000`) provides the chat interface when running through Docker Compose.

### Evaluation Framework

The harness evaluates retrieval against the golden query set (`evaluation/datasets/golden_queries_retrieval`) and writes per-query metrics to `results.xlsx` (`results` sheet) plus aggregate sheets `category_results` and `difficulty_results` (recall, precision, F1, MRR, nDCG, hit_count, latency_ms). The main strategy names are `keyword`, `vector`, `graph`, `hybrid`, and `hybrid-service-aware`; `keyword` and `hybrid` use BM25 keyword ranking.

Service-aware variants (`*-service-aware`) derive routing from query text plus a service acronym catalogue (`service_acronyms.json`) or retrieval `/sources`, then apply metadata mode decisions (GLOBAL / HARD_FILTER / BOOST) and record routing fields alongside metrics.

## Prerequisites

1. Docker Desktop running
2. Valid `.env` values, especially:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...                
OPENAI_MODEL=...                   
OPENAI_EMBEDDING_MODEL=...         
GITHUB_TOKEN=...                   
GITHUB_WEBHOOK_SECRET=...
```

## Important Paths

1. Summaries output folder: `summaries` (per-repo `summaries.json` + `failed_files.json`)
2. DB schema bootstrap file: `database/schema.sql`
3. Compose file: `docker-compose.yaml`
4. Summariser service code: `code_summarisation/main.py`, `code_summarisation/summariser.py`

## Step 1: Start Core Infrastructure

```bash
docker compose up -d postgres pgadmin
docker compose ps postgres pgadmin
docker compose logs --tail 100 postgres
```


## Step 2: Start Summariser

```bash
docker compose up -d --build --force-recreate code-summarisation-agent
until curl -fsS http://localhost:18001/health >/dev/null; do sleep 1; done
```

## Step 3: Run Full Summarisation

```bash
# First run or retry failures (skips already-summarised files)
curl -sS -X POST "http://localhost:18001/summarize/batch"

# Force full re-summarisation
curl -sS -X POST "http://localhost:18001/summarize/batch?force=true"

docker compose logs -f code-summarisation-agent

ls -lah summaries          # one folder per repo
find summaries -name summaries.json | head
find summaries -name failed_files.json | head
```

Optional webhook refresh path:

```text
POST /webhooks/github
```

The endpoint accepts signed GitHub `pull_request` webhooks, ignores unmerged PRs, and refreshes local summaries for changed files in configured repositories.

## Step 4: Ingest Summaries into Vector DB

```bash
docker compose --profile ingest up --build indexing
docker compose --profile ingest down
```

Graph-only refresh (updates knowledge graph evidence only; does not rewrite `document_embeddings` chunks):

```bash
GRAPH_ONLY=true GRAPH_INDEXING_ENABLED=true docker compose --profile ingest up --build indexing
docker compose --profile ingest down
```

## Step 5: Start Retrieval, Generation, UI

```bash
docker compose up -d retrieval generation ui
curl -fsS http://localhost:18000/live
curl -fsS http://localhost:18002/health
curl -fsS -I http://localhost:13000
```

UI URL:

1. `http://localhost:13000`

## DB reset

```bash
docker compose down -v
docker compose up -d postgres pgadmin
```

Then re-run from Step 2.