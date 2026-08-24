## Overview

**Thesis:** *Hybrid Graph-Augmented Retrieval for Large Language Models in Distributed Event-Driven Architectures*

This research investigates retrieval strategies — vector, keyword, hybrid, and graph-based — for grounding LLM responses in large, interconnected codebases. The system summarises source code from production systems into structured, retrieval-optimised documents, which are then indexed into a vector store and a knowledge graph to support evaluation across retrieval strategies.

### Data Pipeline — Code Summarisation

Raw source code is too large and noisy to embed directly. The summarisation service fetches each configured repository from GitHub and uses an LLM to produce structured markdown summaries for each source file (purpose, business logic, service dependencies, data models). These are written to `summaries/<repo>/summaries.json`.

A knowledge graph is derived from the code summaries (entity and relationship extraction → `graph.json`) for use in the GraphRAG retrieval path.

### Indexing Pipeline — Chunking · Embedding · Storage

Summaries are split into overlapping word-window chunks (1500 words, 15% overlap), embedded using `text-embedding-3-large` (3072 dimensions), and stored in Postgres alongside a `tsvector` column for full-text search. HNSW and GIN indexes are built for fast vector and keyword retrieval respectively.

### Data Stores — Vector Store & Graph Store

- **Vector Store** — Postgres + pgvector. Stores embeddings, chunk text, source code, and document metadata.
- **Graph Store** — Apache AGE (graph extension on Postgres). Stores the knowledge graph for GraphRAG traversal.

### Retrieval Pipeline — Keyword · Vector · Hybrid · Graph

The retrieval service exposes keyword (full-text/GIN), vector (cosine/HNSW), hybrid (Reciprocal Rank Fusion), and graph-based retrieval modes. Results are ranked and filtered by relevance score before passing to generation.

### Generation & UI — LLM Q&A Interface

The generation service assembles retrieved context with citations and queries Claude to produce a grounded answer with multi-turn conversation support. The UI (`localhost:3000`) provides the chat interface.

### Evaluation Framework

The evaluation harness defines the methodology for comparing retrieval strategies. A fixed set of golden queries is run through retrieval-only and full end-to-end pipelines. Retrieval quality is assessed via LLM-based relevance classification (Arize Phoenix). End-to-end quality will be measured using RAGAS — faithfulness, answer relevancy, and context precision — with a P95 retrieval latency SLA of <3 seconds. *(Work in progress.)*

## Prerequisites

1. Docker Desktop running
2. Valid `.env` values, especially:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=...                
OPENAI_MODEL=...                   
OPENAI_EMBEDDING_MODEL=...         
GITHUB_TOKEN=...                   
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

## Step 4: Ingest Summaries into Vector DB

```bash
docker compose --profile ingest up --build indexing
docker compose --profile ingest down
```

## Step 5: Start Retrieval, Generation, UI

```bash
docker compose up -d retrieval generation ui
curl -fsS http://localhost:18000/live
curl -fsS http://localhost:18002/health
```

UI URL:

1. `http://localhost:13000`

## DB reset

```bash
docker compose down -v
docker compose up -d postgres pgadmin
```

Then re-run from Step 2.