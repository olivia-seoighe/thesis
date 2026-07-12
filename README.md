## Overview

This project runs a full RAG pipeline with:

1. Code summarisation service
2. Indexing into Postgres + pgvector
3. Retrieval API
4. Generation API
5. UI

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

1. Summaries output folder: `summaries`
2. DB schema bootstrap file: `indexing/database/schema.sql`
3. Compose file: `docker-compose.yaml`
4. Summariser service code: `code_summarisation/main.py`, `code_summarisation/summariser.py`

## Step 1: Start Core Infrastructure

```bash
docker compose up -d postgres pgadmin
docker compose ps postgres pgadmin
docker logs --tail 100 poc-postgres
```


## Step 2: Start Summariser

```bash
docker compose up -d --build --force-recreate code-summarisation-agent
until curl -fsS http://localhost:8001/health >/dev/null; do sleep 1; done
```

## Step 3: Run Full Summarisation

```bash
curl -sS -X POST "http://localhost:8001/summarize/fetch" \
	-H "Content-Type: application/json" \
	-d "{}"

docker logs -f poc-summariser

ls -lah summaries | head
wc -c summaries/summaries.json
```

## Step 4: Ingest Summaries into Vector DB

```bash
docker compose --profile ingest up --build indexing
docker compose --profile ingest down
```

## Step 5: Start Retrieval, Generation, UI

```bash
docker compose up -d retrieval generation ui
curl -fsS http://localhost:8000/live
curl -fsS http://localhost:8002/health
```

UI URL:

1. `http://localhost:3000`

## Delete old summaries. This will be replaced with a batch pipeline should overwrite/delete old summaries

```bash

# Clear old summaries
find summaries -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# Re-run summariser
curl -sS -X POST "http://localhost:8001/summarize/fetch" -H "Content-Type: application/json" -d "{}"

# Re-ingest
docker compose --profile ingest up --build indexing
```


## DB reset

```bash
docker compose down -v
docker compose up -d postgres pgadmin


# And then re-run pipeline:
# docker compose up -d --build --force-recreate code-summarisation-agent
```