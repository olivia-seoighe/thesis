# Section 6 — MCP Integration (Placeholder)

## Goal

Extend the RAG prototype to answer questions using **live operational data** — specifically error-queue contents — via an MCP (Model Context Protocol) server.

## Proposed Architecture

```
RAG Prototype (generation service)
    └── MCP Client
            └── ops-health (MCP Server)
                    └── message broker error queues (live data)
```

## Tasks

1. **Run the ops MCP server locally**
   - Clone the ops MCP server repository
   - Configure connection to your message broker (requires a connection string)
   - Run: `docker compose up ops-health`

2. **Configure MCP client in the generation service**
   - Add `mcp` dependency to `generation/requirements.txt`
   - Implement an `MCPClient` in `generation/mcp_client.py`
   - Expose an MCP tool: `get_error_queue_messages(queue_name: str, top_n: int = 10)`

3. **Wire live error-queue data into `/query` responses**
   - When a query mentions "error queue", "dead letter", or "failed messages",
     call the MCP tool to fetch live queue data
   - Inject queue data as additional context alongside the RAG chunks
   - Cite the queue data with `[Live: <queue_name>]` in the answer

## Example Query

> "Are there any failed evaluations in the dead-letter queue right now?"

Expected response cites both:
- `[Source 1]` — RAG chunk explaining the DLQ retry logic
- `[Live: service-dlq]` — live count and sample messages from the actual queue

## Environment Variables Needed

```env
SERVICE_BUS_CONNECTION_STRING=<broker-connection-string>
MCP_SERVER_URL=http://ops-health:8080
```

## Future Iterations

- Stream real-time queue updates via WebSocket to the UI
- Correlate error queue messages with code summaries (e.g., "which file handles this error type?")
- Alert the LLM when queue depth exceeds a threshold
