"""Generation Service — RAG query endpoint.

Endpoints:
    GET  /health
    POST /query                  retrieval (hybrid/vector/keyword/graph) + LLM answer with citations
    GET  /conversations          list conversation history
    GET  /conversations/{id}     get one conversation
    DELETE /conversations/{id}   delete one conversation
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

from llm_client import LLMClient
from models import (
    Citation,
    Conversation,
    Message,
    QueryRequest,
    QueryResponse,
)

RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8000")
DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI(title="Code RAG Generation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm = LLMClient()
_db: asyncpg.Pool | None = None
# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_db() -> asyncpg.Pool:
    global _db
    if _db is None:
        _db = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _db


async def _db_upsert_conversation(conv: Conversation) -> None:
    pool = await _get_db()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO conversations (id, title, messages, created_at, updated_at)
            VALUES ($1, $2, $3::jsonb, $4, NOW())
            ON CONFLICT (id) DO UPDATE
              SET messages   = EXCLUDED.messages,
                  updated_at = NOW()
            """,
            conv.id,
            conv.title,
            json.dumps([m.model_dump() for m in conv.messages]),
            datetime.fromisoformat(conv.created_at),
        )


async def _db_load_conversation(conv_id: str) -> Conversation | None:
    pool = await _get_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, title, messages, created_at FROM conversations WHERE id = $1",
            conv_id,
        )
    if row is None:
        return None
    msgs_raw = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    messages = [Message(**m) for m in msgs_raw]
    return Conversation(
        id=row["id"],
        title=row["title"],
        messages=messages,
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
    )


async def _db_list_conversations() -> List[Conversation]:
    pool = await _get_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, messages, created_at FROM conversations ORDER BY updated_at DESC"
        )
    result = []
    for row in rows:
        msgs_raw = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
        messages = [Message(**m) for m in msgs_raw]
        result.append(Conversation(
            id=row["id"],
            title=row["title"],
            messages=messages,
            created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
        ))
    return result


async def _db_delete_conversation(conv_id: str) -> bool:
    pool = await _get_db()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM conversations WHERE id = $1", conv_id)
    return result == "DELETE 1"


# ── Retrieval helpers ─────────────────────────────────────────────────────────

RETRIEVAL_MODES = ("hybrid", "vector", "keyword", "graph")
RETRIEVAL_MODE_ALIASES: dict[str, str] = {
    "graph-service-aware": "graph",
    "hybrid-service-aware": "hybrid",
    "keyword-service-aware": "keyword",
    "vector-service-aware": "vector",
}


async def _retrieve(client: httpx.AsyncClient, query: str, source: str | None, top_k: int, mode: str) -> list:
    mode = RETRIEVAL_MODE_ALIASES.get(mode, mode)
    mode = mode if mode in RETRIEVAL_MODES else "hybrid"
    params: dict = {"query": query, "top_k": top_k}
    if source:
        params["source"] = source
    resp = await client.get(
        f"{RETRIEVAL_URL}/search/{mode}",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _flatten_chunks(retrieval_results: list) -> list[dict[str, Any]]:
    return [chunk for response in retrieval_results for chunk in response.get("chunks", [])]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "retrieval_url": RETRIEVAL_URL}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    t_start = time.time()
    source = req.source
    conv_id = req.conversation_id or str(uuid.uuid4())

    # Retrieval — hybrid/vector/keyword/graph depending on req.mode
    t_ret_start = time.time()
    async with httpx.AsyncClient() as client:
        try:
            retrieval_results = await _retrieve(client, req.query, source, req.top_k, req.mode)
            merged_chunks = _flatten_chunks(retrieval_results)
        except httpx.HTTPStatusError as exc:
            log.error(f"Retrieval failed: {exc}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Retrieval error: {exc}")

    retrieval_ms = (time.time() - t_ret_start) * 1000

    if not merged_chunks:
        log.warning(f"No results retrieved for query: {req.query!r}")

    # Build conversation history for multi-turn context
    history: list[dict] = []
    existing_conv = await _db_load_conversation(conv_id)
    if existing_conv:
        for msg in existing_conv.messages[-6:]:  # last 3 turns
            history.append({"role": msg.role, "content": msg.content})

    # LLM generation
    t_gen_start = time.time()
    answer, model_used = await _llm.generate(
        query=req.query,
        chunks=merged_chunks,
        history=history or None,
        model=req.model,
    )
    gen_ms = (time.time() - t_gen_start) * 1000
    total_ms = (time.time() - t_start) * 1000

    citations = [
        Citation(
            title=c.get("document_title", c.get("name", "")),
            url=c.get("url", ""),
            score=c.get("score", 0.0),
            chunk_text=c.get("text", ""),
            source_code=c.get("source_code", ""),
            metadata=c.get("metadata", {}),
        )
        for c in merged_chunks
    ]

    # Store in conversation history
    now = datetime.now(timezone.utc).isoformat()
    user_msg = Message(role="user", content=req.query, timestamp=now)
    asst_msg = Message(role="assistant", content=answer, citations=citations, timestamp=now)

    if existing_conv:
        existing_conv.messages.extend([user_msg, asst_msg])
        conv = existing_conv
    else:
        conv = Conversation(
            id=conv_id,
            title=req.query[:80],
            messages=[user_msg, asst_msg],
            created_at=now,
        )
    await _db_upsert_conversation(conv)

    log.info(
        f"Query complete — conv={conv_id} chunks={len(merged_chunks)} "
        f"retrieval={retrieval_ms:.0f}ms gen={gen_ms:.0f}ms total={total_ms:.0f}ms"
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        conversation_id=conv_id,
        model_used=model_used,
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=gen_ms,
    )


@app.get("/conversations", response_model=List[Conversation])
async def list_conversations() -> List[Conversation]:
    return await _db_list_conversations()


@app.get("/conversations/{conv_id}", response_model=Conversation)
async def get_conversation(conv_id: str) -> Conversation:
    conv = await _db_load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str) -> dict:
    deleted = await _db_delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": conv_id}
