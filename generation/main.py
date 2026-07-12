"""Generation Service — RAG query endpoint.

Endpoints:
    GET  /health
    POST /query                  hybrid retrieval + LLM answer with citations
    GET  /conversations          list conversation history
    GET  /conversations/{id}     get one conversation
    DELETE /conversations/{id}   delete one conversation
    GET  /viz/embeddings         2-D PCA projection of recent query/chunk embeddings
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import asyncpg
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA

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
    EmbeddingPoint,
    Message,
    QueryRequest,
    QueryResponse,
    VizResponse,
)

RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8000")
DEFAULT_SOURCE = os.getenv("DEFAULT_SOURCE", "sample-service")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpassword@postgres:5432/ragdb")

# RRF formula: score = 1/(60 + rank), where 60 is the standard dampening constant
# from Cormack et al. (2009) and rank starts at 1.
# Single-list max (rank 1 in one search only): 1/61 ≈ 0.016.
# Both-lists min (rank 5 in both searches): 2/65 ≈ 0.031.
# 0.025 sits in the gap — a chunk must appear in both vector AND keyword
# results to pass. If no chunk passes, the LLM answers without citing sources.
MIN_TOP_CITATION_SCORE = float(os.getenv("MIN_TOP_CITATION_SCORE", "0.025"))

app = FastAPI(title="Code RAG Generation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm = LLMClient()
_db: asyncpg.Pool | None = None

# Ring buffer of recent embedding data for visualisation
_embedding_store: Deque[Dict[str, Any]] = deque(maxlen=50)


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
            conv.created_at,
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

async def _hybrid_search(client: httpx.AsyncClient, query: str, source: str, top_k: int) -> list:
    resp = await client.get(
        f"{RETRIEVAL_URL}/search/hybrid",
        params={"query": query, "source": source, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "retrieval_url": RETRIEVAL_URL}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    t_start = time.time()
    source = req.source or DEFAULT_SOURCE
    conv_id = req.conversation_id or str(uuid.uuid4())

    # Hybrid retrieval — single call; RRF fusion happens inside the retrieval service
    t_ret_start = time.time()
    async with httpx.AsyncClient() as client:
        try:
            hybrid_results = await _hybrid_search(client, req.query, source, req.top_k)
        except httpx.HTTPStatusError as exc:
            log.error(f"Retrieval failed: {exc}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Retrieval error: {exc}")

    retrieval_ms = (time.time() - t_ret_start) * 1000
    merged_chunks = [chunk for response in hybrid_results for chunk in response.get("chunks", [])]

    top_score = merged_chunks[0]["score"] if merged_chunks else 0.0
    if not merged_chunks:
        log.warning(f"No results retrieved for query: {req.query!r}")
    elif top_score < MIN_TOP_CITATION_SCORE:
        log.info(
            f"Top chunk score {top_score:.4f} below threshold {MIN_TOP_CITATION_SCORE} — "
            f"skipping citations for query: {req.query!r}"
        )
        merged_chunks = []

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

    # Store embeddings metadata for visualisation (we re-fetch inline via a lightweight
    # approach — just record query text + chunk scores for PCA later)
    # NOTE: actual high-dim embeddings would need a separate embedding call or
    # passthrough from the retrieval service. For the POC we generate random
    # placeholder vectors seeded on chunk_id (replaced with real embeds when
    # the retrieval API exposes them).
    _embedding_store.append({
        "conv_id": conv_id,
        "query": req.query,
        "timestamp": now,
        "chunks": [
            {"id": c.get("chunk_id", ""), "text": c.get("text", "")[:100], "score": c.get("score", 0.0)}
            for c in merged_chunks
        ],
    })

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


@app.get("/viz/embeddings", response_model=VizResponse)
async def viz_embeddings() -> VizResponse:
    """Return PCA-projected 2-D embedding points for recent queries + retrieved chunks.

    Uses deterministic hashing to simulate per-text embeddings until the retrieval
    service exposes raw vectors. Replace _text_to_pseudo_vec with a real embedding
    call for production use.
    """
    if len(_embedding_store) < 2:
        return VizResponse(points=[], note="Not enough data yet — run some queries first.")

    raw_points: list[dict] = []

    def _text_to_pseudo_vec(text: str) -> np.ndarray:
        """Deterministic pseudo-embedding seeded on text hash (placeholder)."""
        import hashlib
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        return rng.standard_normal(64)

    for entry in _embedding_store:
        raw_points.append({
            "id": f"q_{entry['conv_id'][:8]}",
            "label": entry["query"][:60],
            "vec": _text_to_pseudo_vec(entry["query"]),
            "score": 1.0,
            "type": "query",
            "source": "",
        })
        for chunk in entry.get("chunks", []):
            raw_points.append({
                "id": chunk["id"],
                "label": chunk["text"][:60],
                "vec": _text_to_pseudo_vec(chunk["text"]),
                "score": chunk["score"],
                "type": "chunk",
                "source": "",
            })

    if len(raw_points) < 2:
        return VizResponse(points=[], note="Not enough distinct points.")

    matrix = np.stack([p["vec"] for p in raw_points])
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(matrix)

    points = [
        EmbeddingPoint(
            id=p["id"],
            label=p["label"],
            x=float(coords[i, 0]),
            y=float(coords[i, 1]) if coords.shape[1] > 1 else 0.0,
            score=p["score"],
            type=p["type"],
            source=p["source"],
        )
        for i, p in enumerate(raw_points)
    ]

    return VizResponse(
        points=points,
        note=f"PCA of {len(points)} points from {len(_embedding_store)} queries "
             f"(pseudo-embeddings — wire real vectors from retrieval service for production).",
    )
