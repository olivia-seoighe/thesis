"""Build embeddings for graph node names used during seed resolution.

The script reads seedable nodes from Apache AGE and refreshes
public.graph_node_name_embeddings. It runs as a dry run unless --apply is set.
"""

import argparse
import asyncio
import os

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DSN = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "15433")),
    "user": os.environ["PGUSER"],
    "password": os.environ["PGPASSWORD"],
    "database": os.environ["PGDATABASE"],
}
GRAPH_NAME = "enterprise_graph"
SEEDABLE_NODE_LABELS = [
    "KAFKA_TOPIC",
    "API",
    "REPO",
    "FRAMEWORK",
    "NUGET_PACKAGE",
    "HANDLER",
    "COMMAND",
    "EVENT",
    "SAGA",
]

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS public.graph_node_name_embeddings (
        node_key       TEXT PRIMARY KEY,
        node_label     TEXT NOT NULL,
        node_name      TEXT NOT NULL,
        confidence     DOUBLE PRECISION NOT NULL,
        evidence_count INTEGER NOT NULL,
        embedding      halfvec(:embedding_dim),
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""".replace(":embedding_dim", os.getenv("EMBEDDING_DIM", "3072"))

CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_gnne_label ON public.graph_node_name_embeddings (node_label)"

FETCH_NODES_SQL = f"""
    WITH graph_hits AS (
        SELECT
            TRIM(BOTH '"' FROM node_key::text) AS node_key,
            TRIM(BOTH '"' FROM node_label::text) AS node_label,
            TRIM(BOTH '"' FROM node_name::text) AS node_name
        FROM ag_catalog.cypher('{GRAPH_NAME}', $$
            MATCH (n)
            WHERE n.label IN $labels
            RETURN n.node_key AS node_key, n.label AS node_label, n.name AS node_name
        $$, $1::ag_catalog.agtype) AS (node_key ag_catalog.agtype, node_label ag_catalog.agtype, node_name ag_catalog.agtype)
    )
    SELECT
        gh.node_key,
        gh.node_label,
        gh.node_name,
        COALESCE(MAX(gne.confidence), 1.0) AS confidence,
        COUNT(gne.*)::int AS evidence_count
    FROM graph_hits gh
    LEFT JOIN graph_node_evidence gne ON gne.node_key = gh.node_key
    GROUP BY gh.node_key, gh.node_label, gh.node_name
"""

UPSERT_SQL = """
    INSERT INTO public.graph_node_name_embeddings
        (node_key, node_label, node_name, confidence, evidence_count, embedding, updated_at)
    VALUES ($1, $2, $3, $4, $5, $6::halfvec, NOW())
    ON CONFLICT (node_key) DO UPDATE SET
        node_label     = EXCLUDED.node_label,
        node_name      = EXCLUDED.node_name,
        confidence     = EXCLUDED.confidence,
        evidence_count = EXCLUDED.evidence_count,
        embedding      = EXCLUDED.embedding,
        updated_at     = NOW()
"""

DELETE_STALE_SQL = "DELETE FROM public.graph_node_name_embeddings WHERE node_key != ALL($1::text[])"


async def embed_batch(texts: list[str]) -> list[list[float]]:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    dim = os.getenv("EMBEDDING_DIM")
    kwargs = {"model": model}
    if dim:
        kwargs["dimensions"] = int(dim)

    embeddings: list[list[float]] = []
    batch_size = 200
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(input=batch, **kwargs)
        items = sorted(response.data, key=lambda x: x.index)
        embeddings.extend(item.embedding for item in items)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    await client.close()
    return embeddings


async def run(apply: bool) -> None:
    conn = await asyncpg.connect(**DSN)
    try:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')

        import json

        agtype_param = json.dumps({"labels": SEEDABLE_NODE_LABELS}, separators=(",", ":"))
        rows = await conn.fetch(FETCH_NODES_SQL, agtype_param)
        await conn.execute("RESET search_path;")
        print(f"Found {len(rows)} distinct seedable node names in the live graph")

        by_label: dict[str, int] = {}
        for row in rows:
            by_label[row["node_label"]] = by_label.get(row["node_label"], 0) + 1
        for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
            print(f"  {label:<16}{count}")

        if not rows:
            print("No nodes found -- aborting without changes.")
            return

        print("\nEmbedding node names...")
        names = [row["node_name"] for row in rows]
        vectors = await embed_batch(names)

        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(CREATE_TABLE_SQL)
            await conn.execute(CREATE_INDEX_SQL)
            batch = [
                (
                    row["node_key"],
                    row["node_label"],
                    row["node_name"],
                    row["confidence"],
                    row["evidence_count"],
                    "[" + ",".join(str(v) for v in vector) + "]",
                )
                for row, vector in zip(rows, vectors)
            ]
            await conn.executemany(UPSERT_SQL, batch)

            current_keys = [row["node_key"] for row in rows]
            deleted = await conn.fetchval(
                "WITH deleted AS (" + DELETE_STALE_SQL + " RETURNING 1) SELECT COUNT(*) FROM deleted",
                current_keys,
            )
            print(f"Upserted {len(batch)} embeddings, removed {deleted} stale row(s)")

            after_count = await conn.fetchval("SELECT COUNT(*) FROM public.graph_node_name_embeddings")
            print(f"public.graph_node_name_embeddings now has {after_count} rows")

            if apply:
                await tx.commit()
                print("\n=== APPLIED ===")
            else:
                await tx.rollback()
                print("\n=== DRY RUN ONLY (rolled back). Re-run with --apply to commit. ===")
        except Exception:
            await tx.rollback()
            raise
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Commit changes (default: dry run / rollback)")
    args = parser.parse_args()
    asyncio.run(run(args.apply))
