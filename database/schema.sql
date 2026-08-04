CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE knowledge graph (openCypher queries over the enterprise architecture graph)
-- Vertex labels: Service, KafkaTopic, ExternalApi, FeatureFlag, Table,
--                ServiceBusEndpoint, Event, Handler
-- Edge labels:   PUBLISHES, SUBSCRIBES, CALLS, GATED_BY, HANDLES, PRODUCES,
--                READS, WRITES, SENDS_TO, RECEIVES_FROM, HAS_FK, BELONGS_TO
CREATE EXTENSION IF NOT EXISTS age;

SET search_path = ag_catalog, "$user", public;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'enterprise_graph'
    ) THEN
        PERFORM ag_catalog.create_graph('enterprise_graph');
    END IF;
END;
$$;
RESET search_path;

CREATE TABLE IF NOT EXISTS document_metadata (
    document_id        TEXT PRIMARY KEY,
    name               TEXT,
    url                TEXT,
    source             TEXT,
    last_modified_date TIMESTAMPTZ,
    last_indexed_at    TIMESTAMPTZ DEFAULT NOW(),
    source_refs        TEXT        
);


\getenv embedding_dim EMBEDDING_DIM
\if :{?embedding_dim}
\else
  \set embedding_dim 3072
\endif

CREATE TABLE IF NOT EXISTS document_embeddings (
    chunk_id       TEXT PRIMARY KEY,
    text           TEXT,
    source_code    TEXT,
    document_id    TEXT,
    document_title TEXT,
    embedding_3072      halfvec(:embedding_dim),
    tsv            tsvector,
    metadata       JSONB,
    source         TEXT
);

CREATE INDEX IF NOT EXISTS idx_de_source    ON document_embeddings (source); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_de_document  ON document_embeddings (document_id); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_dm_source    ON document_metadata (source); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_de_tsv       ON document_embeddings USING GIN (tsv) WITH (fastupdate = on); -- Index for keyword search (GIN on tsvector) -- fastupdate improves bulk-insert throughput
CREATE INDEX IF NOT EXISTS idx_de_embedding_hnsw ON document_embeddings -- Indexes for vector search, HNSW  for approximate nearest-neighbour vector search
    USING hnsw (embedding_3072 halfvec_cosine_ops) WITH (m = 24, ef_construction = 200);


-- Foreign key relationship
ALTER TABLE document_embeddings
    ADD CONSTRAINT fk_embeddings_document_id
    FOREIGN KEY (document_id) REFERENCES document_metadata(document_id)
    ON DELETE CASCADE;

-- Conversation history (persisted so history survives service restarts)
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    messages    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);