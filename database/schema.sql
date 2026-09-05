CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE knowledge graph (openCypher queries over enterprise architecture graph)
-- Two-tier graph truth:
--   1) AST_LOCAL
--   2) CONTRACT_GLOBAL
--
-- Canonical vertex labels (v1):
--   REPO, HANDLER, COMMAND, EVENT, BUSINESS_RULE, STATUS_CODE,
--   SAGA, FEATURE_FLAG, TABLE, KAFKA_TOPIC, API, FRAMEWORK, NUGET_PACKAGE
--
-- Canonical edge labels (v1):
--   OWNS_HANDLER, OWNS_COMMAND, OWNS_EVENT, OWNS_SAGA, OWNS_TABLE,
--   HANDLES_COMMAND, HANDLES_EVENT, EMITS_EVENT, TRANSITIONS_STATUS,
--   ENFORCES_RULE, USES_FEATURE_FLAG, READS_TABLE, WRITES_TABLE,
--   CONSUMES_TOPIC, PRODUCES_TOPIC, CALLS_API, EXPOSES_API, TARGETS_FRAMEWORK, CONTAINS_PACKAGE,
--   SAGA_ORCHESTRATES_COMMAND, SAGA_AWAITS_EVENT
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

-- Evidence persistence for graph provenance:
-- Keep canonical AGE nodes/edges focused on identity + semantics.
-- Persist additive provenance rows in these tables to support citations,
-- confidence scoring, and corroboration from multiple source files.
CREATE TABLE IF NOT EXISTS graph_node_evidence (
    node_key        TEXT NOT NULL, -- canonical key (for example: "HANDLER::CreateAsa")
    node_label      TEXT NOT NULL,
    node_name       TEXT NOT NULL,
    tier            TEXT NOT NULL CHECK (tier IN ('AST_LOCAL', 'CONTRACT_GLOBAL')),
    source_repo     TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    source_kind     TEXT NOT NULL, -- ast, asyncapi, appsettings, configmap, etc.
    line_start      INTEGER,
    line_end        INTEGER,
    extractor_name  TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    evidence_hash   TEXT NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_id     TEXT,
    chunk_id        TEXT,
    PRIMARY KEY (node_key, evidence_hash)
);

CREATE INDEX IF NOT EXISTS idx_gne_node_key ON graph_node_evidence (node_key);
CREATE INDEX IF NOT EXISTS idx_gne_source_repo ON graph_node_evidence (source_repo);
CREATE INDEX IF NOT EXISTS idx_gne_source_path ON graph_node_evidence (source_path);
CREATE INDEX IF NOT EXISTS idx_gne_tier ON graph_node_evidence (tier);
CREATE INDEX IF NOT EXISTS idx_gne_observed_at ON graph_node_evidence (observed_at);

CREATE TABLE IF NOT EXISTS graph_edge_evidence (
    edge_key         TEXT NOT NULL, -- canonical key from full directed relation identity
    predicate        TEXT NOT NULL,
    subject_key      TEXT NOT NULL,
    subject_label    TEXT NOT NULL,
    subject_name     TEXT NOT NULL,
    object_key       TEXT NOT NULL,
    object_label     TEXT NOT NULL,
    object_name      TEXT NOT NULL,
    tier             TEXT NOT NULL CHECK (tier IN ('AST_LOCAL', 'CONTRACT_GLOBAL')),
    source_repo      TEXT NOT NULL,
    source_path      TEXT NOT NULL,
    source_kind      TEXT NOT NULL,
    line_start       INTEGER,
    line_end         INTEGER,
    extractor_name   TEXT NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    evidence_hash    TEXT NOT NULL,
    observed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_id      TEXT,
    chunk_id         TEXT,
    PRIMARY KEY (edge_key, evidence_hash)
);

CREATE INDEX IF NOT EXISTS idx_gee_edge_key ON graph_edge_evidence (edge_key);
CREATE INDEX IF NOT EXISTS idx_gee_subject_key ON graph_edge_evidence (subject_key);
CREATE INDEX IF NOT EXISTS idx_gee_object_key ON graph_edge_evidence (object_key);
CREATE INDEX IF NOT EXISTS idx_gee_predicate ON graph_edge_evidence (predicate);
CREATE INDEX IF NOT EXISTS idx_gee_source_repo ON graph_edge_evidence (source_repo);
CREATE INDEX IF NOT EXISTS idx_gee_source_path ON graph_edge_evidence (source_path);
CREATE INDEX IF NOT EXISTS idx_gee_tier ON graph_edge_evidence (tier);
CREATE INDEX IF NOT EXISTS idx_gee_observed_at ON graph_edge_evidence (observed_at);

CREATE TABLE IF NOT EXISTS document_metadata (
    document_id        TEXT NOT NULL,
    retrieval_corpus    TEXT NOT NULL,
    name               TEXT,
    url                TEXT,
    source             TEXT,
    last_modified_date TIMESTAMPTZ,
    last_indexed_at    TIMESTAMPTZ DEFAULT NOW(),
    source_refs        TEXT        ,
    PRIMARY KEY (document_id, retrieval_corpus)
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
    source         TEXT,
    retrieval_corpus TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_de_source    ON document_embeddings (source); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_de_document  ON document_embeddings (document_id); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_de_corpus    ON document_embeddings (retrieval_corpus); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_dm_source    ON document_metadata (source); -- Filtering indexes
CREATE INDEX IF NOT EXISTS idx_de_tsv       ON document_embeddings USING GIN (tsv) WITH (fastupdate = on); -- Index for keyword search (GIN on tsvector) -- fastupdate improves bulk-insert throughput
CREATE INDEX IF NOT EXISTS idx_de_embedding_hnsw ON document_embeddings -- Indexes for vector search, HNSW  for approximate nearest-neighbour vector search
    USING hnsw (embedding_3072 halfvec_cosine_ops) WITH (m = 24, ef_construction = 200);


-- Foreign key relationship
ALTER TABLE document_embeddings
    ADD CONSTRAINT fk_embeddings_document_id
    FOREIGN KEY (document_id, retrieval_corpus) REFERENCES document_metadata(document_id, retrieval_corpus)
    ON DELETE CASCADE;

-- Precomputed embeddings for graph node names, used by the embedding-based
-- entity linker to bridge naturally-phrased queries to graph entities that
-- token-based exact/substring matching alone cannot find.
CREATE TABLE IF NOT EXISTS graph_node_name_embeddings (
    node_key       TEXT PRIMARY KEY,
    node_label     TEXT NOT NULL,
    node_name      TEXT NOT NULL,
    confidence     DOUBLE PRECISION NOT NULL,
    evidence_count INTEGER NOT NULL,
    embedding      halfvec(:embedding_dim),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gnne_label ON graph_node_name_embeddings (node_label);

-- Conversation history (persisted so history survives service restarts)
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    messages    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);