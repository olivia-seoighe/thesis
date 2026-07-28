# Base image already includes PostgreSQL 16 + pgvector extension.
# This Dockerfile adds Apache AGE on top for knowledge graph (openCypher) support.
# Result: one postgres instance with three query modes —
#   pgvector  → vector similarity search (HNSW)
#   tsvector  → full-text keyword search (BM25-style)
#   Apache AGE → graph traversal (openCypher / Cypher queries)
FROM pgvector/pgvector:pg16

# Build dependencies for Apache AGE
# bison + flex are required for AGE's Cypher parser/lexer code generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    bison \
    flex \
    && rm -rf /var/lib/apt/lists/*

# Build and install Apache AGE from source (PG16 branch)
# Depth 1 keeps the clone fast — we only need the latest commit
RUN git clone --branch PG16 --depth 1 \
    https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config \
    && make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install \
    && rm -rf /tmp/age
