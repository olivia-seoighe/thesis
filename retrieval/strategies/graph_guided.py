"""Graph-guided retrieval strategy.

Combines knowledge graph traversal with vector similarity search.

Retrieval flow:
  1. Graph traversal (GraphClient) — find structurally related documents
     by walking the KG from query entities (Kafka topics, handlers, services)
  2. Vector search (SearchClient) — find semantically similar documents
  3. RRF fusion — merge both ranked lists

This strategy answers queries that require following relationships across
files, e.g:
  - "Which handlers publish to the topic consumed by the external integration?"
  - "What tables does the billing workflow read and write?"
  - "What downstream services are affected if the lab result handler changes?"

These are multi-hop questions where vector search alone retrieves documents
that mention the starting entity but misses the structurally connected ones.

The graph_guided strategy is the primary experimental condition for RQ1:
  "Does GraphRAG outperform standard vector RAG for codebase Q&A?"

Baseline conditions (vector_only, keyword_only, hybrid RRF) are defined
in rrf.py and wired in hybrid_search.py.

See CLAUDE.md §RQ1 and roadmap steps 3–4.
"""
