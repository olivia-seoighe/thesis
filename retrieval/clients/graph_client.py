"""Apache AGE graph retrieval client.

Runs openCypher neighbourhood queries against the enterprise knowledge graph to
retrieve document chunks via graph traversal rather than vector similarity.

Multi-hop retrieval pattern:
  1. Entity recognition — identify concepts in the query
     (service names, Kafka topics, handler names, table names, etc.)
  2. Entry-point lookup — find matching nodes in the graph by name
  3. Neighbourhood expansion — traverse edges up to N hops to find
     structurally related nodes (e.g. all handlers that publish to a
     topic consumed by the queried service)
  4. Chunk retrieval — fetch document_embeddings rows for each discovered
     node's source file, ranked by graph proximity (hop count)

The returned SearchResponse is passed into rrf_merge alongside vector and
keyword responses in retrieval/endpoints/hybrid_search.py.

Cypher queries use the cypher() SQL function from Apache AGE:
    SELECT * FROM cypher('enterprise_graph', $$
        MATCH (h:Handler)-[:PUBLISHES]->(t:KafkaTopic {name: $topic})
        RETURN h
    $$, $1) AS (handler agtype);

See CLAUDE.md §D6 (AGE decision), §D7 (graph schema), roadmap step 4.
"""

from models.models import SearchRequest, SearchResponse


class GraphClient:
    """Retrieves document chunks via knowledge graph traversal (Apache AGE)."""

    def __init__(self, connection_manager) -> None:
        self.cm = connection_manager

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Run a graph neighbourhood query and return ranked document chunks.

        Entity recognition extracts concepts from request.query,
        finds matching graph nodes, expands the neighbourhood up to 2 hops,
        and returns the source files for all discovered nodes as chunks.

        Chunks are ranked by hop distance (1-hop neighbours rank above 2-hop).

        Args:
            request: Standard SearchRequest — query and sources used;
                     top_k caps the number of returned chunks.

        Returns:
            SearchResponse compatible with rrf_merge.
        """
        raise NotImplementedError
