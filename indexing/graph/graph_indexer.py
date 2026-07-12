"""Apache AGE graph indexer.

Writes knowledge graph triples to the AGE graph in PostgreSQL using
openCypher MERGE statements via psycopg2.

AGE wraps Cypher in a SQL function:
    SELECT * FROM cypher('enterprise_graph', $$ MERGE (n:Label {name: 'X'}) $$) AS (v agtype);

MERGE ensures idempotency — re-running ingestion does not create duplicate nodes
or edges. Nodes are matched by their `name` property.

Requires the AGE extension to be installed in the PostgreSQL instance and the
'enterprise_graph' graph to be created:
    CREATE EXTENSION IF NOT EXISTS age;
    SELECT create_graph('enterprise_graph');

See CLAUDE.md §D6 (Apache AGE decision) and §D7 (graph schema).
"""

from indexing.graph.graph_extractor import Triple


class GraphIndexer:
    """Upserts knowledge graph triples into Apache AGE."""

    def __init__(self, connection_manager) -> None:
        self.cm = connection_manager

    async def upsert(self, triples: list[Triple]) -> None:
        """Write a list of triples to the AGE graph using Cypher MERGE.

        Each triple produces:
          1. MERGE on the subject node (by name + label)
          2. MERGE on the object node (by name + label)
          3. MERGE on the directed edge between them

        Args:
            triples: Extracted triples from GraphExtractor.extract().
        """
        raise NotImplementedError

    async def upsert_service_node(self, service: str) -> None:
        """Ensure a top-level Service node exists for the given repo.

        Args:
            service: Service name, e.g. "service_A".
        """
        raise NotImplementedError
