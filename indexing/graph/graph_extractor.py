"""Rule-based graph triple extractor.

Reads the structured sections of a code summary and produces (subject, predicate,
object) triples for ingestion into the Apache AGE knowledge graph.

Extraction is rule-based (not LLM-driven) for reproducibility — each structured
summary section maps directly to a set of triples:

    ## Topics Consumed     → (Handler) -[SUBSCRIBES]→  (KafkaTopic)
    ## Topics Produced     → (Handler) -[PUBLISHES]→   (KafkaTopic)
    ## Azure Service Bus   → (Service) -[SENDS_TO]→    (ServiceBusEndpoint)
    ## External API Calls  → (Handler) -[CALLS]→       (ExternalApi)
    ## Feature Flags       → (Handler) -[GATED_BY]→    (FeatureFlag)
    ## Database Tables Read  → (Handler) -[READS]→     (Table)
    ## Database Tables Written → (Handler) -[WRITES]→  (Table)

Vertex labels: Service, KafkaTopic, ExternalApi, FeatureFlag,
               Table, ServiceBusEndpoint, Event, Handler
Edge labels:   PUBLISHES, SUBSCRIBES, CALLS, GATED_BY, HANDLES,
               PRODUCES, READS, WRITES, SENDS_TO, RECEIVES_FROM,
               HAS_FK, BELONGS_TO

All Cypher inserts use MERGE (not CREATE) so variant references to the same
concept self-resolve without duplicates.

See CLAUDE.md §D7 for the full vertex/edge schema and tiered extraction strategy.
"""

from dataclasses import dataclass, field


@dataclass
class Triple:
    subject: str        # node name, e.g. "BillRequestHandler"
    subject_label: str  # AGE vertex label, e.g. "Handler"
    predicate: str      # AGE edge label, e.g. "PUBLISHES"
    object: str         # node name, e.g. "dps_labresult"
    object_label: str   # AGE vertex label, e.g. "KafkaTopic"
    properties: dict = field(default_factory=dict)


class GraphExtractor:
    """Extracts knowledge graph triples from structured code summaries."""

    def extract(self, summary: str, document_title: str, service: str) -> list[Triple]:
        """Parse a structured summary and return a list of triples.

        Args:
            summary: The embeddable summary text (post split_summary).
            document_title: File path, used to derive handler/service names.
            service: The source repo name, e.g. "service_A".

        Returns:
            List of Triple objects ready for GraphIndexer.upsert.
        """
        raise NotImplementedError
