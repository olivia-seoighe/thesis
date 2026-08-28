"""Apache AGE graph indexer with canonical MERGE + provenance evidence upserts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from indexing.graph.config import GRAPH_NAME
from indexing.graph.models import Triple
from indexing.graph.ontology import ALLOWED_TIERS, EDGE_LABELS, LOCAL_SCOPED_NODE_LABELS, NODE_LABELS

UPSERT_NODE_EVIDENCE_SQL = """
    INSERT INTO graph_node_evidence (
        node_key, node_label, node_name, tier, source_repo, source_path, source_kind,
        line_start, line_end, extractor_name, confidence, evidence_hash, observed_at,
        document_id, chunk_id
    )
    VALUES (
        $1, $2, $3, $4, $5, $6, $7,
        $8, $9, $10, $11, $12, NOW(),
        $13, $14
    )
    ON CONFLICT (node_key, evidence_hash) DO UPDATE SET
        confidence = EXCLUDED.confidence,
        observed_at = NOW(),
        document_id = COALESCE(EXCLUDED.document_id, graph_node_evidence.document_id),
        chunk_id = COALESCE(EXCLUDED.chunk_id, graph_node_evidence.chunk_id)
"""

UPSERT_EDGE_EVIDENCE_SQL = """
    INSERT INTO graph_edge_evidence (
        edge_key, predicate, subject_key, subject_label, subject_name,
        object_key, object_label, object_name, tier, source_repo, source_path, source_kind,
        line_start, line_end, extractor_name, confidence, evidence_hash, observed_at,
        document_id, chunk_id
    )
    VALUES (
        $1, $2, $3, $4, $5,
        $6, $7, $8, $9, $10, $11, $12,
        $13, $14, $15, $16, $17, NOW(),
        $18, $19
    )
    ON CONFLICT (edge_key, evidence_hash) DO UPDATE SET
        confidence = EXCLUDED.confidence,
        observed_at = NOW(),
        document_id = COALESCE(EXCLUDED.document_id, graph_edge_evidence.document_id),
        chunk_id = COALESCE(EXCLUDED.chunk_id, graph_edge_evidence.chunk_id)
"""

_SAFE_LABEL_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class _CanonicalNode:
    key: str
    label: str
    name: str


@dataclass(frozen=True)
class _CanonicalEdge:
    key: str
    predicate: str
    subject: _CanonicalNode
    obj: _CanonicalNode


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
        if not triples:
            return

        nodes: dict[str, _CanonicalNode] = {}
        edges: dict[str, _CanonicalEdge] = {}
        node_evidence_rows: list[tuple] = []
        edge_evidence_rows: list[tuple] = []

        for triple in triples:
            canonical_edge, row_node_subject, row_node_object = self._build_rows(triple)
            nodes[row_node_subject.key] = row_node_subject
            nodes[row_node_object.key] = row_node_object
            edges[canonical_edge.key] = canonical_edge
            node_evidence_rows.extend([
                self._build_node_evidence_row(row_node_subject, triple.properties),
                self._build_node_evidence_row(row_node_object, triple.properties),
            ])
            edge_evidence_rows.append(
                self._build_edge_evidence_row(canonical_edge, triple.properties)
            )

        for node in nodes.values():
            try:
                await self._merge_node(node)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to merge node "
                    f"label={node.label} key={node.key!r} name={node.name!r} "
                    f"lookup={self._build_node_lookup(node)!r}"
                ) from exc

        for edge in edges.values():
            try:
                await self._merge_edge(edge)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to merge edge "
                    f"predicate={edge.predicate} key={edge.key!r} "
                    f"subject={edge.subject.key!r} object={edge.obj.key!r}"
                ) from exc

        # Deduplicate evidence rows by identity tuple before batch-upsert.
        deduped_node_rows = self._dedupe_rows(node_evidence_rows, key_idx=(0, 11))
        deduped_edge_rows = self._dedupe_rows(edge_evidence_rows, key_idx=(0, 16))

        if deduped_node_rows:
            await self.cm.executemany(UPSERT_NODE_EVIDENCE_SQL, deduped_node_rows)
        if deduped_edge_rows:
            await self.cm.executemany(UPSERT_EDGE_EVIDENCE_SQL, deduped_edge_rows)

    async def upsert_service_node(self, service: str) -> None:
        """Ensure a top-level REPO node exists for the given service.

        Args:
            service: Service name, e.g. "service_A".
        """
        name = self._clean(service)
        if not name:
            return
        node = _CanonicalNode(key=self._node_key("REPO", name), label="REPO", name=name)
        await self._merge_node(node)

    async def _merge_node(self, node: _CanonicalNode) -> None:
        self._validate_label(node.label, allowed=NODE_LABELS)
        lookup = self._build_node_lookup(node)
        existing = await self._fetch_cypher(lookup, "node_key ag_catalog.agtype")
        if existing:
            return

        create = (
            f"CREATE (n:{node.label} "
            f"{{node_key: '{self._quote(node.key)}', name: '{self._quote(node.name)}', label: '{self._quote(node.label)}'}})"
        )
        await self._execute_cypher(create)

    async def _merge_edge(self, edge: _CanonicalEdge) -> None:
        self._validate_label(edge.subject.label, allowed=NODE_LABELS)
        self._validate_label(edge.obj.label, allowed=NODE_LABELS)
        self._validate_label(edge.predicate, allowed=EDGE_LABELS)
        lookup = (
            f"MATCH (s:{edge.subject.label})-[r:{edge.predicate} {{edge_key: '{self._quote(edge.key)}'}}]->(o:{edge.obj.label}) "
            "RETURN r.edge_key AS edge_key LIMIT 1"
        )
        existing = await self._fetch_cypher(lookup, "edge_key ag_catalog.agtype")
        if existing:
            return

        create = (
            f"MATCH (s:{edge.subject.label} {{node_key: '{self._quote(edge.subject.key)}'}}) "
            f"MATCH (o:{edge.obj.label} {{node_key: '{self._quote(edge.obj.key)}'}}) "
            f"CREATE (s)-[r:{edge.predicate} "
            f"{{edge_key: '{self._quote(edge.key)}', predicate: '{self._quote(edge.predicate)}'}}]->(o)"
        )
        await self._execute_cypher(create)

    async def _execute_cypher(self, cypher: str) -> None:
        sql = (
            f"SELECT * FROM ag_catalog.cypher('{GRAPH_NAME}', $$ {cypher} $$, $1::ag_catalog.agtype) "
            "AS (v ag_catalog.agtype);"
        )
        await self.cm.execute(sql, "{}")

    def _build_node_lookup(self, node: _CanonicalNode) -> str:
        return (
            f"MATCH (n:{node.label} {{node_key: '{self._quote(node.key)}'}}) "
            "RETURN n.node_key AS node_key LIMIT 1"
        )

    async def _fetch_cypher(self, cypher: str, projection: str) -> list:
        sql = (
            f"SELECT * FROM ag_catalog.cypher('{GRAPH_NAME}', $$ {cypher} $$, $1::ag_catalog.agtype) "
            f"AS ({projection});"
        )
        return await self.cm.fetch(sql, "{}")

    def _build_rows(
        self, triple: Triple
    ) -> tuple[_CanonicalEdge, _CanonicalNode, _CanonicalNode]:
        subject_name = self._clean(triple.subject)
        object_name = self._clean(triple.object)
        subject_label = self._clean(triple.subject_label).upper()
        object_label = self._clean(triple.object_label).upper()
        predicate = self._clean(triple.predicate).upper()
        source_repo = self._required_prop(triple.properties, "source_repo")
        if not subject_name or not object_name:
            raise ValueError("Triple subject/object names must be non-empty.")

        self._validate_label(subject_label, allowed=NODE_LABELS)
        self._validate_label(object_label, allowed=NODE_LABELS)
        self._validate_label(predicate, allowed=EDGE_LABELS)

        subject_node = _CanonicalNode(
            key=self._node_key(subject_label, subject_name, source_repo=source_repo),
            label=subject_label,
            name=subject_name,
        )
        object_node = _CanonicalNode(
            key=self._node_key(object_label, object_name, source_repo=source_repo),
            label=object_label,
            name=object_name,
        )
        edge = _CanonicalEdge(
            key=self._edge_key(subject_node.key, predicate, object_node.key),
            predicate=predicate,
            subject=subject_node,
            obj=object_node,
        )
        return edge, subject_node, object_node

    def _build_node_evidence_row(self, node: _CanonicalNode, props: dict) -> tuple:
        tier = self._tier(props)
        source_repo = self._required_prop(props, "source_repo")
        source_path = self._required_prop(props, "source_path")
        source_kind = self._required_prop(props, "source_kind")
        extractor_name = self._required_prop(props, "extractor_name")
        confidence = self._confidence(props)
        line_start = self._optional_int(props.get("line_start"))
        line_end = self._optional_int(props.get("line_end"))
        document_id = self._optional_text(props.get("document_id"))
        chunk_id = self._optional_text(props.get("chunk_id"))
        evidence_hash = self._evidence_hash(
            {
                "type": "node",
                "node_key": node.key,
                "tier": tier,
                "source_repo": source_repo,
                "source_path": source_path,
                "source_kind": source_kind,
                "line_start": line_start,
                "line_end": line_end,
            }
        )
        return (
            node.key,
            node.label,
            node.name,
            tier,
            source_repo,
            source_path,
            source_kind,
            line_start,
            line_end,
            extractor_name,
            confidence,
            evidence_hash,
            document_id,
            chunk_id,
        )

    def _build_edge_evidence_row(self, edge: _CanonicalEdge, props: dict) -> tuple:
        tier = self._tier(props)
        source_repo = self._required_prop(props, "source_repo")
        source_path = self._required_prop(props, "source_path")
        source_kind = self._required_prop(props, "source_kind")
        extractor_name = self._required_prop(props, "extractor_name")
        confidence = self._confidence(props)
        line_start = self._optional_int(props.get("line_start"))
        line_end = self._optional_int(props.get("line_end"))
        document_id = self._optional_text(props.get("document_id"))
        chunk_id = self._optional_text(props.get("chunk_id"))
        evidence_hash = self._evidence_hash(
            {
                "type": "edge",
                "edge_key": edge.key,
                "tier": tier,
                "source_repo": source_repo,
                "source_path": source_path,
                "source_kind": source_kind,
                "line_start": line_start,
                "line_end": line_end,
            }
        )
        return (
            edge.key,
            edge.predicate,
            edge.subject.key,
            edge.subject.label,
            edge.subject.name,
            edge.obj.key,
            edge.obj.label,
            edge.obj.name,
            tier,
            source_repo,
            source_path,
            source_kind,
            line_start,
            line_end,
            extractor_name,
            confidence,
            evidence_hash,
            document_id,
            chunk_id,
        )

    @staticmethod
    def _dedupe_rows(rows: list[tuple], key_idx: tuple[int, int]) -> list[tuple]:
        deduped: dict[tuple[str, str], tuple] = {}
        idx1, idx2 = key_idx
        for row in rows:
            deduped[(row[idx1], row[idx2])] = row
        return list(deduped.values())

    @staticmethod
    def _node_key(label: str, name: str, *, source_repo: str | None = None) -> str:
        if label in LOCAL_SCOPED_NODE_LABELS:
            if not source_repo:
                raise ValueError(f"source_repo is required for local-scoped label: {label}")
            return f"{label}::{source_repo}::{name}"
        return f"{label}::{name}"

    @staticmethod
    def _edge_key(subject_key: str, predicate: str, object_key: str) -> str:
        return f"{subject_key}::{predicate}::{object_key}"

    @staticmethod
    def _quote(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip()

    @staticmethod
    def _required_prop(props: dict, key: str) -> str:
        value = str(props.get(key, "")).strip()
        if not value:
            raise ValueError(f"Missing required triple property: {key}")
        return value

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        return token or None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _tier(props: dict) -> str:
        value = str(props.get("tier", "")).strip().upper()
        if value not in ALLOWED_TIERS:
            raise ValueError(f"Unsupported tier: {value}")
        return value

    @staticmethod
    def _confidence(props: dict) -> float:
        value = float(props.get("confidence", 1.0))
        if value <= 0:
            raise ValueError("confidence must be > 0")
        return value

    @staticmethod
    def _evidence_hash(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_label(label: str, *, allowed: set[str]) -> None:
        if label not in allowed:
            raise ValueError(f"Unsupported graph label/predicate: {label}")
        if not _SAFE_LABEL_RE.match(label):
            raise ValueError(f"Unsafe graph label/predicate: {label}")
