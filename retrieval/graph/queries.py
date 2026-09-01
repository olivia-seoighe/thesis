"""Apache AGE cypher() query template builders."""

from __future__ import annotations

import json

from .config import GRAPH_NAME, SEED_LOOKUP_LIMIT
from .predicate_catalog import allowed_predicates
from .types import CypherSpec, EntityMention, QueryIntent


def build_seed_lookup_query(seed: EntityMention) -> CypherSpec:
    sql = f"""
        WITH graph_hits AS (
            SELECT
                TRIM(BOTH '"' FROM node_key::text) AS node_key,
                TRIM(BOTH '"' FROM node_label::text) AS node_label,
                TRIM(BOTH '"' FROM node_name::text) AS node_name
            FROM ag_catalog.cypher('{GRAPH_NAME}', $$
                MATCH (n)
                WHERE ($label = '' OR n.label = $label)
                  AND (
                    toLower(n.name) = toLower($name)
                    OR toLower(n.name) CONTAINS toLower($name)
                    OR ($label = '' AND toLower(n.node_key) CONTAINS toLower($name))
                    OR replace(toLower(n.name), '_', '-') = replace(toLower($name), '_', '-')
                    OR replace(toLower(n.name), '_', '-') CONTAINS replace(toLower($name), '_', '-')
                    OR (
                      $label = ''
                      AND replace(toLower(n.node_key), '_', '-') CONTAINS replace(toLower($name), '_', '-')
                    )
                  )
                RETURN n.node_key AS node_key, n.label AS node_label, n.name AS node_name
                LIMIT toInteger($limit)
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
        ORDER BY confidence DESC, evidence_count DESC
    """
    params = _agtype_param(
        {
            "label": seed.preferred_label or "",
            "name": seed.text,
            "limit": SEED_LOOKUP_LIMIT,
        }
    )
    return CypherSpec(query=sql, args=(params,))


def build_label_seed_lookup_query(*, label: str) -> CypherSpec:
    sql = f"""
        WITH graph_hits AS (
            SELECT
                TRIM(BOTH '"' FROM node_key::text) AS node_key,
                TRIM(BOTH '"' FROM node_label::text) AS node_label,
                TRIM(BOTH '"' FROM node_name::text) AS node_name
            FROM ag_catalog.cypher('{GRAPH_NAME}', $$
                MATCH (n)
                WHERE n.label = $label
                RETURN n.node_key AS node_key, n.label AS node_label, n.name AS node_name
                LIMIT toInteger($limit)
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
        ORDER BY confidence DESC, evidence_count DESC
    """
    params = _agtype_param(
        {
            "label": label,
            "limit": SEED_LOOKUP_LIMIT,
        }
    )
    return CypherSpec(query=sql, args=(params,))


def build_frontier_expansion_query(
    *,
    intent: QueryIntent,
    frontier_node_keys: list[str],
    edge_limit: int,
) -> CypherSpec:
    allowed_predicates_for_intent = allowed_predicates(intent)
    sql = f"""
        WITH frontier_edges AS (
            SELECT
                TRIM(BOTH '"' FROM subject_key::text) AS subject_key,
                TRIM(BOTH '"' FROM object_key::text) AS object_key,
                TRIM(BOTH '"' FROM predicate::text) AS predicate,
                TRIM(BOTH '"' FROM edge_key::text) AS edge_key
            FROM ag_catalog.cypher('{GRAPH_NAME}', $$
                MATCH (n)-[rel]-(m)
                WHERE n.node_key IN $frontier_node_keys
                  AND (size($allowed_predicates) = 0 OR rel.predicate IN $allowed_predicates)
                RETURN startNode(rel).node_key AS subject_key,
                       endNode(rel).node_key AS object_key,
                       rel.predicate AS predicate,
                       rel.edge_key AS edge_key
            $$, $1::ag_catalog.agtype) AS (
                subject_key ag_catalog.agtype,
                object_key ag_catalog.agtype,
                predicate ag_catalog.agtype,
                edge_key ag_catalog.agtype
            )
        ),
        edge_stats AS (
            SELECT
                edge_key,
                MAX(confidence) AS confidence,
                (ARRAY_AGG(tier ORDER BY confidence DESC NULLS LAST))[1] AS tier,
                (ARRAY_AGG(source_kind ORDER BY confidence DESC NULLS LAST))[1] AS source_kind,
                (ARRAY_AGG(line_start ORDER BY confidence DESC NULLS LAST))[1] AS line_start,
                (ARRAY_AGG(line_end ORDER BY confidence DESC NULLS LAST))[1] AS line_end
            FROM graph_edge_evidence
            GROUP BY edge_key
        )
        SELECT
            f.subject_key,
            f.object_key,
            f.predicate,
            f.edge_key,
            COALESCE(es.confidence, 1.0) AS confidence,
            COALESCE(es.tier, 'CONTRACT_GLOBAL') AS tier,
            COALESCE(es.source_kind, 'unknown') AS source_kind,
            es.line_start,
            es.line_end
        FROM frontier_edges f
        LEFT JOIN edge_stats es ON es.edge_key = f.edge_key
        ORDER BY
            COALESCE(es.confidence, 1.0) DESC,
            f.subject_key,
            f.object_key
        LIMIT $2
    """
    params = _agtype_param(
        {
            "frontier_node_keys": frontier_node_keys,
            "allowed_predicates": allowed_predicates_for_intent,
        }
    )
    return CypherSpec(query=sql, args=(params, edge_limit))


def build_evidence_query(node_or_edge_ids: list[str], retrieval_corpus: str) -> CypherSpec:
    sql = f"""
        WITH resolved_nodes AS (
            SELECT DISTINCT TRIM(BOTH '"' FROM node_key::text) AS node_key
            FROM ag_catalog.cypher('{GRAPH_NAME}', $$
                MATCH (n)
                WHERE n.node_key IN $node_keys
                RETURN n.node_key AS node_key
            $$, $1::ag_catalog.agtype) AS (node_key ag_catalog.agtype)
        ),
        evidence_stats AS (
            SELECT
                gne.node_key,
                MAX(gne.confidence) AS max_confidence,
                COUNT(*)::int AS evidence_count,
                ARRAY_AGG(DISTINCT gne.source_kind) AS source_kinds,
                ARRAY_AGG(DISTINCT gne.tier) AS tiers
            FROM graph_node_evidence gne
            JOIN resolved_nodes rn ON rn.node_key = gne.node_key
            WHERE gne.document_id IS NOT NULL
            GROUP BY gne.node_key
        ),
        ranked_evidence AS (
            SELECT
                gne.node_key,
                gne.node_label,
                gne.node_name,
                gne.tier,
                gne.source_kind,
                gne.line_start,
                gne.line_end,
                gne.document_id,
                gne.chunk_id,
                ROW_NUMBER() OVER (
                    PARTITION BY gne.node_key
                    ORDER BY
                        CASE gne.source_kind
                            WHEN 'asyncapi' THEN 0
                            WHEN 'configmap' THEN 1
                            WHEN 'appsettings_prod' THEN 2
                            WHEN 'appsettings_base' THEN 3
                            WHEN 'contract' THEN 4
                            WHEN 'ingress' THEN 5
                            WHEN 'ast' THEN 6
                            ELSE 7
                        END,
                        gne.confidence DESC,
                        gne.observed_at DESC
                ) AS rn
            FROM graph_node_evidence gne
            JOIN resolved_nodes rn ON rn.node_key = gne.node_key
            WHERE gne.document_id IS NOT NULL
        )
        SELECT
            re.node_key,
            re.node_label,
            re.node_name,
            es.max_confidence,
            es.evidence_count,
            es.source_kinds,
            es.tiers,
            (re.line_start IS NOT NULL AND re.line_end IS NOT NULL) AS has_line_bounds,
            COALESCE(re.chunk_id, de.chunk_id) AS resolved_chunk_id,
            de.text,
            de.source_code,
            de.document_id,
            COALESCE(dm.name, de.document_title, re.node_name) AS document_title,
            COALESCE(dm.url, '') AS url,
            COALESCE(dm.last_modified_date::text, '') AS last_modified_date,
            COALESCE(de.source, '') AS source,
            COALESCE((de.metadata)::jsonb, '{{}}'::jsonb) AS metadata
        FROM ranked_evidence re
        LEFT JOIN LATERAL (
            SELECT chunk_id
            FROM document_embeddings de2
            WHERE de2.document_id = re.document_id
              AND de2.retrieval_corpus = $2
            ORDER BY de2.chunk_id
            LIMIT 1
        ) fallback ON TRUE
        JOIN document_embeddings de
          ON de.chunk_id = COALESCE(re.chunk_id, fallback.chunk_id)
          AND de.retrieval_corpus = $2
        JOIN evidence_stats es ON es.node_key = re.node_key
        LEFT JOIN document_metadata dm ON dm.document_id = de.document_id AND dm.retrieval_corpus = de.retrieval_corpus
        WHERE re.rn <= 5
    """
    params = _agtype_param({"node_keys": node_or_edge_ids})
    return CypherSpec(query=sql, args=(params, retrieval_corpus))


def _agtype_param(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
