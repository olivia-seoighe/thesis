from __future__ import annotations

import re
from pathlib import PurePosixPath

from indexing.graph.config import (
    AST_FEATURE_FLAG_CONFIDENCE,
    AST_OWNERSHIP_CONFIDENCE,
    AST_RELATION_CONFIDENCE,
)
from indexing.graph.csharp_feature_flags import extract_feature_flags_from_csharp
from indexing.graph.graph_canonicalizer import GraphCanonicalizer
from indexing.graph.models import Triple
from indexing.graph.ontology import OWNERSHIP_PREDICATES, TIER_AST_LOCAL
from indexing.graph.text_cleaning import clean_graph_text

SOURCE_KIND_AST = "ast"
EXTRACTOR_NAME = "graph_extractor_v1"
READ_TABLE_PREFIXES: tuple[str, ...] = (
    "GetOrCreate",
    "Query",
    "Get",
    "Find",
    "Fetch",
    "Load",
    "Read",
    "Select",
    "Lookup",
    "Search",
    "List",
    "Has",
    "Any",
    "Count",
)
WRITE_TABLE_PREFIXES: tuple[str, ...] = (
    "AddOrUpdate",
    "CreateOrUpdate",
    "InsertOrUpdate",
    "Add",
    "Create",
    "Update",
    "Delete",
    "Remove",
    "Insert",
    "Upsert",
    "Save",
    "Set",
)
TABLE_ACTION_PREFIXES_BY_PREDICATE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("READS_TABLE", READ_TABLE_PREFIXES),
    ("WRITES_TABLE", WRITE_TABLE_PREFIXES),
)


class AstLocalExtractor:
    _canonicalizer = GraphCanonicalizer()

    def extract(
        self,
        *,
        source_code: str,
        document_title: str,
        repo_name: str,
        known_tables: set[str] | None = None,
    ) -> list[Triple]:
        if not source_code.strip():
            return []

        suffix = PurePosixPath(document_title.lower()).suffix
        if suffix != ".cs":
            return []

        symbols = self._extract_csharp_symbols(source_code)
        triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for symbol_name, symbol_label in symbols:
            owner_predicate = OWNERSHIP_PREDICATES.get(symbol_label)
            if not owner_predicate:
                continue
            self._add_triple(
                triples=triples,
                seen=seen,
                subject=repo_name,
                subject_label="REPO",
                predicate=owner_predicate,
                obj=symbol_name,
                object_label=symbol_label,
                confidence=AST_OWNERSHIP_CONFIDENCE,
                source_path=document_title,
                source_repo=repo_name,
            )

        for (
            subject_name,
            subject_label,
            predicate,
            object_name,
            object_label,
        ) in self._extract_csharp_local_relations(source_code):
            self._add_triple(
                triples=triples,
                seen=seen,
                subject=subject_name,
                subject_label=subject_label,
                predicate=predicate,
                obj=object_name,
                object_label=object_label,
                confidence=AST_RELATION_CONFIDENCE,
                source_path=document_title,
                source_repo=repo_name,
            )

        for actor_name, actor_label, predicate, table_name in self._extract_csharp_actor_table_relations(
            source_code,
            known_tables,
        ):
            self._add_triple(
                triples=triples,
                seen=seen,
                subject=actor_name,
                subject_label=actor_label,
                predicate=predicate,
                obj=table_name,
                object_label="TABLE",
                confidence=AST_RELATION_CONFIDENCE,
                source_path=document_title,
                source_repo=repo_name,
            )

        for flag_name in extract_feature_flags_from_csharp(source_code):
            self._add_triple(
                triples=triples,
                seen=seen,
                subject=repo_name,
                subject_label="REPO",
                predicate="USES_FEATURE_FLAG",
                obj=flag_name,
                object_label="FEATURE_FLAG",
                confidence=AST_FEATURE_FLAG_CONFIDENCE,
                source_path=document_title,
                source_repo=repo_name,
            )

        return triples

    def _add_triple(
        self,
        *,
        triples: list[Triple],
        seen: set[tuple[str, str, str, str, str]],
        subject: str,
        subject_label: str,
        predicate: str,
        obj: str,
        object_label: str,
        confidence: float,
        source_path: str,
        source_repo: str,
    ) -> None:
        clean_subject = clean_graph_text(subject)
        raw_object = clean_graph_text(obj)
        clean_object = raw_object
        if object_label == "KAFKA_TOPIC":
            clean_object = self._canonicalizer.canonicalize_topic_name(clean_object)
        elif object_label == "API":
            clean_object = self._canonicalizer.canonicalize_api_name(clean_object)
        elif object_label == "FEATURE_FLAG":
            clean_object = self._canonicalizer.canonicalize_feature_flag_name(clean_object)
        if not clean_subject or not clean_object:
            return

        key = (clean_subject, subject_label, predicate, clean_object, object_label)
        if key in seen:
            return
        seen.add(key)
        triples.append(
            Triple(
                subject=clean_subject,
                subject_label=subject_label,
                predicate=predicate,
                object=clean_object,
                object_label=object_label,
                properties={
                    "tier": TIER_AST_LOCAL,
                    "source_repo": clean_graph_text(source_repo),
                    "source_path": clean_graph_text(source_path),
                    "source_kind": SOURCE_KIND_AST,
                    "extractor_name": EXTRACTOR_NAME,
                    "confidence": confidence,
                    "raw_object_name": raw_object,
                    "canonical_object_name": clean_object,
                },
            )
        )

    @staticmethod
    def _extract_csharp_symbols(source_code: str) -> list[tuple[str, str]]:
        try:
            from tree_sitter_languages import get_parser  # type: ignore
        except Exception:
            return AstLocalExtractor._extract_csharp_symbols_regex(source_code)

        try:
            parser = get_parser("c_sharp")
            tree = parser.parse(source_code.encode("utf-8"))
        except Exception:
            return AstLocalExtractor._extract_csharp_symbols_regex(source_code)

        symbols: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in {"class_declaration", "record_declaration", "interface_declaration"}:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_code[name_node.start_byte:name_node.end_byte]
                    bases_node = node.child_by_field_name("bases")
                    bases = ""
                    if bases_node:
                        bases = source_code[bases_node.start_byte:bases_node.end_byte]
                    label = AstLocalExtractor._classify_class_symbol(name, bases)
                    if label:
                        key = (name, label)
                        if key not in seen:
                            seen.add(key)
                            symbols.append(key)
            stack.extend(node.children)

        return symbols

    @staticmethod
    def _extract_csharp_symbols_regex(source_code: str) -> list[tuple[str, str]]:
        pattern = re.compile(
            r"^\s*(?:(?:public|internal|private|protected|sealed|abstract|static|partial)\s+)*"
            r"(?:class|record|interface)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^{};]*\))?\s*(?::\s*([^{]+))?\{",
            re.MULTILINE,
        )
        symbols: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for match in pattern.finditer(source_code):
            name = match.group(1)
            bases = match.group(2) or ""
            label = AstLocalExtractor._classify_class_symbol(name, bases)
            if not label:
                continue
            key = (name, label)
            if key in seen:
                continue
            seen.add(key)
            symbols.append(key)
        return symbols

    @staticmethod
    def _classify_symbol(token: str) -> str | None:
        if token.endswith("Handler"):
            return "HANDLER"
        if token.endswith(("Command", "Request")):
            return "COMMAND"
        if token.endswith("Event"):
            return "EVENT"
        return None

    @staticmethod
    def _classify_class_symbol(class_name: str, bases: str) -> str | None:
        if AstLocalExtractor._is_saga_base(bases):
            return "SAGA"
        return AstLocalExtractor._classify_symbol(class_name)

    @staticmethod
    def _is_saga_base(bases: str) -> bool:
        token = (bases or "").strip()
        if not token:
            return False
        return bool(
            re.search(
                r"(?:(?:^|[,\s])(?:global::)?(?:[A-Za-z_][A-Za-z0-9_]*\.)*Saga\s*<)",
                token,
            )
        )

    @staticmethod
    def _extract_csharp_local_relations(source_code: str) -> list[tuple[str, str, str, str, str]]:
        if not source_code.strip():
            return []
        relations: list[tuple[str, str, str, str, str]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        class_header_pattern = re.compile(
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^{};]*\))?\s*(?::\s*([^{]+))?\{",
            flags=re.MULTILINE,
        )
        interface_pattern = re.compile(
            r"\b(IHandleMessages|IAmStartedByMessages)\s*<\s*([A-Za-z_][A-Za-z0-9_.]*)\s*>"
        )
        for class_match in class_header_pattern.finditer(source_code):
            class_name = class_match.group(1)
            bases = class_match.group(2) or ""
            class_label = AstLocalExtractor._classify_class_symbol(class_name, bases)
            if class_label not in {"HANDLER", "SAGA"}:
                continue
            for iface_name, message_type in interface_pattern.findall(bases):
                symbol_name = message_type.split(".")[-1]
                symbol_label = AstLocalExtractor._classify_symbol(symbol_name)
                if not symbol_label:
                    continue
                relation = AstLocalExtractor._resolve_handler_relation(
                    class_label=class_label,
                    iface_name=iface_name,
                    symbol_label=symbol_label,
                )
                if relation is None:
                    continue
                predicate, object_label = relation
                key = (class_name, class_label, predicate, symbol_name, object_label)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(key)
        return relations

    @staticmethod
    def _extract_csharp_actor_table_relations(
        source_code: str,
        known_tables: set[str] | None,
    ) -> list[tuple[str, str, str, str]]:
        if not source_code.strip() or not known_tables:
            return []

        table_lookup = AstLocalExtractor._build_known_table_lookup(known_tables)
        if not table_lookup:
            return []

        relations: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for actor_name, actor_label, actor_body in AstLocalExtractor._extract_csharp_actor_blocks(source_code):
            reads, writes = AstLocalExtractor._collect_actor_reads_writes(actor_body, table_lookup)
            AstLocalExtractor._append_actor_table_relations(
                relations=relations,
                seen=seen,
                actor_name=actor_name,
                actor_label=actor_label,
                reads=reads,
                writes=writes,
            )

        return relations

    @staticmethod
    def _extract_csharp_actor_blocks(source_code: str) -> list[tuple[str, str, str]]:
        blocks: list[tuple[str, str, str]] = []
        class_pattern = re.compile(
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^{};]*\))?\s*(?::\s*([^{]+))?\{",
            re.MULTILINE,
        )
        for match in class_pattern.finditer(source_code):
            class_name = match.group(1)
            bases = match.group(2) or ""
            class_label = AstLocalExtractor._classify_class_symbol(class_name, bases)
            if class_label not in {"HANDLER", "SAGA"}:
                continue
            body_start = match.end() - 1
            body_end = AstLocalExtractor._find_matching_brace(source_code, body_start)
            if body_end <= body_start:
                actor_body = source_code[body_start:]
            else:
                actor_body = source_code[body_start:body_end + 1]
            blocks.append((class_name, class_label, actor_body))
        return blocks

    @staticmethod
    def _find_matching_brace(source_code: str, open_brace_index: int) -> int:
        depth = 0
        for idx in range(open_brace_index, len(source_code)):
            token = source_code[idx]
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    @staticmethod
    def _extract_table_action_target(identifier: str) -> list[tuple[str, str]]:
        compact = re.sub(r"[^A-Za-z0-9_]", "", identifier)
        if not compact:
            return []

        candidates: list[tuple[str, str]] = []
        for predicate, prefixes in TABLE_ACTION_PREFIXES_BY_PREDICATE:
            for prefix in sorted(prefixes, key=len, reverse=True):
                if not compact.startswith(prefix):
                    continue
                target = compact[len(prefix):]
                target = re.sub(
                    r"(Command|Commands|Event|Events|Request|Requests|Handler|Handlers|Saga|Sagas|Repository|Repositories|Record|Records|Entity|Entities|Dto|Model|Models)$",
                    "",
                    target,
                )
                if len(target) < 3:
                    continue
                candidates.append((predicate, target))
        return candidates

    @staticmethod
    def _build_known_table_lookup(known_tables: set[str]) -> dict[str, set[str]]:
        lookup: dict[str, set[str]] = {}
        for table_name in known_tables:
            for variant in AstLocalExtractor._table_name_variants(table_name):
                if not variant:
                    continue
                lookup.setdefault(variant, set()).add(table_name)
        return lookup

    @staticmethod
    def _resolve_table_name(table_token: str, table_lookup: dict[str, set[str]]) -> str | None:
        for variant in AstLocalExtractor._table_name_variants(table_token):
            matches = table_lookup.get(variant, set())
            if len(matches) == 1:
                return next(iter(matches))
        return None

    @staticmethod
    def _table_name_variants(token: str) -> tuple[str, ...]:
        normalized = re.sub(r"[^A-Za-z0-9]", "", token).lower()
        if not normalized:
            return ()
        variants = [normalized]
        if normalized.endswith("ies") and len(normalized) > 3:
            variants.append(normalized[:-3] + "y")
        if normalized.endswith("es") and len(normalized) > 2:
            variants.append(normalized[:-2])
        if normalized.endswith("s") and len(normalized) > 1:
            variants.append(normalized[:-1])
        return tuple(dict.fromkeys(variants))

    @staticmethod
    def _resolve_handler_relation(
        *,
        class_label: str,
        iface_name: str,
        symbol_label: str,
    ) -> tuple[str, str] | None:
        if class_label == "HANDLER":
            if symbol_label == "COMMAND":
                return ("HANDLES_COMMAND", "COMMAND")
            if symbol_label == "EVENT":
                return ("HANDLES_EVENT", "EVENT")
            return None
        if class_label == "SAGA":
            if iface_name == "IAmStartedByMessages" and symbol_label == "COMMAND":
                return ("SAGA_ORCHESTRATES_COMMAND", "COMMAND")
            if iface_name == "IHandleMessages" and symbol_label == "EVENT":
                return ("SAGA_AWAITS_EVENT", "EVENT")
        return None

    @staticmethod
    def _collect_actor_reads_writes(
        actor_body: str,
        table_lookup: dict[str, set[str]],
    ) -> tuple[set[str], set[str]]:
        reads: set[str] = set()
        writes: set[str] = set()
        AstLocalExtractor._collect_sql_like_table_access(actor_body, table_lookup, reads, writes)
        AstLocalExtractor._collect_identifier_table_actions(actor_body, table_lookup, reads, writes)
        AstLocalExtractor._collect_dbset_table_access(actor_body, table_lookup, reads, writes)
        return reads, writes

    @staticmethod
    def _collect_sql_like_table_access(
        actor_body: str,
        table_lookup: dict[str, set[str]],
        reads: set[str],
        writes: set[str],
    ) -> None:
        for table_token in re.findall(
            r"(?i)\b(?:from|join)\s+[\[`\"]?([A-Za-z_][A-Za-z0-9_]*)[\]`\"]?",
            actor_body,
        ):
            resolved = AstLocalExtractor._resolve_table_name(table_token, table_lookup)
            if resolved:
                reads.add(resolved)

        for table_token in re.findall(
            r"(?i)\b(?:insert\s+into|update|delete\s+from)\s+[\[`\"]?([A-Za-z_][A-Za-z0-9_]*)[\]`\"]?",
            actor_body,
        ):
            resolved = AstLocalExtractor._resolve_table_name(table_token, table_lookup)
            if resolved:
                writes.add(resolved)

    @staticmethod
    def _collect_identifier_table_actions(
        actor_body: str,
        table_lookup: dict[str, set[str]],
        reads: set[str],
        writes: set[str],
    ) -> None:
        identifier_pattern = re.compile(r"\b(?:new\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
        for identifier in identifier_pattern.findall(actor_body):
            for predicate, table_token in AstLocalExtractor._extract_table_action_target(identifier):
                resolved = AstLocalExtractor._resolve_table_name(table_token, table_lookup)
                if not resolved:
                    continue
                if predicate == "READS_TABLE":
                    reads.add(resolved)
                else:
                    writes.add(resolved)

    @staticmethod
    def _collect_dbset_table_access(
        actor_body: str,
        table_lookup: dict[str, set[str]],
        reads: set[str],
        writes: set[str],
    ) -> None:
        dbset_pattern = re.compile(
            r"\b[_A-Za-z][_A-Za-z0-9]*DbContext[_A-Za-z0-9]*\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b",
            re.IGNORECASE,
        )
        for dbset in dbset_pattern.findall(actor_body):
            resolved = AstLocalExtractor._resolve_table_name(dbset, table_lookup)
            if not resolved:
                continue
            reads.add(resolved)
            write_call_pattern = re.compile(
                rf"\b{re.escape(dbset)}\s*\.\s*(?:Add|AddAsync|AddRange|Update|UpdateRange|Remove|RemoveRange|ExecuteUpdate|ExecuteDelete)\s*\(",
                re.IGNORECASE,
            )
            if write_call_pattern.search(actor_body):
                writes.add(resolved)

    @staticmethod
    def _append_actor_table_relations(
        *,
        relations: list[tuple[str, str, str, str]],
        seen: set[tuple[str, str, str, str]],
        actor_name: str,
        actor_label: str,
        reads: set[str],
        writes: set[str],
    ) -> None:
        for predicate, table_names in (("READS_TABLE", sorted(reads)), ("WRITES_TABLE", sorted(writes))):
            for table_name in table_names:
                key = (actor_name, actor_label, predicate, table_name)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(key)
