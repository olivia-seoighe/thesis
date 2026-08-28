"""Coordinator for AST-local and contract-global graph extraction."""

from __future__ import annotations

from pathlib import PurePosixPath

from indexing.graph.ast_extractor import AstLocalExtractor
from indexing.graph.contract_extractor import (
    CONTRACT_OR_CONFIG_HINTS,
    SECTION_CONFIGURATION,
    SECTION_EXTERNAL_APIS,
    SECTION_FEATURE_FLAGS,
    SECTION_PUBLIC_API,
    SECTION_TOPICS_CONSUMED,
    SECTION_TOPICS_PRODUCED,
    AST_EXTENSIONS,
    ContractGlobalExtractor,
)
from indexing.graph.models import Triple
from indexing.graph.text_cleaning import clean_graph_text


class GraphExtractor:
    """Extract triples from source code and structured contract/config content."""

    def __init__(self) -> None:
        self._ast_extractor = AstLocalExtractor()
        self._contract_extractor = ContractGlobalExtractor()
        self._repo_owned_tables: dict[str, set[str]] = {}

    def register_repo_files(self, repo_name: str, file_paths: list[str]) -> None:
        self._contract_extractor.register_repo_files(repo_name, file_paths)

    def extract(
        self,
        summary: str,
        document_title: str,
        service: str,
        source_code: str = "",
    ) -> list[Triple]:
        repo_name = clean_graph_text(service)
        if not repo_name:
            return []

        document_kind = self._classify_document_kind(document_title)
        sections = self._contract_extractor.parse_sections(summary) if summary.strip() else {}

        if document_kind == "CONTRACT":
            contract_triples = self._contract_extractor.extract_contract(
                summary=summary,
                document_title=document_title,
                repo_name=repo_name,
                source_code=source_code,
            )
            self._record_repo_owned_tables(repo_name, contract_triples)
            return contract_triples

        if document_kind == "AST":
            ast_triples = self._ast_extractor.extract(
                source_code=source_code,
                document_title=document_title,
                repo_name=repo_name,
                known_tables=self._repo_owned_tables.get(repo_name),
            )
            if self._has_contract_sections(sections) and self._should_extract_contract_from_ast(document_title):
                contract_triples = self._contract_extractor.extract_contract(
                    summary=summary,
                    document_title=document_title,
                    repo_name=repo_name,
                    source_code=source_code,
                )
                self._record_repo_owned_tables(repo_name, contract_triples)
                return ast_triples + contract_triples
            return ast_triples

        return []

    def _record_repo_owned_tables(self, repo_name: str, triples: list[Triple]) -> None:
        for triple in triples:
            if triple.predicate != "OWNS_TABLE" or triple.object_label != "TABLE":
                continue
            self._repo_owned_tables.setdefault(repo_name, set()).add(triple.object)

    @staticmethod
    def _classify_document_kind(document_title: str) -> str:
        lower_title = document_title.lower()
        if lower_title.endswith("_index.md"):
            return "SKIP"
        if any(token in lower_title for token in CONTRACT_OR_CONFIG_HINTS):
            return "CONTRACT"
        suffix = PurePosixPath(lower_title).suffix
        if suffix in AST_EXTENSIONS:
            return "AST"
        return "SKIP"

    @staticmethod
    def _has_contract_sections(sections: dict[str, str]) -> bool:
        contract_sections = {
            SECTION_TOPICS_CONSUMED,
            SECTION_TOPICS_PRODUCED,
            SECTION_EXTERNAL_APIS,
            SECTION_PUBLIC_API,
            SECTION_FEATURE_FLAGS,
            SECTION_CONFIGURATION,
        }
        return any(name in sections for name in contract_sections)

    @staticmethod
    def _should_extract_contract_from_ast(document_title: str) -> bool:
        lower_title = document_title.lower()
        filename = PurePosixPath(lower_title).name
        return any(
            token in filename
            for token in (
                "servicecollection",
                "startup",
                "program.cs",
                "dependencyinjection",
                "extensions",
            )
        )
