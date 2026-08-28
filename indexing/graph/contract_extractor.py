"""Deterministic graph triple extractor for the two-tier graph model.

Design rules:
- AST_LOCAL triples are emitted only from source-code parsing.
- CONTRACT_GLOBAL triples are emitted from contract/config summaries.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import Callable

from indexing.graph.config import (
    CONTRACT_API_CONFIDENCE,
    CONTRACT_CSPROJ_METADATA_CONFIDENCE,
    CONTRACT_EXPOSES_API_CONFIDENCE,
    CONTRACT_FLAG_CONFIDENCE,
    CONTRACT_TABLE_CONFIDENCE,
    CONTRACT_TOPIC_CONFIDENCE,
    SOURCE_PRIORITY_APPSETTINGS_BASE,
    SOURCE_PRIORITY_APPSETTINGS_PROD,
    SOURCE_PRIORITY_AST,
    SOURCE_PRIORITY_ASYNCAPI,
    SOURCE_PRIORITY_CONFIGMAP,
    SOURCE_PRIORITY_DEFAULT,
    SOURCE_PRIORITY_INGRESS,
)
from indexing.graph.csharp_feature_flags import extract_feature_flags_from_csharp
from indexing.graph.graph_canonicalizer import GraphCanonicalizer
from indexing.graph.models import Triple
from indexing.graph.ontology import TIER_CONTRACT_GLOBAL
from indexing.graph.text_cleaning import clean_graph_text

SOURCE_KIND_AST = "ast"
SOURCE_KIND_CONTRACT = "contract"
SOURCE_KIND_ASYNCAPI = "asyncapi"
SOURCE_KIND_CONFIGMAP = "configmap"
SOURCE_KIND_APPSETTINGS_PROD = "appsettings_prod"
SOURCE_KIND_APPSETTINGS_BASE = "appsettings_base"
SOURCE_KIND_INGRESS = "ingress"
SOURCE_KIND_CSPROJ = "csproj"
EXTRACTOR_NAME = "graph_extractor_v1"
AST_EXTENSIONS: tuple[str, ...] = (".cs",)
CONTRACT_OR_CONFIG_HINTS: tuple[str, ...] = (
    "schema_state_",
    "asyncapi",
    "configmap",
    "appsettings",
    "ingress",
    ".csproj",
)

SECTION_CONFIGURATION = "configuration"
SECTION_FEATURE_FLAGS = "feature flags"
SECTION_TOPICS_CONSUMED = "topics consumed"
SECTION_TOPICS_PRODUCED = "topics produced"
SECTION_EXTERNAL_APIS = "external api calls"
SECTION_PUBLIC_API = "public api surface"

TOPIC_KEYS = {"topics", "kafkatopic", "kafkatopics", "consumers", "producers"}
URL_SKIP_KEY_FRAGMENTS = ("okta", "health", "relic", "pdp", "swagger")
URL_SKIP_HOST_FRAGMENTS = ("okta", "login", "auth", "identity")

class ContractGlobalExtractor:
    """Extracts graph triples from source code and structured summary content."""

    _canonicalizer = GraphCanonicalizer()

    def __init__(self) -> None:
        self._repo_source_kinds: dict[str, set[str]] = {}
        self._repo_entity_priorities: dict[str, dict[str, dict[str, int]]] = {}

    def register_repo_files(self, repo_name: str, file_paths: list[str]) -> None:
        repo = clean_graph_text(repo_name)
        if not repo:
            return
        kinds: set[str] = set()
        for path in file_paths:
            kind = self._derive_contract_source_kind(path)
            if kind != SOURCE_KIND_CONTRACT:
                kinds.add(kind)
        self._repo_source_kinds[repo] = kinds

    def extract_contract(
        self,
        *,
        summary: str,
        document_title: str,
        repo_name: str,
        source_code: str = "",
    ) -> list[Triple]:
        sections = self.parse_sections(summary) if summary.strip() else {}
        return self._extract_contract_global(
            summary_text=summary,
            sections=sections,
            document_title=document_title,
            repo_name=repo_name,
            source_code=source_code,
        )

    def parse_sections(self, summary: str) -> dict[str, str]:
        return self._parse_sections(summary)

    def extract(
        self,
        summary: str,
        document_title: str,
        service: str,
        source_code: str = "",
    ) -> list[Triple]:
        """Backward-compatible wrapper around contract extraction."""
        repo_name = clean_graph_text(service)
        if not repo_name:
            return []
        return self.extract_contract(
            summary=summary,
            document_title=document_title,
            repo_name=repo_name,
            source_code=source_code,
        )

    def _extract_contract_global(
        self,
        *,
        summary_text: str,
        sections: dict[str, str],
        document_title: str,
        repo_name: str,
        source_code: str,
    ) -> list[Triple]:
        if not sections and not source_code.strip():
            return []

        triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        source_kind = self._derive_contract_source_kind(document_title)
        add_triple = self._build_triple_adder(
            triples=triples,
            seen=seen,
            tier=TIER_CONTRACT_GLOBAL,
            source_repo=repo_name,
            source_path=clean_graph_text(document_title),
            source_kind=source_kind,
        )

        if source_kind == SOURCE_KIND_CSPROJ:
            target_frameworks, package_refs = self._extract_csproj_metadata(source_code)
            for framework in target_frameworks:
                add_triple(
                    subject=repo_name,
                    subject_label="REPO",
                    predicate="TARGETS_FRAMEWORK",
                    obj=framework,
                    object_label="FRAMEWORK",
                    confidence=CONTRACT_CSPROJ_METADATA_CONFIDENCE,
                )
            for package_name in package_refs:
                add_triple(
                    subject=repo_name,
                    subject_label="REPO",
                    predicate="CONTAINS_PACKAGE",
                    obj=package_name,
                    object_label="NUGET_PACKAGE",
                    confidence=CONTRACT_CSPROJ_METADATA_CONFIDENCE,
                )
            return triples

        for topic in self._extract_topic_values(
            sections,
            SECTION_TOPICS_CONSUMED,
            source_kind=source_kind,
            source_code=source_code,
            repo_name=repo_name,
        ):
            add_triple(
                subject=repo_name,
                subject_label="REPO",
                predicate="CONSUMES_TOPIC",
                obj=topic,
                object_label="KAFKA_TOPIC",
                confidence=CONTRACT_TOPIC_CONFIDENCE,
            )

        for topic in self._extract_topic_values(
            sections,
            SECTION_TOPICS_PRODUCED,
            source_kind=source_kind,
            source_code=source_code,
            repo_name=repo_name,
        ):
            add_triple(
                subject=repo_name,
                subject_label="REPO",
                predicate="PRODUCES_TOPIC",
                obj=topic,
                object_label="KAFKA_TOPIC",
                confidence=CONTRACT_TOPIC_CONFIDENCE,
            )

        for api_name in self._extract_api_values(
            sections,
            source_kind=source_kind,
            source_code=source_code,
            repo_name=repo_name,
        ):
            add_triple(
                subject=repo_name,
                subject_label="REPO",
                predicate="CALLS_API",
                obj=api_name,
                object_label="API",
                confidence=CONTRACT_API_CONFIDENCE,
            )

        for flag_name in self._extract_feature_flag_values(
            sections,
            source_kind=source_kind,
            source_code=source_code,
            repo_name=repo_name,
        ):
            add_triple(
                subject=repo_name,
                subject_label="REPO",
                predicate="USES_FEATURE_FLAG",
                obj=flag_name,
                object_label="FEATURE_FLAG",
                confidence=CONTRACT_FLAG_CONFIDENCE,
            )

        if self._is_schema_state_document(document_title):
            for table_name in self._extract_schema_state_tables(
                summary_text=summary_text,
                source_code=source_code,
            ):
                add_triple(
                    subject=repo_name,
                    subject_label="REPO",
                    predicate="OWNS_TABLE",
                    obj=table_name,
                    object_label="TABLE",
                    confidence=CONTRACT_TABLE_CONFIDENCE,
                )

        if source_kind == SOURCE_KIND_INGRESS:
            endpoint_values = self._extract_public_api_targets(source_code)
            if not endpoint_values:
                endpoint_values = self._extract_public_api_targets(sections.get(SECTION_PUBLIC_API, ""))
            for endpoint in endpoint_values:
                add_triple(
                    subject=repo_name,
                    subject_label="REPO",
                    predicate="EXPOSES_API",
                    obj=endpoint,
                    object_label="API",
                    confidence=CONTRACT_EXPOSES_API_CONFIDENCE,
                )

        return triples

    @staticmethod
    def _is_schema_state_document(document_title: str) -> bool:
        return "schema_state_" in document_title.lower()

    def _extract_schema_state_tables(self, *, summary_text: str, source_code: str) -> list[str]:
        values: list[str] = []
        text = source_code if source_code.strip() else summary_text
        if not text.strip():
            return []
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

        values.extend(re.findall(r"###\s+`([^`]+)`", normalized_text))
        values.extend(
            re.findall(
                r"\*\*Table:\s*`([^`]+)`\*\*",
                normalized_text,
                flags=re.IGNORECASE,
            )
        )

        for create_match in re.finditer(
            r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"`\[\]]+)",
            normalized_text,
            flags=re.IGNORECASE,
        ):
            values.append(create_match.group(1))

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize_schema_table_name(value)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _normalize_schema_table_name(raw_name: str) -> str:
        token = clean_graph_text(raw_name).strip().strip("`").strip()
        if not token:
            return ""
        token = token.replace('"', "").replace("[", "").replace("]", "")
        token = re.sub(r"\s+", "", token)
        token = token.rstrip(".,;:)")
        if token.lower().startswith("table:"):
            token = token.split(":", 1)[1].strip()
        if not token:
            return ""
        if "." in token:
            token = token.rsplit(".", 1)[-1].strip()
        if not token:
            return ""
        if token.upper() in {"TABLE", "SCHEMA", "OVERVIEW"}:
            return ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{1,120}", token):
            return ""
        return token

    def _build_triple_adder(
        self,
        *,
        triples: list[Triple],
        seen: set[tuple[str, str, str, str, str]],
        tier: str,
        source_repo: str,
        source_path: str,
        source_kind: str,
    ) -> Callable[..., None]:
        def add_triple(
            *,
            subject: str,
            subject_label: str,
            predicate: str,
            obj: str,
            object_label: str,
            confidence: float,
        ) -> None:
            clean_subject = clean_graph_text(subject)
            raw_object = clean_graph_text(obj)
            clean_object = raw_object
            if object_label == "API":
                clean_object = self._canonicalizer.canonicalize_api_name(clean_object)
            elif object_label == "KAFKA_TOPIC":
                clean_object = self._canonicalizer.canonicalize_topic_name(clean_object)
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
                        "tier": tier,
                        "source_repo": source_repo,
                        "source_path": source_path,
                        "source_kind": source_kind,
                        "extractor_name": EXTRACTOR_NAME,
                        "confidence": confidence,
                        "raw_object_name": raw_object,
                        "canonical_object_name": clean_object,
                    },
                )
            )

        return add_triple

    def _extract_topic_values(
        self,
        sections: dict[str, str],
        section_name: str,
        *,
        source_kind: str,
        source_code: str,
        repo_name: str,
    ) -> list[str]:
        values: list[str] = []
        if self._should_use_source_for_topics(repo_name=repo_name, source_kind=source_kind):
            if source_kind == SOURCE_KIND_ASYNCAPI:
                values.extend(self._extract_asyncapi_channels(source_code))
            elif source_kind in {
                SOURCE_KIND_CONFIGMAP,
                SOURCE_KIND_APPSETTINGS_PROD,
                SOURCE_KIND_APPSETTINGS_BASE,
            }:
                produced, consumed = self._extract_topics_from_source_config(
                    source_code=source_code,
                    source_kind=source_kind,
                )
                if section_name == SECTION_TOPICS_PRODUCED:
                    values.extend(produced)
                elif section_name == SECTION_TOPICS_CONSUMED:
                    values.extend(consumed)

        if not values:
            body = sections.get(section_name, "")
            if not body:
                return []
            values.extend(token.strip() for token in re.findall(r"`([^`]+)`", body) if token.strip())

            for line in body.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("|"):
                    if re.match(r"^\|\s*-+\s*\|", stripped):
                        continue
                    cells = [c.strip() for c in stripped.strip("|").split("|")]
                    if cells and cells[0]:
                        values.append(cells[0])
                    continue
                if stripped.startswith("- ") and "`" not in stripped:
                    candidate = stripped[2:].strip()
                    candidate = re.split(r"\s+[—-]\s+", candidate, maxsplit=1)[0].strip()
                    if re.fullmatch(r"[A-Za-z0-9._/-]+", candidate):
                        values.append(candidate)

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._canonicalizer.canonicalize_topic_name(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        entity_kind = (
            "topics_produced"
            if section_name == SECTION_TOPICS_PRODUCED
            else "topics_consumed"
        )
        return self._apply_layered_entity_selection(
            repo_name=repo_name,
            entity_kind=entity_kind,
            source_kind=source_kind,
            values=cleaned,
        )

    def _extract_api_values(
        self,
        sections: dict[str, str],
        *,
        source_kind: str,
        source_code: str,
        repo_name: str,
    ) -> list[str]:
        if source_kind in {SOURCE_KIND_ASYNCAPI, SOURCE_KIND_INGRESS}:
            return []
        raw_external: list[str] = []
        raw_urls: list[str] = []
        config_key_apis: set[str] = set()
        if self._should_use_source_for_apis(repo_name=repo_name, source_kind=source_kind):
            if source_kind in {
                SOURCE_KIND_CONFIGMAP,
                SOURCE_KIND_APPSETTINGS_PROD,
                SOURCE_KIND_APPSETTINGS_BASE,
            }:
                url_entries = self._extract_api_urls_from_config_source(source_code, source_kind=source_kind)
                raw_urls.extend(url for _, url in url_entries)
                for key_name, _ in url_entries:
                    key_api = self._canonicalizer.canonicalize_api_name(key_name)
                    if key_api:
                        config_key_apis.add(key_api)

        if not raw_urls:
            raw_external = self._extract_section_values(sections, SECTION_EXTERNAL_APIS)
            raw_urls = self._extract_urls(sections.get(SECTION_CONFIGURATION, ""))

        trusted_url_apis: set[str] = set()
        for value in raw_urls:
            normalized = self._canonicalizer.canonicalize_api_name(value)
            if normalized:
                trusted_url_apis.add(normalized)

        values: list[str] = list(raw_urls)
        values.extend(raw_external)
        normalized_candidates = [
            self._canonicalizer.canonicalize_api_name(value)
            for value in values
        ]
        normalized_set = {value for value in normalized_candidates if value}
        cleaned: list[str] = []
        seen: set[str] = set()
        for normalized in normalized_candidates:
            if normalized.startswith("i") and len(normalized) > 5:
                alias = normalized[1:]
                if alias in trusted_url_apis or alias in config_key_apis or alias in normalized_set:
                    normalized = alias
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return self._apply_layered_entity_selection(
            repo_name=repo_name,
            entity_kind="apis",
            source_kind=source_kind,
            values=cleaned,
        )

    def _extract_feature_flag_values(
        self,
        sections: dict[str, str],
        *,
        source_kind: str,
        source_code: str,
        repo_name: str,
    ) -> list[str]:
        values: list[str] = []
        if source_kind in {SOURCE_KIND_AST, SOURCE_KIND_CONTRACT}:
            values.extend(extract_feature_flags_from_csharp(source_code))
        elif self._should_use_source_for_flags(repo_name=repo_name, source_kind=source_kind):
            if source_kind in {
                SOURCE_KIND_CONFIGMAP,
                SOURCE_KIND_APPSETTINGS_PROD,
                SOURCE_KIND_APPSETTINGS_BASE,
            }:
                values.extend(
                    self._extract_feature_flags_from_config_source(
                        source_code=source_code,
                        source_kind=source_kind,
                    )
                )

        if not values:
            values = self._extract_section_values(sections, SECTION_FEATURE_FLAGS)
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._canonicalizer.canonicalize_feature_flag_name(value)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return self._apply_layered_entity_selection(
            repo_name=repo_name,
            entity_kind="flags",
            source_kind=source_kind,
            values=cleaned,
        )

    @staticmethod
    def _extract_asyncapi_channels(source_code: str) -> list[str]:
        if not source_code.strip():
            return []
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        in_channels = False
        channels_indent = -1
        seen: set[str] = set()
        channels: list[str] = []
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if not in_channels:
                if stripped == "channels:":
                    in_channels = True
                    channels_indent = indent
                continue
            if indent <= channels_indent:
                break
            match = re.match(r"^(\s*)([^:#][^:]*):\s*$", line)
            if not match:
                continue
            key_indent = len(match.group(1))
            if key_indent != channels_indent + 2:
                continue
            channel_name = match.group(2).strip().strip("'\"")
            if not channel_name or channel_name in seen:
                continue
            seen.add(channel_name)
            channels.append(channel_name)
        return channels

    @staticmethod
    def _parse_sections(summary: str) -> dict[str, str]:
        normalized_summary = summary.replace("\r\n", "\n").replace("\r", "\n")
        normalized_summary = normalized_summary.replace(" --- ", "\n")
        normalized_summary = re.sub(r"(?<!\n)\s+(#{2,3}\s+)", r"\n\1", normalized_summary)

        sections: dict[str, list[str]] = {}
        active_section: str | None = None

        for raw_line in normalized_summary.splitlines():
            line = raw_line.rstrip()
            heading_match = re.match(r"^\s*##\s+(.+?)\s*$", line)
            if heading_match:
                active_section, inline_body = ContractGlobalExtractor._split_heading_and_body(heading_match.group(1))
                sections.setdefault(active_section, [])
                if inline_body:
                    sections[active_section].append(inline_body)
                continue

            subheading_match = re.match(r"^\s*###\s+(.+?)\s*$", line)
            if subheading_match:
                active_section, inline_body = ContractGlobalExtractor._split_heading_and_body(subheading_match.group(1))
                sections.setdefault(active_section, [])
                if inline_body:
                    sections[active_section].append(inline_body)
                continue

            if active_section:
                sections[active_section].append(line)

        return {name: "\n".join(lines).strip() for name, lines in sections.items()}

    @staticmethod
    def _split_heading_and_body(raw_heading: str) -> tuple[str, str]:
        normalized = raw_heading.strip().lower()
        inline_body = ""
        if " - " in normalized:
            section_name, inline_body = normalized.split(" - ", 1)
        else:
            section_name = normalized
        if " | " in section_name:
            section_name, inline_suffix = section_name.split(" | ", 1)
            inline_body = f"{inline_suffix} {inline_body}".strip()
        return section_name.strip(), inline_body.strip()

    def _extract_section_values(self, sections: dict[str, str], section_name: str) -> list[str]:
        body = sections.get(section_name, "")
        if not body:
            return []

        values: list[str] = []
        keyed_topics: list[str] = []
        if section_name in {SECTION_TOPICS_CONSUMED, SECTION_TOPICS_PRODUCED}:
            keyed_topics = re.findall(r"(?:^|-\s*)`([^`]+)`(?=\s*[—-]\s*key\b)", body, flags=re.IGNORECASE)
            if keyed_topics:
                values.extend(topic.strip() for topic in keyed_topics if topic.strip())

        if not keyed_topics:
            inline_tokens = re.findall(r"`([^`]+)`", body)
            if inline_tokens:
                values.extend(token.strip() for token in inline_tokens if token.strip())

        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("|"):
                if re.match(r"^\|\s*-+\s*\|", stripped):
                    continue
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not cells:
                    continue
                candidate = clean_graph_text(cells[0])
                if candidate.lower() in {"channel", "topic", "integration", "flag key"}:
                    continue
                if candidate:
                    values.append(candidate)
                continue

            if stripped.startswith("- "):
                candidate = stripped[2:].strip()
                inline_tokens = re.findall(r"`([^`]+)`", candidate)
                if inline_tokens:
                    values.extend(token.strip() for token in inline_tokens if token.strip())
                else:
                    values.append(candidate)

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = clean_graph_text(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        if not text:
            return []
        urls = re.findall(r"https?://[^\s`)]+", text)
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_url in urls:
            value = raw_url.rstrip(".,;:")
            if value in seen:
                continue
            seen.add(value)
            cleaned.append(value)
        return cleaned

    @classmethod
    def _extract_public_api_targets(cls, text: str) -> list[str]:
        if not text:
            return []

        values: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = clean_graph_text(value)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            values.append(normalized)

        for url in cls._extract_urls(text):
            add(url)

        host_path_patterns = (
            r"host\s+`([^`]+)`\s+(?:under|on)\s+path(?:\s+prefix)?\s+`([^`]+)`",
            r"path(?:\s+prefix)?\s+`([^`]+)`\s+(?:under|on)\s+host\s+`([^`]+)`",
        )
        for pattern in host_path_patterns:
            for first, second in re.findall(pattern, text, flags=re.IGNORECASE):
                if first.startswith("/"):
                    path, host = first, second
                else:
                    host, path = first, second
                host = host.strip().rstrip("/")
                path = path.strip()
                if not host or "." not in host or not path.startswith("/"):
                    continue
                add(f"https://{host}{path}")

        hosts = [token.strip().rstrip("/") for token in re.findall(r"host\s+`([^`]+)`", text, flags=re.IGNORECASE)]
        paths = [token.strip() for token in re.findall(r"path(?:\s+prefix)?\s+`([^`]+)`", text, flags=re.IGNORECASE)]
        valid_hosts = [host for host in hosts if "." in host]
        valid_paths = [path for path in paths if path.startswith("/")]
        for host in valid_hosts:
            add(f"https://{host}")
            for path in valid_paths:
                add(f"https://{host}{path}")

        yaml_hosts = [
            token.strip().rstrip("/")
            for token in re.findall(
                r"(?mi)^\s*-?\s*host:\s*['\"]?([A-Za-z0-9._-]+)",
                text,
            )
        ]
        yaml_paths = [
            token.strip()
            for token in re.findall(
                r"(?mi)^\s*-?\s*path:\s*['\"]?(/[^'\"#\s]*)",
                text,
            )
        ]
        valid_yaml_hosts = [host for host in yaml_hosts if "." in host]
        valid_yaml_paths = [path for path in yaml_paths if path.startswith("/")]
        for host in valid_yaml_hosts:
            add(f"https://{host}")
            for path in valid_yaml_paths:
                add(f"https://{host}{path}")

        return values

    def _should_use_source_for_topics(self, *, repo_name: str, source_kind: str) -> bool:
        return source_kind in {
            SOURCE_KIND_ASYNCAPI,
            SOURCE_KIND_CONFIGMAP,
            SOURCE_KIND_APPSETTINGS_PROD,
            SOURCE_KIND_APPSETTINGS_BASE,
        }

    def _should_use_source_for_apis(self, *, repo_name: str, source_kind: str) -> bool:
        return source_kind in {
            SOURCE_KIND_CONFIGMAP,
            SOURCE_KIND_APPSETTINGS_PROD,
            SOURCE_KIND_APPSETTINGS_BASE,
            SOURCE_KIND_INGRESS,
        }

    def _should_use_source_for_flags(self, *, repo_name: str, source_kind: str) -> bool:
        return source_kind in {
            SOURCE_KIND_CONFIGMAP,
            SOURCE_KIND_APPSETTINGS_PROD,
            SOURCE_KIND_APPSETTINGS_BASE,
        }

    def _apply_layered_entity_selection(
        self,
        *,
        repo_name: str,
        entity_kind: str,
        source_kind: str,
        values: list[str],
    ) -> list[str]:
        if not values:
            return []
        repo_state = self._repo_entity_priorities.setdefault(repo_name, {})
        entity_state = repo_state.setdefault(entity_kind, {})
        priority = self._source_priority(source_kind)
        selected: list[str] = []

        for value in values:
            normalized = value.lower()
            previous_priority = entity_state.get(normalized)
            if previous_priority is not None and previous_priority > priority:
                continue
            entity_state[normalized] = priority
            selected.append(value)

        return selected

    @staticmethod
    def _source_priority(source_kind: str) -> int:
        if source_kind == SOURCE_KIND_ASYNCAPI:
            return SOURCE_PRIORITY_ASYNCAPI
        if source_kind == SOURCE_KIND_CONFIGMAP:
            return SOURCE_PRIORITY_CONFIGMAP
        if source_kind == SOURCE_KIND_APPSETTINGS_PROD:
            return SOURCE_PRIORITY_APPSETTINGS_PROD
        if source_kind == SOURCE_KIND_APPSETTINGS_BASE:
            return SOURCE_PRIORITY_APPSETTINGS_BASE
        if source_kind == SOURCE_KIND_AST:
            return SOURCE_PRIORITY_AST
        if source_kind == SOURCE_KIND_INGRESS:
            return SOURCE_PRIORITY_INGRESS
        return SOURCE_PRIORITY_DEFAULT

    def _extract_topics_from_source_config(
        self,
        *,
        source_code: str,
        source_kind: str,
    ) -> tuple[list[str], list[str]]:
        cfg = self._parse_config_source(source_code=source_code, source_kind=source_kind)
        produced: set[str] = set()
        consumed: set[str] = set()

        def as_strings(value: object) -> list[str]:
            if isinstance(value, str):
                return [value] if value else []
            if isinstance(value, list):
                return [token for token in value if isinstance(token, str) and token]
            if isinstance(value, dict):
                return [token for token in value.values() if isinstance(token, str) and token]
            return []

        def walk(node: object, context: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_lower = str(key).lower()
                    next_context = (
                        "prod" if "producer" in key_lower else
                        "cons" if "consumer" in key_lower else
                        context
                    )
                    if key_lower in TOPIC_KEYS:
                        target = consumed if next_context == "cons" else produced
                        target.update(as_strings(value))
                    if isinstance(value, (dict, list)):
                        walk(value, next_context)
            elif isinstance(node, list):
                for item in node:
                    walk(item, context)

        walk(cfg, "")
        return sorted(produced), sorted(consumed)

    def _extract_api_urls_from_config_source(
        self,
        source_code: str,
        *,
        source_kind: str,
    ) -> list[tuple[str, str]]:
        cfg = self._parse_config_source(source_code=source_code, source_kind=source_kind)
        found: dict[str, str] = {}

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_token = str(key)
                    key_lower = key_token.lower()
                    if (
                        isinstance(value, str)
                        and value.startswith(("http://", "https://"))
                        and "api" in key_lower
                        and not any(fragment in key_lower for fragment in URL_SKIP_KEY_FRAGMENTS)
                    ):
                        host = (urlparse(value).hostname or "").lower()
                        if not any(fragment in host for fragment in URL_SKIP_HOST_FRAGMENTS):
                            found.setdefault(key_token, value)
                    else:
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(cfg)
        return list(found.items())

    def _extract_feature_flags_from_config_source(self, *, source_code: str, source_kind: str) -> list[str]:
        cfg = self._parse_config_source(source_code=source_code, source_kind=source_kind)
        launch_darkly = cfg.get("LaunchDarkly") if isinstance(cfg, dict) else None
        if not isinstance(launch_darkly, dict):
            return []
        flags: set[str] = set()
        for key, value in launch_darkly.items():
            if isinstance(value, dict):
                flags.add(str(key))
                flag_name = value.get("FlagName")
                if isinstance(flag_name, str) and flag_name.strip():
                    flags.add(flag_name.strip())
        return sorted(flags)

    @staticmethod
    def _parse_config_source(*, source_code: str, source_kind: str) -> dict:
        if not source_code.strip():
            return {}
        if source_kind == SOURCE_KIND_CONFIGMAP:
            return ContractGlobalExtractor._parse_configmap(source_code)
        return ContractGlobalExtractor._parse_json_object(source_code)

    @staticmethod
    def _parse_json_object(source_code: str) -> dict:
        try:
            parsed = json.loads(source_code.lstrip("\ufeff"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_configmap(source_code: str) -> dict:
        out: dict = {}
        for line in source_code.splitlines():
            if not line.startswith("  ") or "__" not in line:
                continue
            key, sep, value = line.strip().partition(":")
            if not sep or "__" not in key:
                continue
            parts = key.split("__")
            target = out
            for part in parts[:-1]:
                existing = target.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    target[part] = existing
                target = existing
            target[parts[-1]] = value.strip().strip('"')
        return out

    @staticmethod
    def _extract_csproj_metadata(source_code: str) -> tuple[list[str], list[str]]:
        if not source_code.strip():
            return [], []
        try:
            root = ET.fromstring(source_code)
        except ET.ParseError:
            return [], []

        frameworks: set[str] = set()
        packages: set[str] = set()

        for element in root.iter():
            if not isinstance(element.tag, str):
                continue
            tag_name = element.tag.rsplit("}", 1)[-1]
            if tag_name in {"TargetFramework", "TargetFrameworks"}:
                raw_value = (element.text or "").strip()
                if not raw_value:
                    continue
                for framework in raw_value.split(";"):
                    normalized_framework = clean_graph_text(framework)
                    if normalized_framework:
                        frameworks.add(normalized_framework)
                continue
            if tag_name == "PackageReference":
                include_value = clean_graph_text(
                    element.attrib.get("Include", "") or element.attrib.get("Update", "")
                )
                if include_value:
                    packages.add(include_value)

        return sorted(frameworks), sorted(packages)

    @staticmethod
    def _derive_contract_source_kind(document_title: str) -> str:
        lower_title = document_title.lower()
        if "asyncapi" in lower_title:
            return SOURCE_KIND_ASYNCAPI
        if "configmap" in lower_title:
            return SOURCE_KIND_CONFIGMAP
        if lower_title.endswith("appsettings.prod.json"):
            return SOURCE_KIND_APPSETTINGS_PROD
        if lower_title.endswith("appsettings.json"):
            return SOURCE_KIND_APPSETTINGS_BASE
        if "ingress" in lower_title:
            return SOURCE_KIND_INGRESS
        if lower_title.endswith(".csproj"):
            return SOURCE_KIND_CSPROJ
        return SOURCE_KIND_CONTRACT
