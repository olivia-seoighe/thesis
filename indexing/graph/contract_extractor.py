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


def _dedupe_normalized(values: list[str], normalize_fn: Callable[[str], str]) -> list[str]:
    """Normalize each value, drop empties, and dedupe on the normalized form."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_fn(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned

class ContractGlobalExtractor:
    """Extracts graph triples from source code and structured summary content."""

    _canonicalizer = GraphCanonicalizer()
    _KNOWN_SECTION_NAMES = (
        SECTION_CONFIGURATION,
        SECTION_FEATURE_FLAGS,
        SECTION_TOPICS_CONSUMED,
        SECTION_TOPICS_PRODUCED,
        SECTION_EXTERNAL_APIS,
        SECTION_PUBLIC_API,
    )

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
    ) -> list[str]:
        values: list[str] = []
        used_structured_source = False
        topic_source_kinds = {SOURCE_KIND_ASYNCAPI, SOURCE_KIND_CONFIGMAP, SOURCE_KIND_APPSETTINGS_PROD, SOURCE_KIND_APPSETTINGS_BASE}
        if source_kind in topic_source_kinds and source_code.strip():
            if source_kind == SOURCE_KIND_ASYNCAPI:
                used_structured_source = True
                asyncapi_produced, asyncapi_consumed = self._parse_asyncapi_operations(source_code)
                if section_name == SECTION_TOPICS_PRODUCED:
                    values.extend(asyncapi_produced)
                elif section_name == SECTION_TOPICS_CONSUMED:
                    values.extend(asyncapi_consumed)
            elif source_kind in {
                SOURCE_KIND_CONFIGMAP,
                SOURCE_KIND_APPSETTINGS_PROD,
                SOURCE_KIND_APPSETTINGS_BASE,
            }:
                used_structured_source = True
                produced, consumed = self._extract_topics_from_source_config(
                    source_code=source_code,
                    source_kind=source_kind,
                )
                if section_name == SECTION_TOPICS_PRODUCED:
                    values.extend(produced)
                elif section_name == SECTION_TOPICS_CONSUMED:
                    values.extend(consumed)

        if not values and not used_structured_source:
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

        return _dedupe_normalized(values, self._canonicalizer.canonicalize_topic_name)

    def _extract_api_values(
        self,
        sections: dict[str, str],
        *,
        source_kind: str,
        source_code: str,
    ) -> list[str]:
        if source_kind in {SOURCE_KIND_ASYNCAPI, SOURCE_KIND_INGRESS}:
            return []
        raw_external: list[str] = []
        raw_urls: list[str] = []
        if source_kind in {SOURCE_KIND_CONFIGMAP, SOURCE_KIND_APPSETTINGS_PROD, SOURCE_KIND_APPSETTINGS_BASE}:
            for key_name, url in self._extract_api_urls_from_config_source(source_code, source_kind=source_kind):
                # A config entry's key and its URL value often name the same call two
                # different ways (e.g. "InternalLabResultApiUrl" -> "https://rms-ilr-api...").
                # Emit both: the key matches what the code side's own citations already
                # canonicalize to, while the URL is a cross-repo-consistent anchor (derived
                # from the actual target host, not each repo's own naming choice) -- needed
                # when different repos name the same real target differently. Identical
                # canonicalizations collapse naturally via the seen-dedup below.
                raw_urls.append(key_name)
                raw_urls.append(url)

        if not raw_urls:
            raw_external = self._extract_section_values(sections, SECTION_EXTERNAL_APIS)
            raw_urls = self._extract_urls(sections.get(SECTION_CONFIGURATION, ""))

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
                if alias in normalized_set:
                    normalized = alias
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    def _extract_feature_flag_values(
        self,
        sections: dict[str, str],
        *,
        source_kind: str,
        source_code: str,
    ) -> list[str]:
        values: list[str] = []
        if source_kind in {SOURCE_KIND_AST, SOURCE_KIND_CONTRACT}:
            values.extend(extract_feature_flags_from_csharp(source_code))
        elif source_kind in {SOURCE_KIND_CONFIGMAP, SOURCE_KIND_APPSETTINGS_PROD, SOURCE_KIND_APPSETTINGS_BASE}:
            values.extend(self._extract_feature_flags_from_config_source(source_code=source_code, source_kind=source_kind))

        if not values:
            values = self._extract_section_values(sections, SECTION_FEATURE_FLAGS)
        return _dedupe_normalized(values, self._canonicalizer.canonicalize_feature_flag_name)

    @staticmethod
    def _parse_asyncapi_operations(source_code: str) -> tuple[list[str], list[str]]:
        """Parse an AsyncAPI 3.x document into (produced, consumed) channel names.

        AsyncAPI 3.x moves publish/subscribe direction out of `channels:` (which
        only declares channel names/addresses) into a separate `operations:`
        block, where each operation has an `action: send|receive` and a
        `channel: $ref: '#/channels/<key>'`. Direction cannot be determined
        from `channels:` alone -- older code here (and older AsyncAPI 2.x
        documents, which nest publish/subscribe directly under each channel)
        is not handled and yields no results.
        """
        if not source_code.strip():
            return [], []
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").splitlines()

        channel_names: dict[str, str] = {}
        in_channels = False
        channels_indent = -1
        current_key: str | None = None
        current_key_indent = -1
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
            key_match = re.match(r"^(\s*)([^:#][^:]*):\s*$", line)
            if key_match and len(key_match.group(1)) == channels_indent + 2:
                current_key = key_match.group(2).strip().strip("'\"")
                current_key_indent = len(key_match.group(1))
                if current_key:
                    channel_names.setdefault(current_key, current_key)
                continue
            addr_match = re.match(r"^(\s*)address:\s*(.+?)\s*$", line)
            if addr_match and current_key and len(addr_match.group(1)) > current_key_indent:
                address = addr_match.group(2).strip().strip("'\"")
                if address:
                    channel_names[current_key] = address

        produced: list[str] = []
        consumed: list[str] = []
        in_operations = False
        operations_indent = -1
        pending_action: str | None = None
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if not in_operations:
                if stripped == "operations:":
                    in_operations = True
                    operations_indent = indent
                continue
            if indent <= operations_indent:
                break
            op_match = re.match(r"^(\s*)([^:#][^:]*):\s*$", line)
            if op_match and len(op_match.group(1)) == operations_indent + 2:
                pending_action = None
                continue
            action_match = re.match(r"^\s*action:\s*(send|receive)\s*$", line)
            if action_match:
                pending_action = action_match.group(1)
                continue
            ref_match = re.match(r"^\s*\$ref:\s*['\"]?#/channels/([^'\"\s]+)['\"]?\s*$", line)
            if ref_match and pending_action:
                channel_key = ref_match.group(1)
                resolved = channel_names.get(channel_key, channel_key)
                target = produced if pending_action == "send" else consumed
                if resolved not in target:
                    target.append(resolved)

        return produced, consumed

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

        # A heading's descriptive sentence isn't always set off by " - " or " | "
        # (e.g. "## External API Calls All clients registered via Refit...") --
        # check known section names as a prefix first so that case still splits
        # correctly instead of the whole line becoming an unrecognized section key.
        for known in ContractGlobalExtractor._KNOWN_SECTION_NAMES:
            match = re.match(rf"^{re.escape(known)}\b[\s:-]*(.*)$", normalized)
            if match:
                return known, match.group(1).strip(" -:")

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

        return _dedupe_normalized(values, clean_graph_text)

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        if not text:
            return []
        urls = re.findall(r"https?://[^\s`)]+", text)
        return _dedupe_normalized(urls, lambda raw_url: raw_url.rstrip(".,;:"))

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

        def add_host_path_products(hosts_pattern: str, paths_pattern: str) -> None:
            """A host and path stated as separate tokens (rather than paired in
            one match) name every combination found, plus each bare host."""
            hosts = [t.strip().rstrip("/") for t in re.findall(hosts_pattern, text, flags=re.IGNORECASE) if "." in t]
            paths = [t.strip() for t in re.findall(paths_pattern, text, flags=re.IGNORECASE) if t.strip().startswith("/")]
            for host in hosts:
                add(f"https://{host}")
                for path in paths:
                    add(f"https://{host}{path}")

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

        add_host_path_products(r"host\s+`([^`]+)`", r"path(?:\s+prefix)?\s+`([^`]+)`")
        add_host_path_products(r"(?m)^\s*-?\s*host:\s*['\"]?([A-Za-z0-9._-]+)", r"(?m)^\s*-?\s*path:\s*['\"]?(/[^'\"#\s]*)")

        return values

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
