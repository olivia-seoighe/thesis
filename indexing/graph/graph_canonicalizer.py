"""Canonicalization and validation helpers for graph node values."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from indexing.graph.text_cleaning import clean_graph_text


class GraphCanonicalizer:
    """Normalize and validate extracted API/topic/feature-flag node names."""

    def canonicalize_api_name(self, value: str) -> str:
        token = clean_graph_text(value).strip().strip('"\'')
        if not token:
            return ""
        lowered = token.lower()
        if lowered.startswith("#{"):
            closing = lowered.find("}")
            if closing > 2:
                lowered = lowered[2:closing].strip()
        lowered = lowered.strip().strip('"\'')
        lowered = lowered.replace("`", "")
        lowered = re.sub(r"^(?:apiurls|apiconfig|webapiconfig)[._-]*", "", lowered)

        if lowered.startswith(("http://", "https://")):
            from_url = self._api_name_from_url(lowered)
            if from_url:
                return from_url
            candidate = lowered.rstrip("/")
            return candidate if self._is_valid_api_name(candidate, raw=token) else ""

        lowered = lowered.replace("__", ".").replace("::", ".")
        lowered = re.sub(r"[^a-z0-9._-]+", "", lowered)

        compact_api_match = re.search(r"\b([a-z0-9]+api(?:v[0-9]+)?)\b", lowered)
        if compact_api_match:
            candidate = compact_api_match.group(1)
            candidate = self._strip_interface_prefix(candidate, raw=token)
            return candidate if self._is_valid_api_name(candidate, raw=token) else ""

        if "okta" in lowered:
            return "oktaapi" if self._is_valid_api_name("oktaapi", raw=token) else ""

        for suffix in ("baseurl", "url", "uri", "endpoint", "host", "domain", "address", "name"):
            if lowered.endswith(suffix):
                stem = lowered[: -len(suffix)].rstrip("._-")
                stem_match = re.search(r"([a-z0-9]+api(?:v[0-9]+)?)$", stem)
                if stem_match:
                    candidate = stem_match.group(1)
                    candidate = self._strip_interface_prefix(candidate, raw=token)
                    return candidate if self._is_valid_api_name(candidate, raw=token) else ""

        terminal = lowered.split(".")[-1]
        terminal = terminal.strip("._-")
        terminal = self._strip_interface_prefix(terminal, raw=token)
        if terminal.startswith("nternal"):
            terminal = f"i{terminal}"
        if "api" in terminal:
            return terminal if self._is_valid_api_name(terminal, raw=token) else ""
        return ""

    def canonicalize_topic_name(self, value: str) -> str:
        token = clean_graph_text(value).strip().strip('"\'').replace("`", "")
        if not token:
            return ""
        lowered = token.lower()
        if lowered in {"application/json", "text/plain", "application/xml"}:
            return ""
        if lowered.startswith("#{"):
            closing = lowered.find("}")
            if closing > 2:
                lowered = lowered[2:closing].strip()
        lowered = lowered.strip()
        lowered = re.sub(r"\s+", "", lowered)
        if "/" in lowered:
            lowered = lowered.split("/", 1)[0]
        lowered = lowered.strip("._-")
        if not self._is_valid_topic_name(lowered, raw=token):
            return ""
        return lowered

    def canonicalize_feature_flag_name(self, value: str) -> str:
        token = clean_graph_text(value).strip().strip('"\'').replace("`", "")
        if not token:
            return ""
        lowered = token.lower()

        if lowered.startswith("#{"):
            closing = lowered.find("}")
            if closing > 2:
                token = token[2:closing]
                lowered = token.lower()

        token = token.split(":", 1)[0].strip()
        token = token.rstrip("()")
        lowered = token.lower()

        if lowered.startswith("ifeatureflags."):
            token = token.split(".", 1)[1]
            lowered = token.lower()
        if lowered.startswith("ifeatureflags"):
            return ""

        if "__" in token:
            return ""
        if any(ch.isspace() for ch in token):
            return ""
        if not re.fullmatch(r"[A-Za-z0-9._-]+", token):
            return ""

        blocked_exact = {
            "true",
            "false",
            "development",
            "project",
            "featureflags",
            "launchdarkly",
            "nuvoair",
            "testvendor",
            "heartbeathealth",
        }
        if lowered in blocked_exact:
            return ""

        blocked_fragments = (
            "config",
            "launchdarkly",
            "licensekey",
            "securityprotocol",
            "supportedvendors",
            "sectionname",
            "descriptor",
            "continueondeserializationerrors",
            "continueonfailure",
            "usepostgrespersistence",
            "authsettings",
            "new_relic",
            "aspnetcore_environment",
            "sasl_ssl",
            "automapperlicensekey",
            "ildclient",
            "ilaunchdarklyservice",
            "sdk-",
            "permit__",
            "productconfiguration__",
            "kafkaconfig.",
        )
        if any(fragment in lowered for fragment in blocked_fragments):
            return ""

        if lowered.startswith("enable") and len(token) > 6:
            if token.startswith("Enable"):
                return token
            return f"Enable{token[6:]}"

        if "-" in token:
            return lowered
        if "_" in token:
            return lowered
        return ""

    def _is_valid_api_name(self, candidate: str, *, raw: str) -> bool:
        if not candidate:
            return False
        name = candidate.lower().strip()
        raw_lower = raw.lower().strip()
        if len(name) < 4 or len(name) > 40:
            return False
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            return False
        if name.isdigit():
            return False
        if name.count("api") > 2:
            return False
        blocked_terms = {"api", "apiapi", "apiurls", "apiconfig", "webapiconfig", "defaultapi", "localhostapi"}
        if name in blocked_terms:
            return False
        blocked_fragments = {
            "connectionstring",
            "containername",
            "blobcontainer",
            "httpclienthandler",
            "contentserializer",
            "authorizationurl",
            "allowautoredirect",
            "baseaddress",
            "servicecollection",
            "defaulttenant",
            "clientid",
            "username",
            "password",
            "retry",
            "timeout",
            "policy",
            "exception",
        }
        if any(fragment in name for fragment in blocked_fragments):
            return False
        if any(part in raw_lower for part in ("kafkatopics__", "akkaconfig.topics", "kafkaconsumerconfig.topics")):
            return False
        if name.endswith(("client", "handler", "config", "serializer", "policy")):
            return False
        return True

    def _is_valid_topic_name(self, candidate: str, *, raw: str) -> bool:
        if not candidate:
            return False
        if len(candidate) < 3 or len(candidate) > 120:
            return False
        if not re.fullmatch(r"[a-z0-9._-]+", candidate):
            return False
        if candidate.isdigit():
            return False
        if any(part in candidate for part in ("kafkatopics__", "akkaconfig.topics", "kafkaconsumerconfig.topics")):
            return False
        if any(part in raw.lower() for part in ("constants.outboundtopics.", "constants.kafka.", "akkaconfig.topics[", "kafkaconsumerconfig.topics")):
            return False
        if candidate in {
            "status",
            "result",
            "results",
            "performed",
            "notperformed",
            "orderheld",
            "application/json",
            "type",
            "subscribe",
            "commonheaders",
            "context",
            "eventid",
            "guid",
            "post",
            "get",
            "vendor",
            "productcode",
        }:
            return False
        if candidate.startswith(("http://", "https://")):
            return False
        return True

    @staticmethod
    def _strip_interface_prefix(candidate: str, *, raw: str) -> str:
        if not candidate.startswith("i") or len(candidate) <= 4:
            return candidate
        raw_compact = re.sub(r"[^A-Za-z0-9]", "", raw)
        if len(raw_compact) >= 2 and raw_compact[0].lower() == "i" and raw_compact[1].isupper():
            return candidate[1:]
        return candidate

    def _api_name_from_url(self, url_value: str) -> str:
        try:
            parsed = urlparse(url_value)
        except ValueError:
            return ""

        host = (parsed.hostname or "").lower().strip()
        if not host:
            return ""
        if ".okta." in f".{host}." or host.endswith(".okta.com"):
            return "oktaapi"

        path_segments = [seg for seg in parsed.path.split("/") if seg]
        for segment in path_segments:
            seg = re.sub(r"[^a-z0-9_-]+", "", segment.lower())
            if not seg or seg in {"api", "health", "live", "oauth2", "swagger"}:
                continue
            candidate = re.search(r"([a-z0-9]+api(?:v[0-9]+)?)", seg)
            if candidate:
                return candidate.group(1)
            if seg.startswith("v") and seg[1:].isdigit():
                continue
            return self._ensure_api_suffix(seg)

        host_parts = [part for part in host.split(".") if part]
        if not host_parts:
            return ""
        host_head = host_parts[0]
        if host_head in {"api", "dev", "prod", "staging", "localhost", "default", "hooks"} and len(host_parts) > 1:
            host_head = host_parts[1]
        host_head = re.sub(r"[^a-z0-9_-]+", "", host_head)
        if not host_head:
            return ""
        candidate = re.search(r"([a-z0-9]+api(?:v[0-9]+)?)", host_head)
        if candidate:
            return candidate.group(1)
        return self._ensure_api_suffix(host_head)

    @staticmethod
    def _ensure_api_suffix(value: str) -> str:
        """Append 'api' only if the hyphen/underscore-stripped name doesn't already end with it.

        The regex-based match above can't see across a '-'/'_' boundary (e.g. "dps-operations-api"
        has no unbroken run of [a-z0-9]+ ending in "api"), so without this check the naive fallback
        would double-suffix names like that into "dps-operations-apiapi".
        """
        compact = re.sub(r"[_-]+", "", value)
        if compact.endswith("api"):
            return compact
        return f"{value}api"
