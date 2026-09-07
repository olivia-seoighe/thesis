"""Shared lexical rules for graph extraction and canonicalization."""

from __future__ import annotations

API_CONFIG_PREFIXES: tuple[str, ...] = ("apiurls", "apiconfig", "webapiconfig")

API_URL_SUFFIXES: tuple[str, ...] = (
    "baseurl",
    "url",
    "uri",
    "endpoint",
    "host",
    "domain",
    "address",
    "name",
)

API_NOISE_TERMS: frozenset[str] = frozenset(
    {
        "api",
        "apiapi",
        "apiurls",
        "apiconfig",
        "webapiconfig",
        "defaultapi",
        "localhostapi",
    }
)

API_NOISE_FRAGMENTS: frozenset[str] = frozenset(
    {
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
)

API_NOISE_SUFFIXES: tuple[str, ...] = ("client", "handler", "config", "serializer", "policy")

CONFIG_TOPIC_PATH_FRAGMENTS: tuple[str, ...] = (
    "kafkatopics__",
    "akkaconfig.topics",
    "kafkaconsumerconfig.topics",
)

RAW_TOPIC_REFERENCE_FRAGMENTS: tuple[str, ...] = (
    "constants.outboundtopics.",
    "constants.kafka.",
    "akkaconfig.topics[",
    "kafkaconsumerconfig.topics",
)

TOPIC_NOISE_TERMS: frozenset[str] = frozenset(
    {
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
    }
)

FEATURE_FLAG_NOISE_TERMS: frozenset[str] = frozenset(
    {
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
)

FEATURE_FLAG_NOISE_FRAGMENTS: tuple[str, ...] = (
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

HOST_PREFIX_NOISE_TERMS: frozenset[str] = frozenset(
    {"api", "dev", "prod", "staging", "localhost", "default", "hooks"}
)

URL_PATH_NOISE_TERMS: frozenset[str] = frozenset({"api", "health", "live", "oauth2", "swagger"})

CONFIG_TOPIC_KEYS: frozenset[str] = frozenset(
    {"topics", "kafkatopic", "kafkatopics", "consumers", "producers"}
)

CONFIG_API_KEY_SKIP_FRAGMENTS: tuple[str, ...] = ("okta", "health", "relic", "pdp", "swagger")
CONFIG_API_HOST_SKIP_FRAGMENTS: tuple[str, ...] = ("okta", "login", "auth", "identity")
