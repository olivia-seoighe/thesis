"""Prompts for the code summarisation service."""

# Shared precision rules applied to all prompts.
_STRICT_RULES = """\
- Only include names and values that appear explicitly in the source — do not invent or infer.
- Do not list absent concepts (e.g. avoid "No X, Y, or Z configured here").
- If a section has nothing to say, use one generic negative: \
"No business workflow logic is implemented here." Then stop.
- Do not repeat the same negative across sections.
- Do not expand acronyms that are unexpanded in the source.
- When interpreting an attribute or decorator, base the description on its \
namespace or import — not the attribute name alone.\
"""

SUMMARY_PROMPT = """\
You are writing a retrieval-optimised summary of a source file for a RAG system. \
The summary will be embedded and searched against natural language queries from \
developers and operations staff. Write so that someone searching by service name, \
Kafka topic, business rule, or business process would find this file.

{tenant_context}

Repository: {repo}
File: {file_path}

Precision rules:
{strict_rules}

Instructions:
- Use bullet points throughout.
- Include only sections that add new information for this specific file.
- Cite sources inline: after each concrete claim write the line range in parentheses \
as (Lstart-Lend) or (Lstart). The source below is line-numbered — use those numbers.
- Do not output a reasoning section or any pre-summary analysis.

Required sections (every file):
## Purpose — what this file does and which service it belongs to.
## Key Business Logic — main rules and decisions in this file only. \
If none, write: "No business workflow logic is implemented here."

Optional sections (include only if they add new facts not already stated):
## Public API Surface — routes, public methods, message contracts. \
Contract-only; do not restate dependencies here.
## Data Models — tables, columns, enums, status codes.
## Service Dependencies — systems this file directly calls or references. \
Use sub-sections below; include only those with at least one item.
    ### Topics Consumed
    ### Topics Produced
    ### Message Queues
    ### External API Calls
    ### Feature Flags
    ### Database Tables Read
    ### Database Tables Written
## Configuration — exact keys, values, and defaults defined in this file only.
## Notable Error Handling — retry logic, dead-letter handling, fallback behaviour.

File-type guidance:
- Integration boundary files (Controllers / handlers / API clients): usually include \
Public API Surface + Service Dependencies + Configuration. \
(Key Business Logic is still required; write the fallback if none is present.)
- SQL migrations: focus on Data Models; omit Public API Surface unless relevant.
- Internal domain files: include Public API Surface only if there is a true \
external contract.

```{language}
{content}
```\
"""

MIGRATION_AGGREGATE_PROMPT = """\
You are documenting the current effective database schema of the `{repo}` service \
for a RAG system. You are given all SQL migration files in chronological order. \
Each file is marked with a `-- FILE: <path>` header and line numbers.

{tenant_context}

Apply every migration mentally in order and describe the schema as it stands now. \
Do not narrate migration history — only describe the final state.

Precision rules:
{strict_rules}

Instructions:
- Use bullet points. Write at most 700 words total.
- Cite sources inline: after each table, column, or constraint write the migration \
filename and line range as (filename:Lstart-Lend). Use exact filenames from the \
`-- FILE:` headers.

## Schema Overview
Current tables: columns, types, primary keys, foreign keys, and significant \
nullability constraints. Omit anything created and later dropped.

## Key Business Entities
What domain concepts these tables represent in the service's domain. \
Use exact table and column names.

## Indexes & Constraints
Non-obvious indexes, unique constraints, check constraints, and permission grants \
worth knowing. Omit this section if there are none of note.

Repository: {repo}

Migration files (ordered, line-numbered):

{content}\
"""

CHUNK_SUMMARY_PROMPT = """\
You are synthesising AI-generated file summaries for {group} of the `{repo}` service. \
Produce a detailed summary (100-300 words) that faithfully captures everything \
architecturally significant in these files. Do not narrow your focus — include all of \
the following where evidenced in the summaries: purpose and role in the service, key \
components and their responsibilities, message topics consumed or produced, external \
APIs and service dependencies, data models and entities, configuration and feature \
flags, business logic and decision rules, and error handling patterns. Use exact names \
from the summaries.

{tenant_context}

Precision rules:
{strict_rules}

{summaries}\
"""

REPO_SUMMARY_PROMPT = """\
You are writing a repository-level architectural overview for the `{repo}` service for \
a RAG system. You have been given summaries of every source file in the repository. \
Synthesise these into a single cohesive index that serves as the living source of truth \
for the repo's architecture.

{tenant_context}

Precision rules:
{strict_rules}

Ground your output strictly in what the file summaries contain. Do not invent \
integrations, topics, or workflows not evidenced in the summaries.

The individual file summaries already hold all per-file detail — this document is \
orientation only. Hard limit: 600 words maximum. Every section must be a table or a \
tight bullet list; no prose paragraphs except the single Overview paragraph. Prefer \
omitting a section over padding it.

Produce a structured document with exactly these sections (omit a section only if there \
is genuinely no evidence for it in the summaries):

## Overview
One short paragraph (3-4 sentences max): what this service does, its role in the \
platform, the technology stack, and the architectural pattern. Then a single flat \
bullet list of external integrations (services, APIs, topics) — names only.

## End-to-End Flow
The primary workflow as a compact table:
| Step | Actor | Action | Result |
|------|-------|--------|--------|
Derive strictly from event/command handlers in the summaries.

## Integration Points
| Integration | Direction | Event / Topic / Endpoint | Error handling |
|-------------|-----------|--------------------------|----------------|
Inbound or Outbound. Include all integrations evidenced in the summaries.

## Message Topics
| Topic | Direction | Event Type |
|-------|-----------|------------|
Only topics explicitly named in the summaries.

## Data Model
Bullet list: one line per key entity — name, primary key, and one-phrase purpose. No \
field lists, no types. Omit if a schema_state summary already covers it.

## Business Rules & Decision Logic
Bullet list of the most significant rules only (5 bullets max). Exact names/values.

## Feature Flags
Only list flags managed by a dedicated feature-flag system (e.g. LaunchDarkly, \
`IFeatureFlag`/`IFeatureManager`). One line per flag: the key and the behaviour it \
controls. Do NOT list environment variables, transport or connection switches, \
`appsettings` values, or job-scheduling/trigger toggles — those are configuration, not \
feature flags. Omit this section entirely if the service has no dedicated feature flags.

---

Repository: {repo}

Individual file summaries:

{summaries}\
"""


def format_summary_prompt(
    *,
    repo: str,
    file_path: str,
    language: str,
    content: str,
    tenant_context: str,
) -> str:
    return SUMMARY_PROMPT.format(
        repo=repo,
        file_path=file_path,
        language=language,
        content=content,
        strict_rules=_STRICT_RULES,
        tenant_context=tenant_context,
    )


def format_migration_prompt(
    *,
    repo: str,
    content: str,
    tenant_context: str,
) -> str:
    return MIGRATION_AGGREGATE_PROMPT.format(
        repo=repo,
        content=content,
        strict_rules=_STRICT_RULES,
        tenant_context=tenant_context,
    )


def format_chunk_prompt(
    *,
    repo: str,
    group: str,
    summaries: str,
    tenant_context: str,
) -> str:
    return CHUNK_SUMMARY_PROMPT.format(
        repo=repo,
        group=group,
        summaries=summaries,
        strict_rules=_STRICT_RULES,
        tenant_context=tenant_context,
    )


def format_repo_summary_prompt(
    *,
    repo: str,
    summaries: str,
    tenant_context: str,
) -> str:
    return REPO_SUMMARY_PROMPT.format(
        repo=repo,
        summaries=summaries,
        strict_rules=_STRICT_RULES,
        tenant_context=tenant_context,
    )
