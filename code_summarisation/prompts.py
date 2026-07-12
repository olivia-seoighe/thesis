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

The codebase is an enterprise platform built from microservices that communicate \
over Kafka, message buses, and external service integrations.

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

File-type hints:
- Controllers / handlers / API clients: usually include Public API Surface + \
Service Dependencies + Configuration.
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


def format_summary_prompt(*, repo: str, file_path: str, language: str, content: str) -> str:
    return SUMMARY_PROMPT.format(
        repo=repo,
        file_path=file_path,
        language=language,
        content=content,
        strict_rules=_STRICT_RULES,
    )


def format_migration_prompt(*, repo: str, content: str) -> str:
    return MIGRATION_AGGREGATE_PROMPT.format(
        repo=repo,
        content=content,
        strict_rules=_STRICT_RULES,
    )
