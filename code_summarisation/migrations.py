"""Helpers for selecting and ordering DB migration files.

Migrations are summarised once per repo into a single schema_state (the current
effective schema), never per-file. `migrations.sources` in config is an ordered
list; the first source with any files wins (e.g. Atlas over legacy Flyway), and the
losing source is excluded from per-file summaries entirely.
"""

import re


def natural_sort_key(path: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def is_migration_path(path: str, migrations_cfg: dict | None) -> bool:
    """True if path is under any configured migration source prefix."""
    for source in (migrations_cfg or {}).get("sources", []):
        if path.startswith(source["path_prefix"].rstrip("/") + "/"):
            return True
    return False


def select_migration_files(files: list, migrations_cfg: dict | None) -> list:
    """Return the winning migration source's .sql files, natural-sorted.

    Sources are tried in config order; the first with any files wins (Atlas
    supersedes legacy Flyway). The losing source is excluded entirely.
    """
    for source in (migrations_cfg or {}).get("sources", []):
        prefix = source["path_prefix"].rstrip("/") + "/"
        excludes = set(source.get("exclude_files", []))
        matched = [
            f
            for f in files
            if f.file_path.startswith(prefix)
            and f.file_path.rsplit("/", 1)[-1] not in excludes
            and f.file_path.lower().endswith(".sql")
        ]
        if matched:
            return sorted(matched, key=lambda f: natural_sort_key(f.file_path))
    return []
