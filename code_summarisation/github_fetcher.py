"""Fetch source files from GitHub applying config.yaml include/exclude rules."""

import fnmatch
import logging
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator, List, Optional

import httpx
import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@dataclass
class FetchedFile:
    file_path: str          # repo-relative path  e.g. "src/Services/FooService.cs"
    content: str
    url: str                # GitHub blob URL
    language: str
    last_modified: Optional[str] = None


def _load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _detect_language(file_path: str) -> str:
    ext = PurePosixPath(file_path).suffix.lower()
    return {
        ".cs": "csharp",
        ".sql": "sql",
        ".json": "json",
        ".py": "python",
        ".ts": "typescript",
        ".js": "javascript",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(ext, "text")


def _is_excluded_dir(path: str, exclude_dirs: list[str]) -> bool:
    parts = PurePosixPath(path).parts
    dir_parts = parts[:-1]  # directory components only, not the filename

    for pattern in exclude_dirs:
        if "/" in pattern:
            # Path-prefix pattern like "k8s/base" or "k8s/overlays"
            prefix = pattern.rstrip("/*")
            if path.startswith(prefix + "/") or path == prefix:
                return True
        else:
            # Plain name or glob like "bin", "*.Tests" — match each dir component
            for part in dir_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
    return False


def _is_included_path(path: str, include_path_prefixes: list[str]) -> bool:
    return any(path.startswith(prefix) for prefix in include_path_prefixes)


def _is_excluded_file(filename: str, exclude_files: list[str]) -> bool:
    for pattern in exclude_files:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


def _passes_filter(path: str, config: dict) -> bool:
    """Return True if this file path should be summarised."""
    include_exts = set(config.get("include_extensions", []))
    exclude_dirs = config.get("exclude_dirs", [])
    exclude_files = config.get("exclude_files", [])
    include_prefixes = config.get("include_path_prefixes", [])

    filename = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix.lower()

    if suffix not in include_exts:
        return False
    if _is_excluded_file(filename, exclude_files):
        return False
    # Always include explicitly listed path prefixes
    if _is_included_path(path, include_prefixes):
        return True
    if _is_excluded_dir(path, exclude_dirs):
        return False
    return True


class GitHubFetcher:
    BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None) -> None:
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is required")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def list_files(self, repo: str, ref: str = "main") -> List[dict]:
        """Return list of {path, url, sha} for all tree entries matching config filters."""
        config = _load_config()
        url = f"{self.BASE}/repos/{repo}/git/trees/{ref}?recursive=1"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        tree = data.get("tree", [])
        files = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = entry["path"]
            if _passes_filter(path, config):
                blob_url = f"https://github.com/{repo}/blob/{ref}/{path}"
                files.append({"path": path, "url": blob_url, "sha": entry["sha"]})

        log.info(f"Found {len(files)} qualifying files in {repo}@{ref}")
        return files

    async def fetch_content(self, repo: str, path: str, ref: str = "main") -> str:
        """Fetch raw file content."""
        url = f"{self.BASE}/repos/{repo}/contents/{path}?ref={ref}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        import base64
        raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return raw

    async def fetch_all(self, repo: str, ref: str = "main") -> List[FetchedFile]:
        """Fetch content for all filtered files. Returns list of FetchedFile."""
        file_list = await self.list_files(repo, ref)
        results: List[FetchedFile] = []

        for i, entry in enumerate(file_list, 1):
            path = entry["path"]
            try:
                content = await self.fetch_content(repo, path, ref)
                results.append(FetchedFile(
                    file_path=path,
                    content=content,
                    url=entry["url"],
                    language=_detect_language(path),
                ))
                log.info(f"[{i}/{len(file_list)}] Fetched {path}")
            except Exception as exc:
                log.warning(f"Failed to fetch {path}: {exc}")

        return results
