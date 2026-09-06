"""Code Summarisation Agent — FastAPI service.

Endpoints:
    GET  /health                      liveness check
    POST /summarize/batch             summarise all configured repos → per-repo summaries.json + .md
    POST /webhooks/github             refresh summaries for merged GitHub pull requests
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

from github_fetcher import GitHubFetcher
from migrations import (
    is_migration_path,
    select_migration_files,
)
from summariser import Summariser

TEMP_DIR = Path(os.getenv("TEMP_DIR", "/app/temp"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

app = FastAPI(title="Code Summarisation Agent")
_summariser = Summariser()


def _load_config() -> dict:
    try:
        import yaml

        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        log.warning("Could not read config.yaml. error=%s", exc)
        return {}


def _tenant_context(cfg: dict) -> str:
    ctx = cfg.get("tenant_context")
    if not ctx:
        raise HTTPException(
            status_code=500,
            detail="tenant_context is not set in config.yaml.",
        )
    return ctx


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "temp_dir": str(TEMP_DIR)}


def _write_markdown(repo_name: str, file_path: str, url: str, summary: str) -> Path:
    """Write a readable per-file summary under summaries/<repo>/, mirroring repo structure."""
    rel = file_path if file_path.endswith(".md") else f"{file_path}.md"
    out_path = TEMP_DIR / repo_name / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"<!-- file: {file_path} -->\n<!-- url: {url} -->\n\n{summary}")
    return out_path


def _load_existing_summaries(repo_name: str) -> dict[str, dict]:
    """Return a map of file_path → entry from an existing summaries.json, or {} if absent."""
    summaries_path = TEMP_DIR / repo_name / "summaries.json"
    if not summaries_path.exists():
        return {}
    try:
        data = json.loads(summaries_path.read_text())
        return {entry["file_path"]: entry for entry in data.get("files", [])}
    except Exception:
        return {}


# Verifies GitHub's HMAC signature before accepting a webhook payload.
def _verify_github_signature(body: bytes, signature: str | None) -> None:
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="GITHUB_WEBHOOK_SECRET is not configured.")
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing GitHub webhook signature.")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature.")


# Checks whether a GitHub webhook repo is present in config.yaml.
def _repo_is_configured(repo_name: str, repos: list[str]) -> bool:
    return repo_name in {repo.split("/")[-1] for repo in repos if repo}


# Removes a stale local markdown summary for deleted or renamed files.
def _remove_local_summary(repo_name: str, file_path: str) -> None:
    rel = file_path if file_path.endswith(".md") else f"{file_path}.md"
    summary_path = TEMP_DIR / repo_name / rel
    if summary_path.exists():
        summary_path.unlink()


# Summarises a provided list of fetched files and returns successes plus errors.
async def _summarise_files(
    *,
    repo_name: str,
    files: list,
    tenant_context: str,
) -> tuple[list[dict], list[dict]]:
    sem = asyncio.Semaphore(5)
    errors: list[dict] = []

    async def _summarise_one(f) -> dict | None:
        async with sem:
            try:
                summary, _, _, out_tok = await _summariser.summarise(
                    repo=repo_name,
                    file_path=f.file_path,
                    content=f.content,
                    language=f.language,
                    tenant_context=tenant_context,
                )
                _write_markdown(repo_name, f.file_path, f.url, summary)
                log.info("Summarised. file=%s tokens_out=%d", f.file_path, out_tok)
                return {
                    "file_path": f.file_path,
                    "url": f.url,
                    "source_code": f.content,
                    "summary": summary,
                    "last_modified": f.last_modified,
                }
            except Exception as exc:
                log.warning("Summarise failed. file=%s error=%s", f.file_path, exc)
                errors.append({"file_path": f.file_path, "error": str(exc)})
                return None

    outcomes = await asyncio.gather(*[_summarise_one(f) for f in files])
    return [result for result in outcomes if result is not None], errors


# Writes summaries.json and failed_files.json for one repo.
def _write_repo_outputs(repo_name: str, file_entries: list[dict], errors: list[dict]) -> tuple[Path, Path | None]:
    repo_dir = TEMP_DIR / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = repo_dir / "summaries.json"
    summaries_path.write_text(
        json.dumps({"repository": repo_name, "files": file_entries}, indent=2, ensure_ascii=False)
    )

    failed_path = repo_dir / "failed_files.json"
    if errors:
        failed_path.write_text(json.dumps({"repository": repo_name, "failed": errors}, indent=2, ensure_ascii=False))
        return summaries_path, failed_path
    if failed_path.exists():
        failed_path.unlink()
    return summaries_path, None


async def _summarise_repo(
    *,
    fetcher: GitHubFetcher,
    org: str,
    repo: str,
    tenant_context: str,
    migrations_cfg: dict | None,
    force: bool = False,
) -> dict:
    """Summarise one repo → summaries/<repo>/summaries.json + readable .md artifacts."""
    repo_full = f"{org}/{repo}" if org else repo
    repo_name = repo.split("/")[-1]

    log.info("Starting repo. repo=%s force=%s", repo_full, force)
    fetched_files = await fetcher.fetch_all(repo_full, ref="main")

    migration_files = select_migration_files(fetched_files, migrations_cfg)
    regular_files = [f for f in fetched_files if not is_migration_path(f.file_path, migrations_cfg)]

    existing = {} if force else _load_existing_summaries(repo_name)
    skipped_entries = [entry for fp, entry in existing.items() if any(f.file_path == fp for f in regular_files)]
    files_to_summarise = [f for f in regular_files if f.file_path not in existing]
    log.info("File selection. repo=%s total=%d skipped=%d to_summarise=%d", repo_full, len(regular_files), len(skipped_entries), len(files_to_summarise))

    new_entries, errors = await _summarise_files(
        repo_name=repo_name,
        files=files_to_summarise,
        tenant_context=tenant_context,
    )
    file_entries = skipped_entries + new_entries

    ran_migrations = False
    _schema_base = (migrations_cfg or {}).get("output_file", "schema_state.md")
    schema_state_name = f"{Path(_schema_base).stem}_{repo_name}{Path(_schema_base).suffix}"
    migration_already_done = not force and schema_state_name in existing
    if migration_files and not migration_already_done:
        try:
            summary, _, _, _ = await _summariser.summarise_migrations(
                repo=repo_name,
                files=[(f.file_path, f.content) for f in migration_files],
                tenant_context=tenant_context,
            )
            _write_markdown(repo_name, schema_state_name, "", summary)
            file_entries.append({
                "file_path": schema_state_name,
                "url": "",
                "source_code": "",
                "summary": summary,
                "last_modified": None,
            })
            ran_migrations = True
            log.info("Migration aggregate complete. repo=%s files=%d", repo_name, len(migration_files))
        except Exception as exc:
            log.warning("Migration aggregate failed, skipping. repo=%s error=%s", repo_name, exc)
            errors.append({"file_path": "migration_aggregate", "error": str(exc)})
    elif migration_already_done:
        log.info("Skipping migration aggregate (already summarised). repo=%s", repo_name)
        file_entries.append(existing[schema_state_name])
        ran_migrations = True

    summaries_path, failed_path = _write_repo_outputs(repo_name, file_entries, errors)
    log.info("Wrote summaries.json. path=%s files=%d", summaries_path, len(file_entries))

    if failed_path:
        log.info("Wrote failed_files.json. path=%s count=%d", failed_path, len(errors))

    return {
        "repo": repo_full,
        "summarised": len(file_entries),
        "skipped": len(skipped_entries),
        "migrations": ran_migrations,
        "errors": len(errors),
        "failed_files": str(failed_path) if failed_path else None,
        "output": str(summaries_path),
    }


@app.post("/summarize/batch")
async def summarize_batch(force: bool = False) -> dict:
    """Summarise every configured repo → per-repo summaries.json + readable .md artifacts.
    force=false (default): skip files already present in summaries.json, retry any that failed.
    force=true: re-summarise all files regardless.
    """
    cfg = _load_config()
    repos = [repo for repo in cfg.get("repos", []) if repo]
    if not repos:
        raise HTTPException(
            status_code=500,
            detail="No repositories configured. Set repos in config.yaml.",
        )

    org = cfg.get("github_org", "")
    tenant_context = _tenant_context(cfg)
    migrations_cfg = cfg.get("migrations")

    try:
        fetcher = GitHubFetcher()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GitHub client init failed: {exc}")

    log.info("Starting batch. repos=%d force=%s", len(repos), force)
    results: list[dict] = []
    for repo in repos:
        try:
            results.append(
                await _summarise_repo(
                    fetcher=fetcher,
                    org=org,
                    repo=repo,
                    tenant_context=tenant_context,
                    migrations_cfg=migrations_cfg,
                    force=force,
                )
            )
        except Exception as exc:
            log.error("Repo batch failed. repo=%s error=%s", repo, exc, exc_info=True)
            results.append({"repo": repo, "error": str(exc)})

    return {
        "repos": len(repos),
        "summarised": sum(r.get("summarised", 0) for r in results),
        "results": results,
    }


# Handles merged GitHub pull requests by refreshing local summaries for changed files.
@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    _verify_github_signature(body, x_hub_signature_256)

    if x_github_event != "pull_request":
        return {"status": "ignored", "reason": "not a pull_request event"}

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    pull_request = payload.get("pull_request") or {}
    if payload.get("action") != "closed" or not pull_request.get("merged"):
        return {"status": "ignored", "reason": "not a merged pull request"}

    cfg = _load_config()
    repo_name = (payload.get("repository") or {}).get("name", "")
    repos = [repo for repo in cfg.get("repos", []) if repo]
    if not _repo_is_configured(repo_name, repos):
        return {"status": "ignored", "reason": f"{repo_name} is not configured"}

    org = cfg.get("github_org", "")
    repo_full = (payload.get("repository") or {}).get("full_name") or f"{org}/{repo_name}"
    base_ref = ((pull_request.get("base") or {}).get("ref")) or "main"
    pr_number = int(pull_request["number"])

    try:
        fetcher = GitHubFetcher()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GitHub client init failed: {exc}") from exc

    changed_files, deleted_paths, skipped = await fetcher.fetch_pull_request_changes(
        repo_full,
        pr_number,
        ref=base_ref,
    )
    existing = _load_existing_summaries(repo_name)
    for deleted_path in deleted_paths:
        existing.pop(deleted_path, None)
        _remove_local_summary(repo_name, deleted_path)

    new_entries, errors = await _summarise_files(
        repo_name=repo_name,
        files=changed_files,
        tenant_context=_tenant_context(cfg),
    )
    merged_entries = {**existing, **{entry["file_path"]: entry for entry in new_entries}}
    file_entries = sorted(merged_entries.values(), key=lambda entry: entry["file_path"])
    summaries_path, failed_path = _write_repo_outputs(repo_name, file_entries, errors)

    return {
        "status": "ok",
        "repo": repo_full,
        "pull_request": pr_number,
        "summarised": len(new_entries),
        "deleted": len(deleted_paths),
        "skipped": skipped,
        "errors": len(errors),
        "output": str(summaries_path),
        "failed_files": str(failed_path) if failed_path else None,
    }
