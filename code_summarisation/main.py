"""Code Summarisation Agent — FastAPI service.

Endpoints:
    GET  /health                      liveness check
    POST /summarize                   single file → markdown summary
    POST /summarize/batch             summarise all configured repos → per-repo summaries.json + .md
    POST /summarize/migrations        aggregate SQL migration files → schema_state summary
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

from github_fetcher import GitHubFetcher
from migrations import (
    is_migration_path,
    natural_sort_key,
    select_migration_files,
)
from models import (
    MigrationAggregateRequest,
    SummarizeRequest,
    SummarizeResponse,
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


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest) -> SummarizeResponse:
    """Summarise a single file. Pass raw content + file_path."""
    cfg = _load_config()
    repos = [repo for repo in cfg.get("repos", []) if repo]
    if not repos:
        raise HTTPException(
            status_code=500,
            detail="No repositories configured. Set repos in config.yaml.",
        )
    repo_name = repos[0].split("/")[-1]
    tenant_context = _tenant_context(cfg)
    try:
        summary, model_used, in_tok, out_tok = await _summariser.summarise(
            repo=repo_name,
            file_path=req.file_path,
            content=req.content,
            language=req.language,
            model=req.model,
            tenant_context=tenant_context,
        )
    except Exception as exc:
        log.error("LLM call failed. file_path=%s error=%s", req.file_path, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    safe_name = req.file_path.replace("/", "__").replace("\\", "__")
    out_path = TEMP_DIR / f"{safe_name}.md"
    out_path.write_text(
        f"<!-- file: {req.file_path} -->\n<!-- url: {req.url} -->\n\n{summary}"
    )
    log.info("Wrote summary. path=%s file_path=%s", out_path, req.file_path)

    return SummarizeResponse(
        file_path=req.file_path,
        url=req.url,
        summary=summary,
        model_used=model_used,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


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

    outcomes = await asyncio.gather(*[_summarise_one(f) for f in files_to_summarise])
    new_entries = [r for r in outcomes if r is not None]
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

    repo_dir = TEMP_DIR / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    output = {"repository": repo_name, "files": file_entries}
    summaries_path = repo_dir / "summaries.json"
    summaries_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("Wrote summaries.json. path=%s files=%d", summaries_path, len(file_entries))

    failed_path = repo_dir / "failed_files.json"
    if errors:
        failed_path.write_text(json.dumps({"repository": repo_name, "failed": errors}, indent=2, ensure_ascii=False))
        log.info("Wrote failed_files.json. path=%s count=%d", failed_path, len(errors))
    elif failed_path.exists():
        failed_path.unlink()

    return {
        "repo": repo_full,
        "summarised": len(file_entries),
        "skipped": len(skipped_entries),
        "migrations": ran_migrations,
        "errors": len(errors),
        "failed_files": str(failed_path) if errors else None,
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


@app.post("/summarize/migrations", response_model=SummarizeResponse)
async def summarize_migrations(req: MigrationAggregateRequest) -> SummarizeResponse:
    """Aggregate all SQL migration files into a single current-state schema summary."""
    if not req.files:
        raise HTTPException(status_code=422, detail="At least one migration file is required.")

    ordered = sorted(req.files, key=lambda f: natural_sort_key(f.file_path))
    tenant_context = _tenant_context(_load_config())
    try:
        summary, model_used, in_tok, out_tok = await _summariser.summarise_migrations(
            repo=req.repo,
            files=[(f.file_path, f.content) for f in ordered],
            model=req.model,
            tenant_context=tenant_context,
        )
    except Exception as exc:
        log.error("Migration aggregate LLM call failed. repo=%s error=%s", req.repo, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    schema_state_name = f"schema_state_{req.repo.split('/')[-1]}.md"
    out_path = TEMP_DIR / schema_state_name
    out_path.write_text(f"<!-- repo: {req.repo} -->\n\n{summary}")
    log.info("Wrote migration aggregate. path=%s repo=%s files=%d", out_path, req.repo, len(ordered))

    return SummarizeResponse(
        file_path=schema_state_name,
        url="",
        summary=summary,
        model_used=model_used,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
