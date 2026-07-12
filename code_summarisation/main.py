"""Code Summarisation Agent — FastAPI service.

Endpoints:
    GET  /health                      liveness check
    POST /summarize                   single file → markdown summary
    POST /summarize/batch             fetch from GitHub + summarise all files → summaries.json
    POST /summarize/migrations        aggregate SQL migration files → schema_state summary
"""

import asyncio
import json
import logging
import os
import re
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
    try:
        summary, model_used, in_tok, out_tok = await _summariser.summarise(
            repo=repo_name,
            file_path=req.file_path,
            content=req.content,
            language=req.language,
            model=req.model,
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


@app.post("/summarize/batch")
async def summarize_batch() -> dict:
    """Fetch all files from GitHub, summarise them, write summaries.json for the indexer."""
    cfg = _load_config()
    repos = [repo for repo in cfg.get("repos", []) if repo]
    if not repos:
        raise HTTPException(
            status_code=500,
            detail="No repositories configured. Set repos in config.yaml.",
        )

    org = cfg.get("github_org", "")
    repo = repos[0]
    repo_full = f"{org}/{repo}" if org else repo
    repo_name = repo.split("/")[-1]
    migration_prefixes = cfg.get("include_path_prefixes", [])

    log.info("Starting batch. repo=%s", repo_full)

    try:
        fetcher = GitHubFetcher()
        fetched_files = await fetcher.fetch_all(repo_full, ref="main")
    except Exception as exc:
        log.error("GitHub fetch failed. repo=%s error=%s", repo_full, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"GitHub fetch failed: {exc}")

    migration_files = [f for f in fetched_files if any(f.file_path.startswith(p) for p in migration_prefixes)]
    regular_files = [f for f in fetched_files if f not in migration_files]

    sem = asyncio.Semaphore(5)
    file_entries: list[dict] = []
    errors: list[str] = []

    async def _summarise_one(f) -> dict | None:
        async with sem:
            try:
                summary, model_used, in_tok, out_tok = await _summariser.summarise(
                    repo=repo_name,
                    file_path=f.file_path,
                    content=f.content,
                    language=f.language,
                )
                log.info("Summarised. file=%s tokens_out=%d", f.file_path, out_tok)
                return {
                    "file_path": f.file_path,
                    "url": f.url,
                    "source_code": f.content,
                    "summary": summary,
                    "last_modified": f.last_modified,
                }
            except Exception as exc:
                msg = f"{f.file_path}: {exc}"
                log.warning("Summarise failed. file=%s error=%s", f.file_path, exc)
                errors.append(msg)
                return None

    tasks = [_summarise_one(f) for f in regular_files]
    outcomes = await asyncio.gather(*tasks)
    file_entries = [r for r in outcomes if r is not None]

    ran_migrations = False
    if migration_files:
        try:
            summary, _, _, _ = await _summariser.summarise_migrations(
                repo=repo_name,
                files=[(f.file_path, f.content) for f in migration_files],
            )
            file_entries.append({
                "file_path": "schema_state.md",
                "url": "",
                "source_code": "",
                "summary": summary,
                "last_modified": None,
            })
            ran_migrations = True
            log.info("Migration aggregate complete. files=%d", len(migration_files))
        except Exception as exc:
            log.warning("Migration aggregate failed, skipping. error=%s", exc)
            errors.append(f"migration_aggregate: {exc}")

    output = {"repository": repo_name, "files": file_entries}
    summaries_path = TEMP_DIR / "summaries.json"
    summaries_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("Wrote summaries.json. path=%s files=%d", summaries_path, len(file_entries))

    return {
        "repo": repo_full,
        "summarised": len(file_entries),
        "migrations": ran_migrations,
        "errors": len(errors),
        "output": str(summaries_path),
    }


def _natural_sort_key(path: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


@app.post("/summarize/migrations", response_model=SummarizeResponse)
async def summarize_migrations(req: MigrationAggregateRequest) -> SummarizeResponse:
    """Aggregate all SQL migration files into a single current-state schema summary."""
    if not req.files:
        raise HTTPException(status_code=422, detail="At least one migration file is required.")

    ordered = sorted(req.files, key=lambda f: _natural_sort_key(f.file_path))
    try:
        summary, model_used, in_tok, out_tok = await _summariser.summarise_migrations(
            repo=req.repo,
            files=[(f.file_path, f.content) for f in ordered],
            model=req.model,
        )
    except Exception as exc:
        log.error("Migration aggregate LLM call failed. repo=%s error=%s", req.repo, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    out_path = TEMP_DIR / "schema_state.md"
    out_path.write_text(f"<!-- repo: {req.repo} -->\n\n{summary}")
    log.info("Wrote migration aggregate. path=%s repo=%s files=%d", out_path, req.repo, len(ordered))

    return SummarizeResponse(
        file_path="schema_state.md",
        url="",
        summary=summary,
        model_used=model_used,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
