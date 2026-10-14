# Backend Dev Agent

AI-powered backend development agent. Currently at **Phase 1: Project Foundation**
(FastAPI app, configuration, logging, health check, tests).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
```

## Run

```powershell
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/health.

## Test

```powershell
pytest
```

## Configuration

Settings are read from environment variables with the `AGENT_` prefix (or `.env`).
See `.env.example` and `app/config.py`.

## Gemini (LLM integration)

Put `GEMINI_API_KEY` in your local `.env` (never commit it). Without it, only the `/llm/*`
endpoints are unavailable; health and repository endpoints still work.

- `POST /llm/test`: body `{"prompt": "...", "max_output_tokens": 256}`. Returns text, token usage,
  finish reason, and estimated cost (`status` is `unknown` when usage or pricing is missing).
- `GET /llm/usage`: in-process totals since startup (requests, failures by kind, tokens, estimated USD).

Default model: `gemini-3.5-flash-lite` (`AGENT_GEMINI_MODEL`). `gemini-2.5-flash` is no longer
available to new users. Prices are in `app/llm/token_tracker.py`, verified against
https://ai.google.dev/gemini-api/docs/pricing (2026-10-13). Retries are disabled.

Live check (makes one small paid request, skipped by default):
`RUN_LIVE_GEMINI=1 pytest tests/test_llm_live.py -s`

## Bug analysis (`POST /agent/analyze`)

Body: `{"path": "<repo dir inside AGENT_ALLOWED_REPO_ROOT>", "bug_description": "..."}`.

The workflow searches the repository deterministically first, selects a bounded set of excerpts,
and makes at most one Gemini call. The response separates `confirmed_facts` (search and Git
results) from `hypotheses` (model output). Each hypothesis has evidence marked `verified` only when
the quoted text appears in the excerpts that were sent. Nothing in the repository is modified.

Status values: `completed`, `insufficient_context` (no match, no model call), `llm_unavailable`
(no key), `llm_error` (provider error, reported with its kind), `invalid_model_output`.
Limits: `AGENT_CONTEXT_MAX_FILES`, `AGENT_CONTEXT_MAX_CHARS`, `AGENT_CONTEXT_WINDOW_LINES`,
`AGENT_CONTEXT_MAX_TERMS`, `AGENT_MAX_MATCHES_PER_TERM`, `AGENT_ANALYSIS_MAX_OUTPUT_TOKENS`.

## Fix proposals (`POST /agent/fix-proposal`)

Body: `{"path": "<repo dir>", "bug_description": "...", "analysis": <report from /agent/analyze>}`.
The analysis report is passed in, so its model call is not repeated. The cited quotes are located again
in the current files, and at most one Gemini call asks for a unified diff. Returned statuses:
`proposed`, `no_patch`, `patch_rejected`, `insufficient_evidence`, `llm_unavailable`, `llm_error`,
`invalid_model_output`.

A proposed diff is only a draft. It is validated for structure, target paths (inside the analyzed
directory, not excluded or sensitive, part of the context sent), size limits, and whether every hunk
matches the current file. Python results are parsed with `ast`, without running them. Nothing is
applied, compiled, or tested, and the response says so. Limits: `AGENT_FIX_*` in `.env.example`.

## Repository analysis (read-only)

Set `AGENT_ALLOWED_REPO_ROOT` to a directory. Each request's `path` must resolve to a
directory inside that root. Endpoints (all `POST`, JSON body with `path`):

- `/repo/summary`: file counts and extension breakdown. No file contents.
- `/repo/search`: literal text search (`query`, optional `extensions`, `case_sensitive`, `max_results`).
- `/repo/git/status`: branch and changed files.
- `/repo/git/diff`: unstaged diff, or staged with `"staged": true`.

Virtual environments, `node_modules`, build outputs, `.env` files, and key or credential files
are never read. Git commands are read-only and run without a shell.
