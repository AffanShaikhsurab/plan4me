# Contributing to plan4me

Thanks for helping improve plan4me. This guide covers local setup, workflow, and what we look for in a PR.

## Ways to contribute

- Bug fixes and reliability improvements (ingestion, Bedrock errors, job polling)
- Docs and onboarding (README, MCP/skills, examples)
- Frontend UX for the research/report flow
- Pipeline quality (selection, clustering, synthesis prompts)
- Tests and smoke coverage

Please open an issue first for large features or breaking API/MCP changes.

## Code of conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Prerequisites

- Python 3.11+ with a virtualenv
- Node.js 20+ and npm
- AWS credentials with Bedrock model access in your configured region
- Optional: Deepgram key; ffmpeg for audio transcription paths

## Local setup

1. Fork the repo on GitHub, then clone your fork.
2. Copy env and install deps (see [README.md](README.md) Quick start).
3. Run API + frontend in two terminals.
4. Confirm `GET /health` and optionally `GET /health/bedrock`.
5. Run `python -m scripts.smoke_test` before non-trivial pipeline changes.
6. For agent work: run `./scripts/setup_mcp.sh` or `.\scripts\setup_mcp.ps1`, then read [docs/MCP_AND_SKILLS.md](docs/MCP_AND_SKILLS.md).

## Branch naming

Use short, descriptive branches from `master`:

- `fix/<short-description>`
- `feat/<short-description>`
- `docs/<short-description>`
- `chore/<short-description>`

Examples: `feat/async-job-cancel`, `fix/caption-language-fallback`, `docs/mcp-windows-paths`.

## Commit style

Use **imperative, sentence-case** messages focused on intent:

- `Add portal research flow with quiet progress and guide reader.`
- `Fix caption language fallback when primary track is missing.`

Guidelines:

- Prefer one logical change per commit when practical
- Start with a verb: Add / Fix / Update / Remove / Refactor
- Avoid noisy “WIP” commits on the branch you open as a PR

## Pull requests

1. Push your branch to your fork.
2. Open a PR against `master` on the upstream repo.
3. Fill out the PR template checklist.
4. Keep the diff focused; separate unrelated refactors.
5. Link related issues.
6. Expect review on: correctness, cost/safety of Bedrock/Deepgram calls, secrets not committed, and docs for user-facing changes.

Do **not** commit:

- `.env`, `.env.local`, or API keys
- Personal absolute paths in `.cursor/mcp.json` (that file is gitignored; commit only `.cursor/mcp.json.example`)
- Local databases (`*.db`), audio caches, or `node_modules`

## Code style

### Python (`backend/`, `mcp_server/`, `scripts/`)

- Type hints on public functions; Pydantic models for request/response shapes
- Config only via `backend.config.Settings` / env (no hard-coded secrets)
- Keep FastAPI handlers thin; put pipeline logic in `backend/pipeline/` and domain modules
- Prefer clear `logging` over silent failures on ingestion/LLM paths
- Lazy-import optional heavy deps (Deepgram, Whisper) so core installs stay light

### TypeScript (`frontend/`)

- Match existing Next.js App Router patterns (`frontend/app/`)
- Use typed helpers in `frontend/app/api.ts` for API calls
- Prefer readable UI over new dependencies unless necessary
- Run `npm run lint` in `frontend/` before opening a PR that touches the UI

## Testing

There is no large automated suite yet. Minimum bar:

```bash
# From repo root, venv active
python -m scripts.smoke_test

# When changing Bedrock extraction/synthesis/clustering:
python -m scripts.smoke_test --full --topic "your test topic" --videos 5
```

For API/UI changes:

- Start uvicorn + `npm run dev`
- Run a short research job (low `max_videos`) and confirm stages + report render
- Hit `/health` and `/health/bedrock` if you touch AWS/LLM config

For MCP tool changes:

- Regenerate or restart the plan4me MCP server in Cursor after edits
- Call `health`, then a small `search_videos`, then optionally a short `research_topic`

## Security & cost hygiene

- Never commit credentials or real `.env` files
- Treat `research_topic` / `POST /report` / `POST /research` as **paid** (Bedrock ± Deepgram)
- Prefer low `max_videos` while developing
- Do not expand CORS allowlists to wildcards without discussion
- Report security issues privately — see [SECURITY.md](SECURITY.md)

## Questions

Open a GitHub issue with context (OS, Python/Node versions, relevant logs from smoke test or `/health/bedrock`).
