# plan4me

<p align="center">
  <img src="docs/assets/plan4me-banner.png" alt="plan4me — collective knowledge from video" width="100%" />
</p>

Turn hours of YouTube interviews, talks, and podcasts into one **evidence-backed knowledge guide**.

plan4me searches YouTube for a topic, pulls transcripts (captions, optional Deepgram/Whisper), extracts cited “knowledge atoms” with your choice of LLM provider, clusters overlapping claims, and synthesizes a Markdown report you can read in the web UI—or hand to a coding agent via MCP.

## Features

- **Topic → knowledge guide** — search, transcribe, extract, cluster, synthesize
- **Cited insights** — claims tied back to videos (and timestamps when available)
- **Web app** — Next.js UI with stage progress and a Markdown guide reader
- **Async research jobs** — start a run, poll progress (`search` → `synthesize`)
- **Pluggable LLM provider** — Bedrock, OpenAI, Anthropic, Gemini, Ollama, or Moonshot via one `LLM_PROVIDER` setting
- **MCP server** — Cursor / Claude agents can call `health`, `search_videos`, `research_topic`, `get_latest_report`
- **Cursor skill + agent** — project skill `video-research` and agent `video-researcher` under `.cursor/`

## Tech stack

| Layer | Stack |
|-------|--------|
| Frontend | Next.js 15, React 19, TypeScript, react-markdown |
| API | FastAPI, Uvicorn, Pydantic Settings |
| Pipeline | LangGraph / LangChain; pluggable chat provider + pluggable embeddings |
| Ingestion | yt-dlp, youtube-transcript-api, optional Deepgram / faster-whisper |
| Storage | SQLite (`plan4me.db`) |
| Agents | MCP stdio server (`mcp_server`), Cursor skills/agents |

## Quick start

### Prerequisites

- **Python 3.11+** recommended (very new Pythons may lack wheels for optional Whisper deps)
- **Node.js 20+** (for the frontend)
- **One LLM provider.** Either AWS credentials with Bedrock model access, or an API key for
  OpenAI / Anthropic / Gemini / Moonshot, or a local [Ollama](https://ollama.com) server
- Optional: **Deepgram API key** for caption-less top videos
- Optional: **ffmpeg** on `PATH` if you use Deepgram/Whisper audio paths

### 1. Clone and configure

```bash
git clone https://github.com/AffanShaikhsurab/plan4me.git
cd plan4me
cp .env.example .env
# Edit .env — set AWS region/models; optionally DEEPGRAM_API_KEY
```

PowerShell:

```powershell
git clone https://github.com/AffanShaikhsurab/plan4me.git
Set-Location plan4me
Copy-Item .env.example .env
# Edit .env
```

Credentials use the standard AWS chain (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, SSO, etc.).

### 2. Backend

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# Optional local Whisper fallback:
# pip install -r requirements-whisper.txt

uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)  
Provider check: [http://127.0.0.1:8000/health/llm](http://127.0.0.1:8000/health/llm)  
Embeddings check: [http://127.0.0.1:8000/health/embeddings](http://127.0.0.1:8000/health/embeddings)

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The UI calls `http://localhost:8000` by default (`NEXT_PUBLIC_API_BASE`).

### Choosing an LLM provider

Bedrock is the default. To use anything else, set `LLM_PROVIDER` in `.env` and
install that provider's package:

```bash
pip install -r requirements-providers.txt   # anthropic + gemini + ollama
```

| `LLM_PROVIDER` | Needs | Default models (extraction / synthesis) |
|---|---|---|
| `bedrock` | AWS creds + model access | `EXTRACTION_MODEL_ID` / `SYNTHESIS_MODEL_ID` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6-luna` / `gpt-5.6-sol` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` / `claude-opus-5` |
| `gemini` | `GOOGLE_API_KEY` | `gemini-3.5-flash-lite` / `gemini-3.7-flash` |
| `moonshot` | `MOONSHOT_API_KEY` | `kimi-k2.5` / `kimi-k2.5` |
| `ollama` | a running Ollama server | `llama3.1` / `llama3.1` |

Override either model with `EXTRACTION_MODEL` / `SYNTHESIS_MODEL`.

Clustering needs **embeddings**, which most chat vendors do not serve. Pick one:

- `EMBEDDING_PROVIDER=bedrock` — Titan (requires AWS)
- `EMBEDDING_PROVIDER=openai` — any OpenAI-compatible `/embeddings` endpoint
- `EMBEDDING_PROVIDER=local` — keyless hashed TF-IDF; no network, no torch, and
  weaker at matching paraphrases that share no vocabulary

`OPENAI_BASE_URL` and `MOONSHOT_BASE_URL` let the OpenAI-compatible providers
point at any gateway (vLLM, OpenRouter, Together).

Adding a provider means one class in `backend/llm/providers/chat.py` decorated
with `@register_chat`, implementing `_build()`. No dispatch code changes.

### 4. Smoke test (no UI)

```bash
# From repo root, venv active
python -m scripts.smoke_test
# Include live extraction/synthesis (costs tokens):
python -m scripts.smoke_test --full --topic "B2B SaaS pricing interviews"
```

## Environment variables

See [`.env.example`](.env.example). Summary:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `bedrock` (default), `openai`, `anthropic`, `gemini`, `ollama`, `moonshot` |
| `EXTRACTION_MODEL` / `SYNTHESIS_MODEL` | Override models for the active provider (blank = provider default) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `MOONSHOT_API_KEY` | Credential for the chosen provider |
| `OLLAMA_BASE_URL` | Ollama server (default `http://localhost:11434`) |
| `EMBEDDING_PROVIDER` | `bedrock` (Titan), `openai`, or `local` (keyless TF-IDF) |
| `AWS_REGION` | Bedrock region (default `us-east-1`) |
| `EXTRACTION_MODEL_ID` | Fast/cheap model for atom extraction |
| `SYNTHESIS_MODEL_ID` | Stronger model for final Markdown guide |
| `EMBEDDING_MODEL_ID` | Embeddings for clustering (default Titan v2) |
| `NUM_CANDIDATES` / `TARGET_VIDEOS` / `FORCE_DEEPGRAM_TOP_N` | Search + selection policy |
| `TRANSCRIPT_LANGUAGES` | Caption language priority (e.g. `en` or `en,hi`) |
| `DEEPGRAM_API_KEY` / `DEEPGRAM_MODEL` | Paid STT for top caption-less videos |
| `DEDUPE_SIMILARITY_THRESHOLD` | Clustering similarity cutoff |
| `ENABLE_WHISPER_FALLBACK` + `WHISPER_*` | Optional local STT |
| `DB_PATH` | SQLite path (default `plan4me.db`) |
| `NEXT_PUBLIC_API_BASE` | Frontend → API base (default `http://localhost:8000`) |

`.env.local` overrides `.env` when present.

## Project structure

```
plan4me/
├── backend/           # FastAPI app, ingestion, LLM providers, LangGraph pipeline, SQLite
├── frontend/          # Next.js UI
├── mcp_server/        # stdio MCP tools for agents
├── scripts/           # smoke_test, Deepgram probe, model probes, MCP setup
├── docs/              # MCP & skills guide
├── .cursor/           # MCP example, video-research skill, video-researcher agent
├── .env.example
├── requirements.txt
└── requirements-whisper.txt
```

## Development scripts

| Command | What it does |
|---------|----------------|
| `uvicorn backend.api.main:app --reload --port 8000` | API + pipeline |
| `cd frontend && npm run dev` | Next.js dev server |
| `cd frontend && npm run build` / `npm run lint` | Production build / lint |
| `python -m mcp_server` | Run MCP server (stdio; usually launched by Cursor) |
| `python -m scripts.smoke_test` | Cheap layer checks |
| `python -m scripts.smoke_test --full` | + live LLM layers |
| `pytest tests -q` | Unit tests (providers, pipeline, API) |
| `./scripts/setup_mcp.sh` or `.\scripts\setup_mcp.ps1` | Generate local `.cursor/mcp.json` |
| `python scripts/test_deepgram.py` | Deepgram path check |
| `python scripts/probe_models.py` | Bedrock model probe |
| `python -m scripts.list_moonshot_models` | List models a Moonshot key can call |

## Cursor MCP & skills

plan4me ships a **plan4me** MCP server and a **video-research** skill so agents can run the same pipeline from chat.

```powershell
# Windows — from repo root, after creating .venv
.\scripts\setup_mcp.ps1
```

```bash
# macOS / Linux
./scripts/setup_mcp.sh
```

Then reload MCP in Cursor and call `health`.

Full setup, tools, and troubleshooting: **[docs/MCP_AND_SKILLS.md](docs/MCP_AND_SKILLS.md)**

- Skill: [`.cursor/skills/video-research/SKILL.md`](.cursor/skills/video-research/SKILL.md)
- Agent: [`.cursor/agents/video-researcher.md`](.cursor/agents/video-researcher.md)

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for fork/clone, branches, commits, PRs, and local testing.

## License

[MIT](LICENSE) © Affan Shaikhsurab
