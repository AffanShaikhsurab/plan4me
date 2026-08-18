# MCP and Cursor skills for plan4me

This project exposes the research pipeline to agents through a **stdio MCP server** and ships a Cursor **skill** + **agent** that know when and how to use it.

| Piece | Location |
|-------|----------|
| MCP server package | `mcp_server/` (`python -m mcp_server`) |
| Cursor MCP config (local) | `.cursor/mcp.json` (generated; gitignored) |
| MCP config template | `.cursor/mcp.json.example` |
| Skill | `.cursor/skills/video-research/SKILL.md` |
| Agent | `.cursor/agents/video-researcher.md` |
| Setup scripts | `scripts/setup_mcp.sh`, `scripts/setup_mcp.ps1` |

## MCP tools

| Tool | Cost / duration | Use |
|------|-----------------|-----|
| `health` | Cheap LLM provider ping | Verify AWS/models before a long run |
| `search_videos` | Cheap (yt-dlp only) | Preview YouTube candidates |
| `research_topic` | **Long** (minutes) + LLM ± Deepgram | Full pipeline → cited Markdown |
| `get_latest_report` | Free (SQLite) | Recover last saved report |

Typical agent flow: clarify topic → optional `health` → `search_videos` → `research_topic` → present Markdown (or `get_latest_report` if interrupted).

## Easy connect (recommended)

### 1. Install backend deps

**bash**

```bash
cd /path/to/plan4me
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env   # then edit
```

**PowerShell**

```powershell
Set-Location "C:\path\to\plan4me"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
# Edit .env — LLM_PROVIDER and its credential, optional DEEPGRAM_API_KEY
```

### 2. Generate local MCP config

**Windows PowerShell**

```powershell
.\scripts\setup_mcp.ps1
```

**macOS / Linux**

```bash
chmod +x scripts/setup_mcp.sh
./scripts/setup_mcp.sh
```

This writes `.cursor/mcp.json` with the correct absolute paths for your clone and venv. That file is gitignored so personal paths are never committed.

### 3. Reload in Cursor

1. Ensure `.env` (or `.env.local`) exists at the repo root — the MCP process loads both.
2. Cursor Settings → MCP → restart **plan4me** (or reopen the project).
3. In chat, ask an agent to call `health`. Expect `"llm": "ok"` when credentials and model IDs are valid.

### Manual check (stdio)

```bash
# From repo root, venv active — waits on stdio; Ctrl+C to stop
python -m mcp_server
```

## Configure by hand (optional)

Copy the example and replace placeholders:

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
```

```json
{
  "mcpServers": {
    "plan4me": {
      "command": "REPLACE_WITH_VENV_PYTHON",
      "args": ["-m", "mcp_server"],
      "cwd": "REPLACE_WITH_REPO_ROOT",
      "env": {
        "PYTHONPATH": "REPLACE_WITH_REPO_ROOT"
      }
    }
  }
}
```

### Print paths to paste

**PowerShell**

```powershell
Set-Location "C:\path\to\plan4me"
$root = (Get-Location).Path
$py = Join-Path $root ".venv\Scripts\python.exe"
Write-Host "command: $py"
Write-Host "cwd / PYTHONPATH: $root"
```

**bash**

```bash
cd /path/to/plan4me
ROOT="$(pwd)"
echo "command: $ROOT/.venv/bin/python"
echo "cwd / PYTHONPATH: $ROOT"
```

## Using the video-research skill

The skill at `.cursor/skills/video-research/SKILL.md` is a **project skill** (name: `video-research`).

**When it applies:** research that needs many YouTube interviews/talks/podcasts; evidence-backed guides from video; comprehensive passes before deep topic work.

**When not:** docs-only or code-only tasks.

Agents that load the skill should:

1. Clarify topic + angles
2. Optionally call `health`
3. `search_videos` per angle
4. `research_topic` (often `max_videos` 8–15)
5. Synthesize for the user without inventing sources

Ask in chat, e.g. “Use the plan4me MCP / video-research skill on …”

## Using the video-researcher agent

[`.cursor/agents/video-researcher.md`](../.cursor/agents/video-researcher.md) defines a specialized agent that prefers plan4me tools over browsing individual YouTube pages for bulk research.

Use it (or Task subagents) when fanning out **multi-angle** topics: one `research_topic` per angle, then merge takeaways and contradictions.

## Manual tool smoke (from an agent)

1. `health` — settings + LLM provider check
2. `search_videos` with `query` and `max_results` 10–15
3. `research_topic` with a concrete YouTube-style query and modest `max_videos` (e.g. 5 while testing)
4. `get_latest_report` if the run was interrupted or you need the last saved Markdown

Phrase `topic` like a search for expert talks (e.g. `"B2B SaaS pricing interviews"`), not a vague essay title. Pass `languages` as comma-separated codes when needed (`en`, `en,hi`).

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| MCP won’t start | Venv Python path in `mcp.json`; `pip install -r requirements.txt`; run `setup_mcp` again |
| Import errors for `backend` | `PYTHONPATH` and `cwd` both set to repo root |
| `llm: "error"` in `health` (see `llm_error`) | AWS credentials, region, model IDs enabled in Bedrock |
| Empty / blocked transcripts | Captions availability; Deepgram key; network / YouTube rate limits |
| Long hang on `research_topic` | Expected for full runs; use `get_latest_report` before retrying |
| Wrong DB file | Server `chdir`s to repo root; `DB_PATH` is relative to that |

## Related docs

- [README.md](../README.md) — product quick start and env vars
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR and testing expectations
- [`.env.example`](../.env.example) — full env reference
