"""stdio MCP server wrapping the plan4me YouTube research pipeline.

Tools:
    health              - settings / active LLM provider reachability summary
    search_videos       - yt-dlp search only (cheap preview)
    research_topic      - full pipeline (long-running): search → transcripts →
                          extract → cluster → synthesize Markdown report
    get_latest_report   - most recently saved report from SQLite

Run (from repo root, with venv activated):
    python -m mcp_server
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is on sys.path so `backend.*` imports resolve when Cursor
# launches this as a stdio subprocess with a cwd that may not be the root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env / .env.local before Settings is constructed. pydantic-settings also
# reads env_file, but dotenv guarantees overrides when the process cwd differs.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_REPO_ROOT / ".env.local", override=True)
except ImportError:
    pass

# Keep cwd at repo root so relative DB_PATH (plan4me.db) lands next to the API.
os.chdir(_REPO_ROOT)

from mcp.server import MCPServer  # noqa: E402

from backend.config import get_settings  # noqa: E402
from backend.ingestion.search import search_videos as yt_search  # noqa: E402
from backend.pipeline.graph import run_pipeline  # noqa: E402
from backend.schemas import KnowledgeReport, cluster_to_insight  # noqa: E402
from backend.storage.store import (  # noqa: E402
    get_latest_report as store_latest,
    init_db,
    save_atoms,
    save_report,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("plan4me.mcp")

mcp = MCPServer("plan4me")


@mcp.tool()
def health() -> dict[str, Any]:
    """Return plan4me settings and a cheap reachability check of the active LLM.

    Use before a long research_topic call to confirm the configured provider's
    credentials and model access. Which provider runs is set by LLM_PROVIDER.
    """
    s = get_settings()
    # Seeded from the raw setting so these keys exist even if provider
    # resolution itself fails (e.g. an unknown LLM_PROVIDER value).
    result: dict[str, Any] = {
        "status": "ok",
        "llm": "unchecked",
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "target_videos": s.target_videos,
        "num_candidates": s.num_candidates,
        "deepgram_enabled": s.deepgram_enabled,
        "whisper_fallback": s.enable_whisper_fallback,
        "db_path": str((_REPO_ROOT / s.db_path).resolve()),
    }

    # Resolved model names come from the provider, not from the Bedrock-only
    # settings, so they stay truthful on every provider.
    try:
        from backend.llm.chat import active_models

        result.update(active_models())
    except Exception as exc:  # noqa: BLE001
        result["status"] = "degraded"
        result["llm"] = "error"
        result["llm_error"] = str(exc)[:300]
        return result

    try:
        from backend.llm.embeddings import describe as describe_embeddings

        result["embeddings"] = describe_embeddings()
    except Exception as exc:  # noqa: BLE001
        # Clustering cannot run without embeddings, so this is not cosmetic:
        # leaving status "ok" would let an agent start a long research_topic
        # run after a preflight that already knows it will fail.
        result["status"] = "degraded"
        result["embeddings"] = {"error": str(exc)[:200]}

    try:
        from backend.llm.chat import get_extraction_llm

        resp = get_extraction_llm().invoke("Reply with the single word: ok")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        result["llm"] = "ok"
        result["llm_reply"] = text.strip()[:50]
    except Exception as exc:  # noqa: BLE001
        result["status"] = "degraded"
        result["llm"] = "error"
        result["llm_error"] = str(exc)[:300]
    return result


@mcp.tool()
def search_videos(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search YouTube for videos matching a topic (no transcripts or LLM).

    Cheap preview of candidates before calling research_topic. Returns video
    id, title, url, channel, duration, and view_count.
    """
    if not query.strip():
        return []
    max_results = max(1, min(max_results, 50))
    videos = yt_search(query.strip(), max_results)
    return [v.model_dump() for v in videos]


@mcp.tool()
def research_topic(
    topic: str,
    max_videos: int = 10,
    languages: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full plan4me video research pipeline for a topic.

    LONG-RUNNING (often several minutes): searches YouTube, fetches transcripts
    (captions, or Deepgram when enabled), extracts knowledge atoms with the
    configured LLM provider, clusters them, and synthesizes a cited Markdown
    knowledge guide.

    Args:
        topic: Research topic or search query (e.g. "B2B SaaS pricing interviews").
        max_videos: How many videos to process (1–50). Higher = slower and costlier.
        languages: Optional comma-separated transcript language codes (e.g. "en,hi").
                   Defaults to TRANSCRIPT_LANGUAGES from env.

    Returns markdown report plus counts (videos, transcripts, atoms, clusters).
    Prefer search_videos first to validate the query, then call this once.
    """
    topic = topic.strip()
    if not topic:
        return {"error": "topic is required"}

    max_videos = max(1, min(max_videos, 50))
    lang_list: list[str] | None = None
    if languages and languages.strip():
        lang_list = [c.strip() for c in languages.split(",") if c.strip()]

    init_db()
    logger.info("research_topic start topic=%r max_videos=%s", topic, max_videos)

    state = run_pipeline(topic, max_videos, lang_list)

    atoms = state.get("atoms", [])
    save_atoms(atoms)

    clusters = state.get("clusters", [])
    insights = [cluster_to_insight(c) for c in clusters[:80]]

    result = KnowledgeReport(
        topic=topic,
        videos_processed=len(state.get("videos", [])),
        transcripts_found=len(state.get("transcripts", [])),
        atoms_extracted=len(atoms),
        clusters=len(clusters),
        markdown=state.get("report_markdown", ""),
        insights=insights,
    )
    save_report(result)

    logger.info(
        "research_topic done topic=%r transcripts=%s atoms=%s clusters=%s",
        topic,
        result.transcripts_found,
        result.atoms_extracted,
        result.clusters,
    )

    return {
        "topic": result.topic,
        "videos_processed": result.videos_processed,
        "transcripts_found": result.transcripts_found,
        "atoms_extracted": result.atoms_extracted,
        "clusters": result.clusters,
        "markdown": result.markdown,
        "insights": [i.model_dump() for i in insights[:20]],
    }


@mcp.tool()
def get_latest_report() -> dict[str, Any]:
    """Return the most recently saved research report from local SQLite, if any.

    Use after research_topic or to recover the last report without re-running
    the pipeline.
    """
    init_db()
    row = store_latest()
    if not row:
        return {"found": False, "message": "No reports saved yet."}
    meta = row.get("meta") or {}
    return {
        "found": True,
        "topic": row["topic"],
        "videos_processed": meta.get("videos_processed", 0),
        "transcripts_found": meta.get("transcripts_found", 0),
        "atoms_extracted": meta.get("atoms_extracted", 0),
        "clusters": meta.get("clusters", 0),
        "markdown": row["markdown"],
        "created_at": row.get("created_at"),
    }


def main() -> None:
    init_db()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
