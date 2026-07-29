"""FastAPI application exposing the knowledge pipeline.

Endpoints:
    GET  /health                 - liveness
    GET  /health/bedrock         - verify Bedrock reachability (cheap call)
    POST /search                 - yt-dlp search only (no LLM, cheap sanity check)
    POST /report                 - run the full pipeline for a topic
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import get_settings
from backend.ingestion.search import search_videos
from backend.pipeline.graph import run_pipeline
from backend.schemas import (
    KnowledgeReport,
    ReportRequest,
    VideoMeta,
    cluster_to_insight,
)
from backend.storage.store import (
    get_latest_report,
    init_db,
    save_atoms,
    save_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plan4me")

app = FastAPI(title="plan4me", version="0.1.0")

# Allow the Next.js dev server (and local tooling) to call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "region": s.aws_region,
        "extraction_model": s.extraction_model_id,
        "synthesis_model": s.synthesis_model_id,
        "whisper_fallback": s.enable_whisper_fallback,
    }


@app.get("/health/bedrock")
def health_bedrock() -> dict:
    """Make a tiny Bedrock call to confirm credentials + model access."""
    from backend.llm.bedrock import get_extraction_llm
    try:
        resp = get_extraction_llm().invoke("Reply with the single word: ok")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"status": "ok", "model_reply": text.strip()[:50]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"bedrock unreachable: {exc}")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10


@app.post("/search", response_model=list[VideoMeta])
def search(req: SearchRequest) -> list[VideoMeta]:
    return search_videos(req.query, req.max_results)


@app.get("/reports/latest", response_model=KnowledgeReport | None)
def latest_report() -> KnowledgeReport | None:
    """Return the most recently generated report, if any."""
    row = get_latest_report()
    if not row:
        return None
    meta = row.get("meta", {})
    return KnowledgeReport(
        topic=row["topic"],
        videos_processed=meta.get("videos_processed", 0),
        transcripts_found=meta.get("transcripts_found", 0),
        atoms_extracted=meta.get("atoms_extracted", 0),
        clusters=meta.get("clusters", 0),
        markdown=row["markdown"],
        insights=[],
    )


@app.post("/report", response_model=KnowledgeReport)
def report(req: ReportRequest) -> KnowledgeReport:
    state = run_pipeline(req.topic, req.max_videos, req.languages)

    atoms = state.get("atoms", [])
    save_atoms(atoms)

    clusters = state.get("clusters", [])
    insights = [cluster_to_insight(c) for c in clusters[:80]]

    result = KnowledgeReport(
        topic=req.topic,
        videos_processed=len(state.get("videos", [])),
        transcripts_found=len(state.get("transcripts", [])),
        atoms_extracted=len(atoms),
        clusters=len(clusters),
        markdown=state.get("report_markdown", ""),
        insights=insights,
    )
    save_report(result)
    return result
