"""FastAPI application exposing the knowledge pipeline.

Endpoints:
    GET  /health                 - liveness
    GET  /health/llm             - verify the active chat provider (cheap call)
    GET  /health/bedrock         - alias of /health/llm (kept for older docs)
    POST /search                 - yt-dlp search only (no LLM, cheap sanity check)
    POST /report                 - run the full pipeline for a topic (blocking)
    POST /research               - start the pipeline as a job and return its id
    GET  /research/{job_id}      - poll stage progress and the finished report
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import get_settings
from backend.llm.chat import active_models
from backend.ingestion.search import search_videos
from backend.pipeline.graph import run_pipeline
from backend.pipeline.jobs import get_job, start_job
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
    """Liveness. Reports misconfiguration as data rather than failing.

    An unknown LLM_PROVIDER must not take down the liveness probe, so provider
    resolution is allowed to fail here and is surfaced in `status`/`llm_error`.
    """
    s = get_settings()
    try:
        models = active_models()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "degraded",
            "region": s.aws_region,
            "extraction_model": None,
            "synthesis_model": None,
            "whisper_fallback": s.enable_whisper_fallback,
            "llm_provider": s.llm_provider,
            "embedding_provider": s.embedding_provider,
            "llm_error": str(exc)[:300],
        }
    return {
        "status": "ok",
        # Kept flat for the existing frontend HealthInfo shape.
        "region": s.aws_region,
        "extraction_model": models["extraction_model"],
        "synthesis_model": models["synthesis_model"],
        "whisper_fallback": s.enable_whisper_fallback,
        "llm_provider": models["provider"],
        "embedding_provider": s.embedding_provider,
    }


@app.get("/health/llm")
def health_llm() -> dict:
    """Make a tiny chat call to confirm credentials + model access."""
    s = get_settings()
    # Provider resolution is inside the try: an unknown LLM_PROVIDER is a
    # configuration problem, which is a 503, not an unhandled 500.
    try:
        from backend.llm.chat import get_extraction_llm

        models = active_models()
        resp = get_extraction_llm().invoke("Reply with the single word: ok")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"status": "ok", **models, "model_reply": text.strip()[:50]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"{s.llm_provider} unreachable: {exc}",
        )


@app.get("/health/bedrock")
def health_bedrock() -> dict:
    """Back-compat alias; the provider is whatever LLM_PROVIDER selects."""
    return health_llm()


@app.get("/health/embeddings")
def health_embeddings() -> dict:
    """Confirm the embedding provider returns a usable vector."""
    from backend.llm.embeddings import describe, embed_text

    s = get_settings()
    try:
        vec = embed_text("healthcheck")
        return {"status": "ok", **describe(), "dims": len(vec)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"embeddings ({s.embedding_provider}) unreachable: {exc}",
        )


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


@app.post("/research")
def start_research(req: ReportRequest) -> dict:
    """Kick off the pipeline in the background and return the job id to poll."""
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="topic is required")
    job = start_job(topic, req.max_videos, req.languages)
    return job.as_dict()


@app.get("/research/{job_id}")
def research_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job.as_dict()


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
