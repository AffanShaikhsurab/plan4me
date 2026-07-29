"""In-memory research jobs so clients can follow pipeline stages live.

`POST /report` runs the pipeline synchronously, which leaves a browser waiting
minutes with no feedback. A job runs the same graph on a worker thread and
records which stage finished, letting the UI narrate progress while it works.

Jobs are process-local by design: the prototype runs one API instance, and a
finished report is already persisted in SQLite.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.pipeline.graph import stream_pipeline
from backend.schemas import AtomCluster, KnowledgeAtom, KnowledgeReport, cluster_to_insight
from backend.storage.store import save_atoms, save_report

logger = logging.getLogger(__name__)

# Stage order the UI can rely on, matching the LangGraph node names.
STAGE_ORDER = ("search", "transcribe", "extract", "cluster", "synthesize")

# Jobs are kept only long enough for a client to read the result.
_MAX_JOBS = 12


@dataclass
class ResearchJob:
    job_id: str
    topic: str
    max_videos: int
    status: str = "running"  # running | done | error
    stage: str = "search"  # stage currently being worked on
    completed_stages: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    report: Optional[KnowledgeReport] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "status": self.status,
            "stage": self.stage,
            "completed_stages": list(self.completed_stages),
            "stage_order": list(STAGE_ORDER),
            "counts": dict(self.counts),
            "elapsed_seconds": round(
                (self.finished_at or time.time()) - self.started_at, 1
            ),
            "report": self.report.model_dump() if self.report else None,
            "error": self.error,
        }


_jobs: dict[str, ResearchJob] = {}
_lock = threading.Lock()


def _next_stage(completed: list[str]) -> str:
    for stage in STAGE_ORDER:
        if stage not in completed:
            return stage
    return STAGE_ORDER[-1]


def _prune_locked() -> None:
    if len(_jobs) <= _MAX_JOBS:
        return
    finished = [j for j in _jobs.values() if j.status != "running"]
    finished.sort(key=lambda j: j.finished_at or j.started_at)
    for job in finished[: len(_jobs) - _MAX_JOBS]:
        _jobs.pop(job.job_id, None)


def _run(job: ResearchJob, languages: Optional[list[str]]) -> None:
    videos = transcripts = 0
    atoms: list[KnowledgeAtom] = []
    clusters: list[AtomCluster] = []
    markdown = ""

    try:
        for stage, update in stream_pipeline(job.topic, job.max_videos, languages):
            if "videos" in update:
                videos = len(update["videos"])
            if "transcripts" in update:
                transcripts = len(update["transcripts"])
            if "atoms" in update:
                atoms = update["atoms"]
            if "clusters" in update:
                clusters = update["clusters"]
            if "report_markdown" in update:
                markdown = update["report_markdown"]

            with _lock:
                if stage not in job.completed_stages:
                    job.completed_stages.append(stage)
                job.stage = _next_stage(job.completed_stages)
                job.counts = {
                    "videos": videos,
                    "transcripts": transcripts,
                    "atoms": len(atoms),
                    "clusters": len(clusters),
                }

        # Clustering annotates atoms with cluster ids, so persist after the run.
        save_atoms(atoms)

        report = KnowledgeReport(
            topic=job.topic,
            videos_processed=videos,
            transcripts_found=transcripts,
            atoms_extracted=len(atoms),
            clusters=len(clusters),
            markdown=markdown,
            insights=[cluster_to_insight(c) for c in clusters[:80]],
        )
        save_report(report)

        with _lock:
            job.report = report
            job.status = "done"
            job.stage = "done"
            job.completed_stages = list(STAGE_ORDER)
            job.finished_at = time.time()
    except Exception as exc:  # noqa: BLE001 - any pipeline failure belongs to the client
        logger.exception("research job %s failed", job.job_id)
        with _lock:
            job.status = "error"
            job.error = str(exc)[:400]
            job.finished_at = time.time()


def start_job(
    topic: str, max_videos: int, languages: Optional[list[str]] = None
) -> ResearchJob:
    job = ResearchJob(job_id=uuid.uuid4().hex[:12], topic=topic, max_videos=max_videos)
    with _lock:
        _jobs[job.job_id] = job
        _prune_locked()
    threading.Thread(
        target=_run, args=(job, languages), name=f"research-{job.job_id}", daemon=True
    ).start()
    return job


def get_job(job_id: str) -> Optional[ResearchJob]:
    with _lock:
        return _jobs.get(job_id)
