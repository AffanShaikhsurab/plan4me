"""Run the two-depth research pipeline and persist each stage under runs/."""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.search import search_videos
from backend.ingestion.selection import select_and_transcribe
from backend.research.agenda import build_agenda
from backend.research.deep import assign_slugs, research_concept
from backend.research.final_doc import write_final_document
from backend.research.video_brief import produce_video_brief, save_video_brief
from backend.research.web import CURRENT_RUN_DIR
from backend.schemas import (
    Agenda,
    ConceptNote,
    KnowledgeReport,
    Transcript,
    VideoBrief,
    VideoMeta,
)
from backend.storage.store import init_db, save_report

logger = logging.getLogger(__name__)


class ResearchError(RuntimeError):
    """A research run stopped with a message the API can show as-is."""


def default_runs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "runs"


def run_research(
    topic: str,
    max_videos: int | None = None,
    languages: list[str] | None = None,
    job_id: str | None = None,
    resume_job_id: str | None = None,
) -> KnowledgeReport:
    """Run every stage and return the saved report."""
    report: KnowledgeReport | None = None
    for _stage, counts in stream_research(
        topic, max_videos, languages, job_id, resume_job_id=resume_job_id
    ):
        found = counts.get("report")
        if isinstance(found, KnowledgeReport):
            report = found
    if report is None:
        raise ResearchError("research produced no report")
    return report


def stream_research(
    topic: str,
    max_videos: int | None = None,
    languages: list[str] | None = None,
    job_id: str | None = None,
    resume_job_id: str | None = None,
) -> Iterator[tuple[str, dict]]:
    """Yield `(stage_name, counts)` after each stage's files are on disk."""
    active_id = resume_job_id or job_id or uuid.uuid4().hex[:12]
    if max_videos is None:
        max_videos = get_settings().default_max_videos
    if isinstance(max_videos, bool) or not isinstance(max_videos, int) or not 1 <= max_videos <= 50:
        raise ResearchError("max_videos must be 1 to 50")

    run_dir = default_runs_root() / active_id
    run_dir.mkdir(parents=True, exist_ok=True)
    token = CURRENT_RUN_DIR.set(run_dir)
    try:
        yield from _run_stages(topic, max_videos, languages, active_id, run_dir)
    finally:
        CURRENT_RUN_DIR.reset(token)


def _run_stages(
    topic: str,
    max_videos: int,
    languages: list[str] | None,
    job_id: str,
    run_dir: Path,
) -> Iterator[tuple[str, dict]]:
    videos = _search(topic, run_dir)
    yield "search", _counts(videos, [], [], None, [])

    skipped, transcripts = _transcribe(videos, languages, max_videos, run_dir)
    if not transcripts:
        raise ResearchError("no captions found for this topic")
    yield "transcribe", _counts(videos, transcripts, [], None, skipped)

    briefs = _briefs(topic, transcripts, run_dir)
    usable = [brief for brief in briefs if not brief.error and (brief.topics or brief.claims)]
    if not usable:
        raise ResearchError("every video brief failed")
    yield "video_briefs", _counts(videos, transcripts, briefs, None, skipped)

    agenda = _agenda(topic, usable, run_dir)
    yield "merge_agenda", _counts(videos, transcripts, briefs, agenda, skipped)

    notes = _deep(topic, agenda, usable, run_dir)
    yield "deep_research", _counts(videos, transcripts, briefs, agenda, skipped, notes)

    report = _final(
        topic, agenda, notes, usable, job_id, videos, transcripts, skipped, run_dir
    )
    counts = _counts(videos, transcripts, briefs, agenda, skipped, notes)
    counts["report"] = report
    yield "final_doc", counts


def _search(topic: str, run_dir: Path) -> list[VideoMeta]:
    path = run_dir / "search.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [VideoMeta.model_validate(item) for item in payload]
    videos = search_videos(topic, get_settings().num_candidates)
    path.write_text(
        json.dumps([video.model_dump() for video in videos], indent=2),
        encoding="utf-8",
    )
    return videos


def _transcribe(
    videos: list[VideoMeta],
    languages: list[str] | None,
    max_videos: int,
    run_dir: Path,
) -> tuple[list[dict], list[Transcript]]:
    skipped_path = run_dir / "skipped.json"
    transcript_dir = run_dir / "transcripts"
    if skipped_path.exists():
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
        transcripts = []
        if transcript_dir.exists():
            for path in sorted(transcript_dir.glob("*.json")):
                transcripts.append(
                    Transcript.model_validate_json(path.read_text(encoding="utf-8"))
                )
        return skipped, transcripts

    skipped: list[dict] = []
    transcripts = select_and_transcribe(
        videos, languages, target=max_videos, skipped=skipped
    )
    transcript_dir.mkdir(parents=True, exist_ok=True)
    for transcript in transcripts:
        (transcript_dir / f"{transcript.video_id}.json").write_text(
            transcript.model_dump_json(indent=2), encoding="utf-8"
        )
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    return skipped, transcripts


def _briefs(topic: str, transcripts: list[Transcript], run_dir: Path) -> list[VideoBrief]:
    pending: list[Transcript] = []
    by_id: dict[str, VideoBrief] = {}
    for transcript in transcripts:
        path = run_dir / "videos" / f"{transcript.video_id}.json"
        if path.exists():
            by_id[transcript.video_id] = VideoBrief.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        else:
            pending.append(transcript)
    if pending:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(produce_video_brief, topic, transcript, run_dir): transcript
                for transcript in pending
            }
            for future in as_completed(futures):
                transcript = futures[future]
                try:
                    by_id[transcript.video_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("brief failed for %s: %s", transcript.video_id, exc)
                    brief = VideoBrief(
                        video_id=transcript.video_id,
                        title=transcript.title,
                        channel=transcript.channel,
                        error="empty brief",
                    )
                    save_video_brief(run_dir, brief)
                    by_id[transcript.video_id] = brief
    return [by_id[transcript.video_id] for transcript in transcripts if transcript.video_id in by_id]


def _agenda(topic: str, briefs: list[VideoBrief], run_dir: Path) -> Agenda:
    path = run_dir / "agenda.json"
    if path.exists():
        return Agenda.model_validate_json(path.read_text(encoding="utf-8"))
    agenda = build_agenda(topic, briefs)
    path.write_text(agenda.model_dump_json(indent=2), encoding="utf-8")
    return agenda


def _deep(
    topic: str,
    agenda: Agenda,
    briefs: list[VideoBrief],
    run_dir: Path,
) -> list[ConceptNote]:
    selected = [concept for concept in agenda.concepts if concept.deep_research]
    slugs = assign_slugs([concept.name for concept in selected])
    notes: list[ConceptNote] = []
    dest = run_dir / "concepts"
    for concept, slug in zip(selected, slugs):
        path = dest / f"{slug}.json"
        if path.exists():
            notes.append(ConceptNote.model_validate_json(path.read_text(encoding="utf-8")))
            continue
        note = research_concept(topic, concept, briefs=briefs, slug=slug)
        dest.mkdir(parents=True, exist_ok=True)
        path.write_text(note.model_dump_json(indent=2), encoding="utf-8")
        (dest / f"{slug}.md").write_text(note.markdown, encoding="utf-8")
        notes.append(note)
    return notes


def _final(
    topic: str,
    agenda: Agenda,
    notes: list[ConceptNote],
    briefs: list[VideoBrief],
    job_id: str,
    videos: list[VideoMeta],
    transcripts: list[Transcript],
    skipped: list[dict],
    run_dir: Path,
) -> KnowledgeReport:
    skipped_ids = [str(item.get("video_id", "")) for item in skipped if item.get("video_id")]
    markdown = write_final_document(
        topic,
        agenda,
        notes,
        briefs,
        job_id=job_id,
        candidates_searched=len(videos),
        transcripts_used=len(transcripts),
        skipped_ids=skipped_ids,
    )
    (run_dir / "FINAL.md").write_text(markdown, encoding="utf-8")
    claim_count = sum(len(brief.claims) for brief in briefs if not brief.error)
    report = KnowledgeReport(
        topic=topic,
        videos_processed=len(transcripts),
        transcripts_found=len(transcripts),
        atoms_extracted=claim_count,
        clusters=len(agenda.concepts),
        markdown=markdown,
        insights=[],
        candidates_searched=len(videos),
        videos_skipped=len(skipped),
        concepts_deep_researched=len(notes),
        job_id=job_id,
    )
    init_db()
    save_report(report)
    return report


def _counts(
    videos: list[VideoMeta],
    transcripts: list[Transcript],
    briefs: list[VideoBrief],
    agenda: Agenda | None,
    skipped: list[dict],
    notes: list[ConceptNote] | None = None,
) -> dict:
    claims = sum(len(brief.claims) for brief in briefs if not brief.error)
    return {
        "videos": len(videos),
        "transcripts": len(transcripts),
        "atoms": claims,
        "clusters": len(agenda.concepts) if agenda else 0,
        "candidates": len(videos),
        "skipped": len(skipped),
        "deep": len(notes or []),
    }
