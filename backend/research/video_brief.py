"""Normal pass: one caption brief per video. No web search."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.llm.chat import get_extraction_llm, structured
from backend.research.render import render_video_brief
from backend.schemas import Transcript, VideoBrief

logger = logging.getLogger(__name__)

VIDEO_BRIEF_SYSTEM = (
    "You extract a research brief from one video transcript. You only use the "
    "transcript. You do not add advice from your training. You do not search.\n"
    "\n"
    "Return topics, claims, ordered processes, checklist items, named books, "
    "named people, named tools, disagreements inside this video, and numbers.\n"
    "\n"
    "A topic is a named idea a person could act on or argue with: a rule, a "
    "stage, a distinction, a failure mode, or a question. A metaphor that is "
    "only a metaphor is not a topic.\n"
    "\n"
    "Each claim has the seconds where the speaker says it. Copy at most 25 "
    "words into quote. Paraphrase the claim.\n"
    "\n"
    "A number has a label. Use self-reported when the speaker describes their "
    "own results. Use estimate when they say the number is rough. Use heuristic "
    "when it is a rule of thumb. Never use verified.\n"
    "\n"
    "If the speaker contradicts themselves, put both sides in disagreements. "
    "Do not pick a winner."
)

MERGE_SYSTEM = (
    "drop duplicates, keep every disagreement, keep every number, do not add "
    "a topic that was not in a partial brief."
)

_CHUNK_TRIGGER = 24000
_CHUNK_SIZE = 12000


def write_video_brief(topic: str, transcript: Transcript) -> VideoBrief:
    """Extract one brief from a transcript. Empty model output is an error brief."""
    try:
        chunks = _transcript_chunks(transcript)
        if len(chunks) == 1:
            brief = _invoke(topic, transcript, chunks[0])
        else:
            partials = [_invoke(topic, transcript, chunk) for chunk in chunks]
            partials = [brief for brief in partials if brief is not None]
            brief = _merge(topic, transcript, partials) if partials else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("video brief failed for %s: %s", transcript.video_id, exc)
        brief = None
    if brief is None or (not brief.topics and not brief.claims):
        return _error_brief(transcript)
    brief.video_id = transcript.video_id
    brief.title = transcript.title
    brief.channel = transcript.channel
    brief.error = None
    return brief


def save_video_brief(run_dir: Path, brief: VideoBrief) -> None:
    dest = run_dir / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{brief.video_id}.json").write_text(
        brief.model_dump_json(indent=2), encoding="utf-8"
    )
    (dest / f"{brief.video_id}.md").write_text(
        render_video_brief(brief), encoding="utf-8"
    )


def produce_video_brief(topic: str, transcript: Transcript, run_dir: Path) -> VideoBrief:
    brief = write_video_brief(topic, transcript)
    save_video_brief(run_dir, brief)
    return brief


def _invoke(topic: str, transcript: Transcript, body: str) -> VideoBrief | None:
    model = structured(get_extraction_llm(), VideoBrief)
    result = model.invoke(
        [("system", VIDEO_BRIEF_SYSTEM), ("human", _human(topic, transcript, body))]
    )
    return _as_brief(result, transcript)


def _merge(topic: str, transcript: Transcript, partials: list[VideoBrief]) -> VideoBrief | None:
    payload = [brief.model_dump() for brief in partials]
    human = (
        f"TOPIC: {topic}\n"
        f"VIDEO_ID: {transcript.video_id}\n\n"
        "PARTIAL BRIEFS:\n"
        f"{payload}"
    )
    model = structured(get_extraction_llm(), VideoBrief)
    result = model.invoke([("system", MERGE_SYSTEM), ("human", human)])
    return _as_brief(result, transcript)


def _as_brief(result: object, transcript: Transcript) -> VideoBrief | None:
    if result is None:
        return None
    if isinstance(result, VideoBrief):
        brief = result
    else:
        brief = VideoBrief.model_validate(result)
    brief.video_id = transcript.video_id
    brief.title = transcript.title
    brief.channel = transcript.channel
    return brief


def _human(topic: str, transcript: Transcript, body: str) -> str:
    return (
        f"TOPIC: {topic}\n"
        f"VIDEO_ID: {transcript.video_id}\n"
        f"TITLE: {transcript.title}\n"
        f"CHANNEL: {transcript.channel or ''}\n"
        "\n"
        "TRANSCRIPT:\n"
        f"{body}"
    )


def _transcript_chunks(transcript: Transcript) -> list[str]:
    formatted = [
        f"[{int(snip.start)}] {snip.text}" for snip in transcript.snippets
    ]
    whole = "\n".join(formatted)
    if len(transcript.full_text) <= _CHUNK_TRIGGER:
        return [whole]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in formatted:
        extra = len(line) + (1 if buf else 0)
        if buf and size + extra > _CHUNK_SIZE:
            chunks.append("\n".join(buf))
            buf = []
            size = 0
        buf.append(line)
        size += len(line) + (1 if len(buf) > 1 else 0)
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [whole]


def _error_brief(transcript: Transcript) -> VideoBrief:
    return VideoBrief(
        video_id=transcript.video_id,
        title=transcript.title,
        channel=transcript.channel,
        error="empty brief",
    )
