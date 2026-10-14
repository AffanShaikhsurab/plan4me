"""Final operating document. Provenance and missing headings are code, not the model."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.llm.chat import get_synthesis_llm
from backend.schemas import Agenda, ConceptNote, VideoBrief

logger = logging.getLogger(__name__)

FINAL_SYSTEM = (
    "You write one operating document from research notes. You do not add steps, "
    "numbers, books, or tools that are not in the notes or the agenda.\n"
    "\n"
    "Organize by what a person does, in order. Do not organize by video or by book.\n"
    "\n"
    "A required step must be multi-source in the notes. A single-source point goes "
    "in the appendix, not in the required steps.\n"
    "\n"
    "Where notes disagree, keep both sides in Disagreements. Do not average numbers.\n"
    "\n"
    "Every number you mention appears again in the Numbers table with its label.\n"
    "\n"
    'If a section has nothing, write the sentence "Nothing in the sources opened." '
    "Do not delete the heading."
)

FINAL_HEADINGS = [
    "## How this document was made",
    "## What to do",
    "## What not to do",
    "## Disagreements",
    "## Numbers",
    "## Concepts researched",
    "## Single-source appendix",
    "## Books named but not read",
    "## Not researched this run",
    "## Not established",
]

NOTHING = "Nothing in the sources opened."


def ensure_headings(markdown: str) -> str:
    """Insert any required heading the model omitted, in plan order."""
    text = markdown.strip()
    for heading in FINAL_HEADINGS:
        if _has_heading(text, heading):
            continue
        block = f"{heading}\n\n{NOTHING}"
        text = _insert_block(text, heading, block)
    return text.strip() + "\n"


def write_final_document(
    topic: str,
    agenda: Agenda,
    notes: list[ConceptNote],
    briefs: list[VideoBrief],
    *,
    job_id: str,
    candidates_searched: int,
    transcripts_used: int,
    skipped_ids: list[str],
    opened_on: str | None = None,
) -> str:
    """Synthesize one operating document and pin the factual sections in code."""
    date = opened_on or datetime.now(timezone.utc).date().isoformat()
    model_text = _invoke(topic, agenda, notes, briefs)
    model_text = _strip_section(model_text, "## How this document was made")
    provenance = _provenance(
        topic=topic,
        job_id=job_id,
        date=date,
        candidates_searched=candidates_searched,
        transcripts_used=transcripts_used,
        skipped_ids=skipped_ids,
        deep_names=[concept.name for concept in agenda.concepts if concept.deep_research],
    )
    document = provenance.rstrip() + "\n\n" + model_text.strip() + "\n"
    document = _replace_section(document, "## Concepts researched", _concepts_body(agenda))
    document = _replace_section(document, "## Single-source appendix", _appendix_body(agenda))
    document = _replace_section(document, "## Books named but not read", _books_body(agenda, notes))
    document = _replace_section(document, "## Not researched this run", _not_researched_body(agenda))
    if not any(concept.deep_research for concept in agenda.concepts):
        document = _replace_section(document, "## What to do", NOTHING)
        document = _replace_section(document, "## What not to do", NOTHING)
    else:
        allowed_urls, allowed_videos = _locators(notes, briefs)
        document = _filter_steps(document, "## What to do", allowed_urls, allowed_videos)
        document = _filter_steps(document, "## What not to do", allowed_urls, allowed_videos)
    return ensure_headings(document)


def _invoke(topic: str, agenda: Agenda, notes: list[ConceptNote], briefs: list[VideoBrief]) -> str:
    disagreements = []
    numbers = []
    for brief in briefs:
        if brief.error:
            continue
        disagreements.append(
            {
                "video_id": brief.video_id,
                "disagreements": [item.model_dump() for item in brief.disagreements],
                "numbers": [item.model_dump() for item in brief.numbers],
            }
        )
        numbers.extend(item.figure for item in brief.numbers)
    human = (
        f"TOPIC: {topic}\n\n"
        "AGENDA:\n"
        f"{agenda.model_dump_json(indent=2)}\n\n"
        "CONCEPT NOTES:\n"
        + "\n\n".join(note.markdown for note in notes)
        + "\n\nBRIEF DISAGREEMENTS AND NUMBERS:\n"
        f"{disagreements}\n"
    )
    try:
        llm = get_synthesis_llm()
        response = llm.invoke([("system", FINAL_SYSTEM), ("human", human)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("final document model failed: %s", exc)
        return ""
    return _response_text(response)


def _provenance(
    *,
    topic: str,
    job_id: str,
    date: str,
    candidates_searched: int,
    transcripts_used: int,
    skipped_ids: list[str],
    deep_names: list[str],
) -> str:
    skipped = ", ".join(skipped_ids) if skipped_ids else "none"
    researched = ", ".join(deep_names) if deep_names else "none"
    lines = [
        "## How this document was made",
        "",
        f"Topic: {topic}",
        f"Job id: {job_id}",
        f"Date: {date}",
        f"Candidates searched: {candidates_searched}",
        f"Transcripts used: {transcripts_used}",
        f"Video ids skipped: {skipped}",
        f"Concepts that received a deep pass: {researched}",
        "",
        "Video figures are self-reported or heuristic unless a row in Numbers says otherwise.",
    ]
    if not deep_names:
        lines.append("No concept was multi-source.")
    return "\n".join(lines)


def _concepts_body(agenda: Agenda) -> str:
    names = [concept.name for concept in agenda.concepts if concept.deep_research]
    if not names:
        return NOTHING
    return "\n".join(f"- {name}" for name in names)


def _appendix_body(agenda: Agenda) -> str:
    lines: list[str] = []
    for concept in agenda.concepts:
        if concept.label != "single-source":
            continue
        video_id = concept.video_ids[0] if concept.video_ids else ""
        lines.append(f"- {concept.name} ({video_id}) — single-source")
    if not lines:
        return NOTHING
    return "\n".join(lines)


def _books_body(agenda: Agenda, notes: list[ConceptNote]) -> str:
    access_by_title: dict[str, str] = {}
    author_by_title: dict[str, str | None] = {}
    for concept in agenda.concepts:
        for book in concept.books:
            author_by_title.setdefault(book.title, book.author)
            access_by_title.setdefault(book.title, "access-required")
    for note in notes:
        for book in note.book_maps:
            access_by_title[book.title] = book.access
            if book.author:
                author_by_title[book.title] = book.author
    if not access_by_title:
        return NOTHING
    lines = []
    for title in sorted(access_by_title):
        author = author_by_title.get(title) or "author unknown"
        lines.append(f"- {title} — {author} — {access_by_title[title]}")
    return "\n".join(lines)


def _not_researched_body(agenda: Agenda) -> str:
    skipped = [
        concept.name
        for concept in agenda.concepts
        if concept.label == "multi-source" and not concept.deep_research
    ]
    if not skipped:
        return NOTHING
    return "\n".join(f"- {name}" for name in skipped)


def _locators(notes: list[ConceptNote], briefs: list[VideoBrief]) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    for note in notes:
        for page in note.pages:
            if page.kept and page.url:
                urls.add(page.url)
    videos = {brief.video_id for brief in briefs if brief.video_id and not brief.error}
    return urls, videos


def _filter_steps(markdown: str, heading: str, urls: set[str], videos: set[str]) -> str:
    body = _section_body(markdown, heading)
    if body is None:
        return markdown
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("##", "|")) or set(stripped) <= {"-", " "}:
            kept.append(line)
            continue
        if not _line_is_located(stripped, urls, videos):
            continue
        kept.append(line)
    replacement = "\n".join(kept).strip() or NOTHING
    return _replace_section(markdown, heading, replacement)


def _line_is_located(line: str, urls: set[str], videos: set[str]) -> bool:
    if any(url and url in line for url in urls):
        return True
    found = [video_id for video_id in videos if video_id and video_id in line]
    return len(found) >= 2


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content or "")


def _has_heading(markdown: str, heading: str) -> bool:
    return any(line.strip() == heading for line in markdown.splitlines())


def _insert_block(markdown: str, heading: str, block: str) -> str:
    index = FINAL_HEADINGS.index(heading)
    later = set(FINAL_HEADINGS[index + 1 :])
    lines = markdown.splitlines()
    for line_index, line in enumerate(lines):
        if line.strip() in later:
            prefix = "\n".join(lines[:line_index]).rstrip()
            suffix = "\n".join(lines[line_index:])
            joiner = "\n\n" if prefix else ""
            return f"{prefix}{joiner}{block}\n\n{suffix}"
    prefix = markdown.rstrip()
    joiner = "\n\n" if prefix else ""
    return f"{prefix}{joiner}{block}"


def _heading_span(markdown: str, heading: str) -> tuple[int, int] | None:
    lines = markdown.splitlines(keepends=True)
    start = None
    offset = 0
    for line in lines:
        if start is None and line.strip() == heading:
            start = offset
        elif start is not None and line.startswith("## "):
            return start, offset
        offset += len(line)
    if start is None:
        return None
    return start, len(markdown)


def _section_body(markdown: str, heading: str) -> str | None:
    span = _heading_span(markdown, heading)
    if span is None:
        return None
    start, end = span
    block = markdown[start:end]
    lines = block.splitlines()
    if lines and lines[0].strip() == heading:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _replace_section(markdown: str, heading: str, body: str) -> str:
    span = _heading_span(markdown, heading)
    block = f"{heading}\n\n{body.strip()}\n"
    if span is None:
        return markdown.rstrip() + "\n\n" + block
    start, end = span
    suffix = markdown[end:]
    prefix = markdown[:start].rstrip()
    joiner = "\n\n" if prefix else ""
    tail = "\n" + suffix if suffix and not suffix.startswith("\n") else suffix
    return f"{prefix}{joiner}{block}{tail}"


def _strip_section(markdown: str, heading: str) -> str:
    span = _heading_span(markdown, heading)
    if span is None:
        return markdown
    start, end = span
    return (markdown[:start] + markdown[end:]).strip()
