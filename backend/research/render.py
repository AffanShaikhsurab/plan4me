"""Markdown renderers for caption briefs and concept notes."""
from __future__ import annotations

from backend.schemas import (
    AgendaConcept,
    BookMap,
    PageRecord,
    SubConcept,
    VideoBrief,
)


def _ts(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value)


def render_video_brief(brief: VideoBrief) -> str:
    """Render a caption brief with the headings in the pipeline plan."""
    lines = [
        f"# {brief.title}",
        "",
        f"Video: https://www.youtube.com/watch?v={brief.video_id}",
        f"Channel: {brief.channel or ''}",
        "This file is a caption brief. No web page was opened for it.",
        "",
        "## Topics",
    ]
    for topic in brief.topics:
        lines.append(f"- {topic.name}: {topic.what_it_is}")
    lines.extend(["", "## Claims"])
    for claim in brief.claims:
        lines.append(
            f"- {claim.text} ({brief.video_id} {_ts(claim.timestamp_seconds)}s, {claim.label})"
        )
    lines.extend(["", "## Processes"])
    for process in brief.processes:
        lines.append(f"- {process.name}")
        for step in process.steps:
            lines.append(f"  - {step.text} ({_ts(step.timestamp_seconds)}s)")
    lines.extend(["", "## Checklist"])
    for item in brief.checklist:
        lines.append(f"- {item.text} ({_ts(item.timestamp_seconds)}s)")
    lines.extend(["", "## Books named"])
    for book in brief.books:
        if book.author:
            lines.append(f"- {book.title} — {book.author}")
        else:
            lines.append(f"- {book.title}")
    lines.extend(["", "## Numbers"])
    for number in brief.numbers:
        lines.append(
            f"- {number.figure} ({number.label}) at {_ts(number.timestamp_seconds)}s: {number.context}"
        )
    lines.extend(["", "## Disagreements inside this video"])
    for item in brief.disagreements:
        lines.append(f"- {item.side_a} / {item.side_b}")
    lines.append("")
    return "\n".join(lines)


def render_concept_note(
    *,
    topic: str,
    concept: AgendaConcept,
    book_maps: list[BookMap],
    pages: list[PageRecord],
    steps: list[tuple[str, str]],
    donts: list[tuple[str, str]],
    disagreement: str,
    numbers: list[tuple[str, str, str, str]],
    subconcepts: list[SubConcept],
    not_established: str,
    opened_on: str,
) -> str:
    """Render one concept note. `steps` are (text, locator) pairs."""
    if any(book.access == "summaries-mapped" for book in book_maps):
        access = "summaries-mapped"
    else:
        access = "access-required"
    lines = [
        f"# {concept.name}",
        "",
        f"User topic: {topic}",
        f"This note: public sources opened on {opened_on}.",
        (
            f"Book map: {access}. The book was not read, unless access is a "
            "licensed file, which this pipeline does not have."
        ),
        "",
        "## What to do",
    ]
    for index, (text, locator) in enumerate(steps, start=1):
        lines.append(f"{index}. {text} ({locator})")
    lines.extend(["", "## What not to do"])
    for index, (text, locator) in enumerate(donts, start=1):
        lines.append(f"{index}. {text} ({locator})")
    lines.extend(["", "## Disagreement"])
    lines.append(disagreement or "no disagreement found in the sources opened.")
    lines.extend(
        [
            "",
            "## Numbers",
            "",
            "| figure | label | who | where |",
            "| --- | --- | --- | --- |",
        ]
    )
    for figure, label, who, where in numbers:
        lines.append(f"| {figure} | {label} | {who} | {where} |")
    lines.extend(["", "## Summary-derived, not yet operable"])
    for sub in subconcepts:
        lines.append(
            f"- {sub.name}: {sub.agreed_definition} ({sub.label})"
        )
    lines.extend(["", "## Pages opened"])
    for page in pages:
        state = "kept" if page.kept else f"not kept: {page.rejected_reason}"
        lines.append(f"- {page.url} — {state} ({page.word_count} words)")
    lines.extend(["", "## Not established", not_established, ""])
    return "\n".join(lines)
