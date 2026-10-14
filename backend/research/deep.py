"""Deep pass: one concept at a time, books first, then public pages."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from backend.llm.chat import get_synthesis_llm, structured
from backend.research.render import render_concept_note
from backend.research.web import fetch_page, query_allowed, rejection_reason, search_web
from backend.schemas import (
    AgendaBook,
    AgendaConcept,
    BookMap,
    ConceptNote,
    FetchedPage,
    PageRecord,
    SubConcept,
    SummaryRecord,
    VideoBrief,
    WebHit,
)

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[WebHit]]
FetchFn = Callable[[str], FetchedPage]

_SUMMARY_VOICE = (
    "summary",
    "takeaway",
    "the author",
    "this chapter",
    "in short",
    "key point",
    "overview",
    "we learn",
)
_ACTION_MARKERS = (
    "step",
    "how to",
    "failure",
    "fail",
    "do not",
    "don't",
    "instead",
    "mistake",
    "limitation",
    "critique",
    "versus",
    "vs.",
    "difference",
    "checklist",
    "avoid",
)


class SourcedStep(BaseModel):
    text: str = ""
    url: str = ""


class NoteDraft(BaseModel):
    what_to_do: list[SourcedStep] = Field(default_factory=list)
    what_not_to_do: list[SourcedStep] = Field(default_factory=list)
    disagreement: str = ""


class SubConceptBatch(BaseModel):
    subconcepts: list[SubConcept] = Field(default_factory=list)


def book_access(summaries: list) -> str:
    """Three kept summaries map a book. Fewer than that leaves it unread."""
    if len(summaries) >= 3:
        return "summaries-mapped"
    return "access-required"


def concept_slug(name: str) -> str:
    hyphenated = re.sub(r"\s+", "-", name.lower())
    cleaned = re.sub(r"[^a-z0-9-]", "", hyphenated)[:60]
    return cleaned or "concept"


def assign_slugs(names: list[str]) -> list[str]:
    used: set[str] = set()
    slugs: list[str] = []
    for name in names:
        base = concept_slug(name)
        slug = base
        number = 2
        while slug in used:
            slug = f"{base}-{number}"
            number += 1
        used.add(slug)
        slugs.append(slug)
    return slugs


def research_concept(
    topic: str,
    concept: AgendaConcept,
    *,
    briefs: list[VideoBrief] | None = None,
    slug: str | None = None,
    search: SearchFn | None = None,
    fetch: FetchFn | None = None,
    opened_on: str | None = None,
) -> ConceptNote:
    """Run Pass A (named books) and Pass B (public pages) for one concept."""
    search_fn = search or search_web
    fetch_fn = fetch or fetch_page
    opened = opened_on or datetime.now(timezone.utc).date().isoformat()
    supporting = [
        brief
        for brief in (briefs or [])
        if brief.video_id in set(concept.video_ids) and not brief.error
    ]
    book_maps = [
        _map_book(book, search_fn, fetch_fn, opened)
        for book in _books_to_map(concept)
    ]
    pages, kept_pages = _pass_b(
        concept, topic, _first_person(supporting), search_fn, fetch_fn, opened
    )
    kept_urls = {page.url for page in kept_pages}
    steps: list[tuple[str, str]] = []
    donts: list[tuple[str, str]] = []
    disagreement = "no disagreement found in the sources opened."
    if concept.contested and concept.contest.strip():
        disagreement = concept.contest.strip()
    if kept_pages:
        draft = _draft_steps(topic, concept, kept_pages)
        steps = _kept_steps(draft.what_to_do if draft else [], kept_urls)
        donts = _kept_steps(draft.what_not_to_do if draft else [], kept_urls)
        if draft and draft.disagreement.strip() and not concept.contested:
            disagreement = draft.disagreement.strip()
    if not kept_pages:
        not_established = "no public page opened that passed the keep rules"
    else:
        not_established = ""
    subconcepts = [
        sub
        for book in book_maps
        if book.access == "summaries-mapped"
        for sub in book.subconcepts
    ]
    note_slug = slug or concept_slug(concept.name)
    markdown = render_concept_note(
        topic=topic,
        concept=concept,
        book_maps=book_maps,
        pages=pages,
        steps=steps,
        donts=donts,
        disagreement=disagreement,
        numbers=_numbers(concept, supporting),
        subconcepts=subconcepts,
        not_established=not_established,
        opened_on=opened,
    )
    return ConceptNote(
        concept=concept.name,
        slug=note_slug,
        book_maps=book_maps,
        pages=pages,
        markdown=markdown,
    )


def _books_to_map(concept: AgendaConcept) -> list[AgendaBook]:
    grouped: dict[str, dict] = {}
    for book in concept.books:
        if not book.title:
            continue
        slot = grouped.setdefault(
            book.title,
            {"title": book.title, "author": book.author, "n": 0, "video_id": book.video_id},
        )
        slot["n"] += 1
        if book.author and not slot["author"]:
            slot["author"] = book.author
    ranked = sorted(grouped.values(), key=lambda item: (-item["n"], item["title"]))
    return [
        AgendaBook(title=item["title"], author=item["author"], video_id=item["video_id"])
        for item in ranked[:2]
    ]


def _map_book(book: AgendaBook, search_fn: SearchFn, fetch_fn: FetchFn, opened: str) -> BookMap:
    kept: list[SummaryRecord] = []
    seen_titles: list[tuple[str, str]] = []
    for query in _pass_a_queries(book.title, book.author):
        if len(kept) >= 4:
            break
        if not query_allowed(query):
            continue
        try:
            hits = search_fn(query, 5)[:5]
        except Exception as exc:  # noqa: BLE001
            logger.warning("book search failed: %s", exc)
            continue
        for hit in hits:
            if len(kept) >= 4:
                break
            if rejection_reason(hit.url):
                continue
            page = fetch_fn(hit.url)
            reason = _summary_rejection(page, book.title, seen_titles)
            if reason:
                continue
            author = _byline(page.text) or (urlparse(page.url).hostname or "")
            kept.append(
                SummaryRecord(
                    url=page.url,
                    author=author,
                    word_count=page.word_count,
                    fetched_at=opened,
                )
            )
            seen_titles.append((page.url, page.title or hit.title))
    access = book_access(kept)
    subconcepts: list[SubConcept] = []
    if access == "summaries-mapped":
        subconcepts = _subconcepts(book, kept, search_fn, fetch_fn)
    return BookMap(
        title=book.title,
        author=book.author,
        access=access,
        summaries=kept if access == "summaries-mapped" else kept,
        subconcepts=subconcepts if access == "summaries-mapped" else [],
    )


def _pass_a_queries(title: str, author: str | None) -> list[str]:
    if author:
        first = f'"{title}" "{author}" detailed summary'
    else:
        first = f'"{title}" detailed summary'
    return [first, f'"{title}" chapter summary', f'"{title}" book notes']


def _pass_b(
    concept: AgendaConcept,
    topic: str,
    person: str | None,
    search_fn: SearchFn,
    fetch_fn: FetchFn,
    opened: str,
) -> tuple[list[PageRecord], list[FetchedPage]]:
    records: list[PageRecord] = []
    kept_pages: list[FetchedPage] = []
    seen: set[str] = set()
    seen_titles: list[tuple[str, str]] = []
    opened_count = 0
    for query in _pass_b_queries(concept, topic, person):
        if len(kept_pages) >= 4 or opened_count >= 8:
            break
        try:
            hits = search_fn(query, 5)[:5]
        except Exception as exc:  # noqa: BLE001
            logger.warning("concept search failed: %s", exc)
            continue
        for hit in hits:
            if len(kept_pages) >= 4 or opened_count >= 8:
                break
            if not hit.url or hit.url in seen:
                continue
            seen.add(hit.url)
            reason = rejection_reason(hit.url)
            if reason:
                records.append(
                    PageRecord(
                        url=hit.url,
                        title=hit.title,
                        fetched_at=opened,
                        word_count=0,
                        kept=False,
                        rejected_reason=reason,
                    )
                )
                continue
            opened_count += 1
            page = fetch_fn(hit.url)
            why = _public_rejection(page, seen_titles)
            kept = why == ""
            records.append(
                PageRecord(
                    url=hit.url,
                    title=page.title or hit.title,
                    fetched_at=opened,
                    word_count=page.word_count,
                    kept=kept,
                    rejected_reason="" if kept else why,
                )
            )
            if kept:
                kept_pages.append(page)
                seen_titles.append((page.url, page.title or hit.title))
    return records, kept_pages


def _pass_b_queries(concept: AgendaConcept, topic: str, person: str | None) -> list[str]:
    queries = [f"{concept.name} {topic}"]
    if person:
        queries.append(f"{concept.name} {person} talk")
    else:
        queries.append(f"{concept.name} talk")
    queries.append(f"{concept.name} limitation")
    queries.append(f"{concept.name} critique")
    return [
        query
        for query in queries
        if "summary of" not in query.lower() and query_allowed(query)
    ]


def _summary_rejection(
    page: FetchedPage, title: str, seen_titles: list[tuple[str, str]]
) -> str:
    if page.error:
        return page.error
    if page.word_count < 800:
        return "under 800 words"
    if title.lower() not in page.text.lower():
        return "title not named"
    if _looks_like_book_paste(page.text):
        return "looks like a book paste"
    if not (_byline(page.text) or urlparse(page.url).hostname):
        return "no author or site"
    if any(_same_article(page.url, page.title, url, other) for url, other in seen_titles):
        return "duplicate article"
    return ""


def _public_rejection(page: FetchedPage, seen_titles: list[tuple[str, str]]) -> str:
    if page.error:
        return page.error
    if page.word_count < 500:
        return "under 500 words"
    if not _actionable(page.text):
        return "too thin"
    if any(_same_article(page.url, page.title, url, other) for url, other in seen_titles):
        return "duplicate article"
    return ""


def _looks_like_book_paste(text: str) -> bool:
    run = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            run = 0
            continue
        lowered = stripped.lower()
        summary_voice = any(marker in lowered for marker in _SUMMARY_VOICE)
        if len(stripped) > 80 and not summary_voice:
            run += 1
            if run > 30:
                return True
        else:
            run = 0
    return False


def _actionable(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ACTION_MARKERS)


def _byline(text: str) -> str:
    match = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", text)
    if not match:
        return ""
    return match.group(1)


def _same_article(url_a: str, title_a: str, url_b: str, title_b: str) -> bool:
    host_a = (urlparse(url_a).hostname or "").lower()
    host_b = (urlparse(url_b).hostname or "").lower()
    if not host_a or host_a != host_b:
        return False
    prefix_a = (title_a or "")[:40].lower()
    prefix_b = (title_b or "")[:40].lower()
    return bool(prefix_a) and prefix_a == prefix_b


def _subconcepts(
    book: AgendaBook,
    summaries: list[SummaryRecord],
    search_fn: SearchFn,
    fetch_fn: FetchFn,
) -> list[SubConcept]:
    del search_fn  # summaries were already fetched; do not search again here
    excerpts = []
    for summary in summaries:
        page = fetch_fn(summary.url)
        excerpts.append(f"URL: {summary.url}\n{(page.text or '')[:4000]}")
    prompt = (
        f'BOOK: {book.title}\nAUTHOR: {book.author or ""}\n\n'
        "List sub-concepts that appear in at least two of these summaries. "
        "Set label to summary-derived. If they do not disagree, set disagreement "
        'to "no disagreement found". Do not invent a chapter that is not in the text.\n\n'
        + "\n\n".join(excerpts)
    )
    try:
        model = structured(get_synthesis_llm(), SubConceptBatch)
        result = model.invoke(
            [
                (
                    "system",
                    "You compare book summaries. You do not add a sub-concept that "
                    "fewer than two summaries support.",
                ),
                ("human", prompt),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("book sub-concept extraction failed: %s", exc)
        return []
    if result is None:
        return []
    batch = result if isinstance(result, SubConceptBatch) else SubConceptBatch.model_validate(result)
    known = {summary.url for summary in summaries}
    kept: list[SubConcept] = []
    for sub in batch.subconcepts:
        urls = [url for url in sub.summary_urls if url in known]
        if len(set(urls)) < 2:
            continue
        kept.append(
            SubConcept(
                name=sub.name,
                agreed_definition=sub.agreed_definition,
                summary_urls=urls,
                disagreement=sub.disagreement or "no disagreement found",
                operable_question=sub.operable_question,
                label="summary-derived",
            )
        )
    return kept


def _draft_steps(
    topic: str, concept: AgendaConcept, pages: list[FetchedPage]
) -> NoteDraft | None:
    excerpts = [
        f"URL: {page.url}\nTITLE: {page.title}\n{(page.text or '')[:4000]}"
        for page in pages
    ]
    prompt = (
        f"TOPIC: {topic}\nCONCEPT: {concept.name}\n\n"
        "PAGES:\n"
        + "\n\n".join(excerpts)
    )
    try:
        model = structured(get_synthesis_llm(), NoteDraft)
        result = model.invoke(
            [
                (
                    "system",
                    "You list steps that the opened pages describe. You do not add "
                    "a step the pages do not describe. Every step url must be copied "
                    "from the pages. If the pages do not describe a step, return an "
                    "empty list.",
                ),
                ("human", prompt),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("concept draft failed for %s: %s", concept.name, exc)
        return None
    if result is None:
        return None
    if isinstance(result, NoteDraft):
        return result
    return NoteDraft.model_validate(result)


def _kept_steps(steps: list[SourcedStep], urls: set[str]) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for step in steps:
        if step.url in urls and step.text.strip():
            kept.append((step.text.strip(), step.url))
    return kept


def _first_person(briefs: list[VideoBrief]) -> str | None:
    for brief in briefs:
        for person in brief.people:
            if person.strip():
                return person.strip()
    return None


def _numbers(
    concept: AgendaConcept, briefs: list[VideoBrief]
) -> list[tuple[str, str, str, str]]:
    words = set(re.findall(r"[a-z0-9]{4,}", concept.name.lower()))
    rows: list[tuple[str, str, str, str]] = []
    for brief in briefs:
        who = brief.channel or brief.video_id
        for number in brief.numbers:
            haystack = f"{number.context} {number.figure}".lower()
            if words and not any(word in haystack for word in words):
                continue
            where = f"video {brief.video_id} at {int(number.timestamp_seconds)}s"
            rows.append((number.figure, number.label, who, where))
    return rows
