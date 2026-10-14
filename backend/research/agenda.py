"""Merge video briefs into one agenda, then recompute support in code."""
from __future__ import annotations

import json
import logging

from backend.config import get_settings
from backend.llm.chat import get_extraction_llm, structured
from backend.schemas import Agenda, AgendaBook, AgendaConcept, VideoBrief

logger = logging.getLogger(__name__)

AGENDA_SYSTEM = (
    "You merge video briefs into one research agenda. You do not add topics "
    "that no brief contains. You do not search.\n"
    "\n"
    "Two topics are the same concept when a person following one would do the "
    "same action as a person following the other. Different names for that "
    "action are one concept. Use the shortest name that still says the action.\n"
    "\n"
    "support_count is the number of distinct video_id values that contain the "
    "concept. Set label to multi-source when support_count is 2 or more. Set "
    "label to single-source when support_count is 1.\n"
    "\n"
    "If two videos give different numbers or opposite actions for the same "
    "concept, set contested to true and write both sides. Do not average numbers.\n"
    "\n"
    "books are titles a speaker named. Copy the video_id that named each title. "
    "A book is not a source you read."
)


def apply_agenda_rules(
    raw: Agenda,
    briefs: list[VideoBrief],
    deep_max: int | None = None,
) -> Agenda:
    """Recompute support, labels, and which concepts receive a deep pass.

    The model may inflate support_count or cite a video that was not in the
    briefs. Those outputs are dropped or rewritten here.
    """
    brief_ids = {brief.video_id for brief in briefs}
    kept: list[AgendaConcept] = []
    for concept in raw.concepts:
        video_ids = list(dict.fromkeys(concept.video_ids))
        if not video_ids or not set(video_ids).issubset(brief_ids):
            continue
        support = len(set(video_ids))
        label = "multi-source" if support >= 2 else "single-source"
        allowed = set(video_ids)
        books = [
            AgendaBook(title=book.title, author=book.author, video_id=book.video_id)
            for book in concept.books
            if book.video_id in allowed and book.title
        ]
        kept.append(
            AgendaConcept(
                name=concept.name,
                what_videos_agree=concept.what_videos_agree,
                video_ids=video_ids,
                support_count=support,
                label=label,
                contested=concept.contested,
                contest=concept.contest if concept.contested else "",
                operable_question=concept.operable_question,
                books=books,
                deep_research=False,
            )
        )

    multi = [concept for concept in kept if concept.label == "multi-source"]
    single = [concept for concept in kept if concept.label != "multi-source"]
    multi.sort(key=lambda concept: (-concept.support_count, concept.name))
    single.sort(key=lambda concept: concept.name)
    cap = _deep_cap(deep_max)
    for index, concept in enumerate(multi):
        concept.deep_research = index < cap
    return Agenda(topic=raw.topic, concepts=multi + single)


def build_agenda(topic: str, briefs: list[VideoBrief]) -> Agenda:
    """Ask the extraction model to merge briefs, then apply the support rules."""
    payload = [_brief_view(brief) for brief in briefs if _successful(brief)]
    human = (
        f"TOPIC: {topic}\n\n"
        "BRIEFS:\n"
        f"{json.dumps(payload, indent=2)}"
    )
    model = structured(get_extraction_llm(), Agenda)
    try:
        raw = model.invoke([("system", AGENDA_SYSTEM), ("human", human)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("agenda merge failed: %s", exc)
        raw = None
    if raw is None:
        raw = Agenda(topic=topic, concepts=[])
    elif not isinstance(raw, Agenda):
        raw = Agenda.model_validate(raw)
    raw.topic = topic
    return apply_agenda_rules(raw, briefs)


def _deep_cap(deep_max: int | None) -> int:
    if deep_max is None:
        deep_max = get_settings().deep_concept_cap
    return max(1, min(int(deep_max), 12))


def _successful(brief: VideoBrief) -> bool:
    return not brief.error and bool(brief.topics or brief.claims)


def _brief_view(brief: VideoBrief) -> dict:
    return {
        "video_id": brief.video_id,
        "topics": [topic.model_dump() for topic in brief.topics],
        "claims": [claim.text for claim in brief.claims],
        "processes": [
            {"name": process.name, "steps": [step.text for step in process.steps]}
            for process in brief.processes
        ],
        "checklist": [item.text for item in brief.checklist],
        "books": [book.model_dump() for book in brief.books],
    }
