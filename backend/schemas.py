"""Pydantic data models shared across the pipeline.

The `KnowledgeAtom` is the heart of the system: instead of prose summaries we
extract atomic, traceable units of knowledge that can be deduplicated,
counted, and cited back to a specific video + timestamp.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class AtomType(str, Enum):
    advice = "advice"
    example = "example"
    failure_mode = "failure_mode"
    tool = "tool"
    resource = "resource"
    framework = "framework"
    claim = "claim"


class VideoMeta(BaseModel):
    """Basic metadata returned from a YouTube search (no download required)."""

    video_id: str
    title: str
    url: str
    channel: Optional[str] = None
    duration: Optional[int] = None  # seconds
    view_count: Optional[int] = None


class TranscriptSnippet(BaseModel):
    text: str
    start: float  # seconds
    duration: Optional[float] = None
    speaker: Optional[str] = None  # e.g. "SPEAKER_0" from Deepgram diarization


class Transcript(BaseModel):
    video_id: str
    title: str
    channel: Optional[str] = None
    language: Optional[str] = None
    source: str = "captions"  # "captions" | "whisper"
    snippets: list[TranscriptSnippet] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.snippets)


class RawAtom(BaseModel):
    """What the extraction LLM returns per item (no provenance yet).

    Fields have safe defaults so a single malformed atom from the model does
    not fail validation for the entire batch. Atoms with an empty `claim` are
    filtered out downstream.
    """

    type: AtomType = AtomType.claim
    claim: str = Field(
        default="", description="A single, self-contained statement of the insight."
    )
    actionable_step: Optional[str] = Field(
        default=None, description="Concrete action a learner can take, if applicable."
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How strongly the speaker asserts this (0-1).",
    )
    quote: Optional[str] = Field(
        default=None, description="Short verbatim quote supporting the claim, if notable."
    )
    approx_timestamp: Optional[float] = Field(
        default=None, description="Approx start time in seconds within the transcript."
    )

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> Any:
        """Models sometimes invent categories (e.g. 'niche'); map unknowns to 'claim'."""
        if isinstance(v, str):
            valid = {t.value for t in AtomType}
            if v.lower() not in valid:
                return AtomType.claim
            return v.lower()
        return v


class AtomExtraction(BaseModel):
    """Structured-output container the LLM fills for one transcript chunk."""

    atoms: list[RawAtom] = Field(default_factory=list)

    @field_validator("atoms", mode="before")
    @classmethod
    def _parse_atoms(cls, v: Any) -> Any:
        """Some models return the list as a JSON string; parse it defensively."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
            if isinstance(parsed, dict) and "atoms" in parsed:
                return parsed["atoms"]
            return parsed
        return v


class KnowledgeAtom(BaseModel):
    """A RawAtom enriched with full provenance, stored and clustered."""

    id: str
    topic: str
    type: AtomType
    claim: str
    actionable_step: Optional[str] = None
    confidence: float
    quote: Optional[str] = None
    video_id: str
    video_title: str
    channel: Optional[str] = None
    timestamp: Optional[float] = None
    cluster_id: Optional[int] = None


class AtomCluster(BaseModel):
    cluster_id: int
    representative_claim: str
    type: AtomType
    support_count: int  # how many distinct videos/speakers said something similar
    atoms: list[KnowledgeAtom]


class ClusterInsight(BaseModel):
    """A single card for the UI: one deduplicated insight with its support."""

    type: AtomType
    claim: str
    support_count: int
    actionable_step: Optional[str] = None
    quote: Optional[str] = None
    confidence: float
    sources: list[str] = Field(default_factory=list)  # distinct video titles


class ReportRequest(BaseModel):
    topic: str
    max_videos: int = Field(default=10, ge=1, le=50)
    languages: Optional[list[str]] = None

    @field_validator("max_videos", mode="before")
    @classmethod
    def _max_videos_bounds(cls, v: Any) -> Any:
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 50:
            # Floats and numeric strings still fall through to Field(ge, le)
            # after coercion. Integers outside 1..50 must use this exact text.
            if isinstance(v, int) and not isinstance(v, bool):
                raise ValueError("max_videos must be 1 to 50")
        return v


class TopicItem(BaseModel):
    name: str = ""
    what_it_is: str = ""


class ClaimItem(BaseModel):
    text: str = ""
    timestamp_seconds: float = 0
    quote: str = ""
    label: str = "single-source"

    @field_validator("label", mode="before")
    @classmethod
    def _claim_label(cls, v: Any) -> str:
        allowed = {"self-reported", "estimate", "heuristic", "single-source"}
        if v in allowed:
            return str(v)
        return "single-source"

    @field_validator("quote", mode="before")
    @classmethod
    def _clip_quote(cls, v: Any) -> str:
        text = "" if v is None else str(v)
        words = text.split()
        if len(words) <= 25:
            return text
        return " ".join(words[:25])


class ProcessStep(BaseModel):
    text: str = ""
    timestamp_seconds: float = 0


class ProcessItem(BaseModel):
    name: str = ""
    steps: list[ProcessStep] = Field(default_factory=list)


class ChecklistItem(BaseModel):
    text: str = ""
    timestamp_seconds: float = 0


class BookMention(BaseModel):
    title: str = ""
    author: Optional[str] = None


class Disagreement(BaseModel):
    side_a: str = ""
    side_b: str = ""


class NumberItem(BaseModel):
    figure: str = ""
    label: str = "heuristic"
    timestamp_seconds: float = 0
    context: str = ""

    @field_validator("figure", mode="before")
    @classmethod
    def _figure_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("label", mode="before")
    @classmethod
    def _number_label(cls, v: Any) -> str:
        allowed = {"self-reported", "estimate", "heuristic"}
        if v in allowed:
            return str(v)
        return "heuristic"


class VideoBrief(BaseModel):
    video_id: str = ""
    title: str = ""
    channel: Optional[str] = None
    topics: list[TopicItem] = Field(default_factory=list)
    claims: list[ClaimItem] = Field(default_factory=list)
    processes: list[ProcessItem] = Field(default_factory=list)
    checklist: list[ChecklistItem] = Field(default_factory=list)
    books: list[BookMention] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    numbers: list[NumberItem] = Field(default_factory=list)
    error: Optional[str] = None


class AgendaBook(BaseModel):
    title: str = ""
    author: Optional[str] = None
    video_id: str = ""


class AgendaConcept(BaseModel):
    name: str = ""
    what_videos_agree: str = ""
    video_ids: list[str] = Field(default_factory=list)
    support_count: int = 0
    label: str = "single-source"
    contested: bool = False
    contest: str = ""
    operable_question: str = ""
    books: list[AgendaBook] = Field(default_factory=list)
    deep_research: bool = False

    @field_validator("label", mode="before")
    @classmethod
    def _agenda_label(cls, v: Any) -> str:
        if v in {"multi-source", "single-source"}:
            return str(v)
        return "single-source"


class Agenda(BaseModel):
    topic: str = ""
    concepts: list[AgendaConcept] = Field(default_factory=list)


class SummaryRecord(BaseModel):
    url: str
    author: str = ""
    word_count: int = 0
    fetched_at: str = ""


class SubConcept(BaseModel):
    name: str = ""
    agreed_definition: str = ""
    summary_urls: list[str] = Field(default_factory=list)
    disagreement: str = "no disagreement found"
    operable_question: str = ""
    label: str = "summary-derived"


class BookMap(BaseModel):
    title: str
    author: Optional[str] = None
    access: str = "access-required"
    summaries: list[SummaryRecord] = Field(default_factory=list)
    subconcepts: list[SubConcept] = Field(default_factory=list)


class PageRecord(BaseModel):
    url: str
    title: str = ""
    fetched_at: str = ""
    word_count: int = 0
    kept: bool = False
    rejected_reason: str = ""


class ConceptNote(BaseModel):
    concept: str
    slug: str
    book_maps: list[BookMap] = Field(default_factory=list)
    pages: list[PageRecord] = Field(default_factory=list)
    markdown: str = ""


class WebHit(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""


class FetchedPage(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    word_count: int = 0
    error: Optional[str] = None


class KnowledgeReport(BaseModel):
    topic: str
    videos_processed: int
    transcripts_found: int
    atoms_extracted: int
    clusters: int
    markdown: str
    insights: list[ClusterInsight] = Field(default_factory=list)
    candidates_searched: int = 0
    videos_skipped: int = 0
    concepts_deep_researched: int = 0
    job_id: str = ""


def cluster_to_insight(cluster: "AtomCluster") -> ClusterInsight:
    """Build a UI insight card from a cluster, picking the best action/quote."""
    action = next((a.actionable_step for a in cluster.atoms if a.actionable_step), None)
    quote = next((a.quote for a in cluster.atoms if a.quote), None)
    confidence = max((a.confidence for a in cluster.atoms), default=0.5)
    sources = sorted({a.video_title for a in cluster.atoms})
    return ClusterInsight(
        type=cluster.type,
        claim=cluster.representative_claim,
        support_count=cluster.support_count,
        actionable_step=action,
        quote=quote,
        confidence=confidence,
        sources=sources,
    )
