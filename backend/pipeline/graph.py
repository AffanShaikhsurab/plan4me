"""LangGraph orchestration of the knowledge pipeline.

Flow:
    search -> transcribe -> extract -> cluster -> synthesize

Each node reads/writes a shared TypedDict state. This is a linear graph today,
but LangGraph lets us later add conditional edges (e.g. skip clustering when
few atoms, or loop for more videos if coverage is low) without restructuring.
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.config import get_settings
from backend.ingestion.search import search_videos
from backend.ingestion.selection import select_and_transcribe
from backend.llm.extraction import extract_atoms
from backend.llm.synthesis import synthesize_report
from backend.pipeline.clustering import cluster_atoms
from backend.schemas import (
    AtomCluster,
    KnowledgeAtom,
    Transcript,
    VideoMeta,
)

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    topic: str
    max_videos: int
    languages: Optional[list[str]]
    videos: list[VideoMeta]
    transcripts: list[Transcript]
    atoms: list[KnowledgeAtom]
    clusters: list[AtomCluster]
    report_markdown: str


# --- Nodes ---

def search_node(state: PipelineState) -> PipelineState:
    """Fetch a broad candidate pool (default 50) in relevance order."""
    settings = get_settings()
    videos = search_videos(state["topic"], settings.num_candidates)
    return {"videos": videos}


def transcribe_node(state: PipelineState) -> PipelineState:
    """Rank-based selection: forced top-N (captions|Deepgram) + caption-fill."""
    transcripts = select_and_transcribe(
        state.get("videos", []),
        state.get("languages"),
        target=state.get("max_videos"),
    )
    return {"transcripts": transcripts}


def extract_node(state: PipelineState) -> PipelineState:
    atoms: list[KnowledgeAtom] = []
    topic = state["topic"]
    for t in state.get("transcripts", []):
        atoms.extend(extract_atoms(topic, t))
    return {"atoms": atoms}


def cluster_node(state: PipelineState) -> PipelineState:
    clusters = cluster_atoms(state.get("atoms", []))
    return {"clusters": clusters}


def synthesize_node(state: PipelineState) -> PipelineState:
    md = synthesize_report(state["topic"], state.get("clusters", []))
    return {"report_markdown": md}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("search", search_node)
    g.add_node("transcribe", transcribe_node)
    g.add_node("extract", extract_node)
    g.add_node("cluster", cluster_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "search")
    g.add_edge("search", "transcribe")
    g.add_edge("transcribe", "extract")
    g.add_edge("extract", "cluster")
    g.add_edge("cluster", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


# Compiled once at import; cheap and reusable.
GRAPH = build_graph()


def run_pipeline(topic: str, max_videos: int | None = None,
                 languages: list[str] | None = None) -> PipelineState:
    settings = get_settings()
    initial: PipelineState = {
        "topic": topic,
        "max_videos": max_videos or settings.default_max_videos,
        "languages": languages,
    }
    return GRAPH.invoke(initial)
