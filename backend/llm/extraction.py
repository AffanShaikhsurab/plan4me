"""Knowledge-atom extraction.

Each transcript is chunked and passed to the extraction LLM with
`with_structured_output(AtomExtraction)` so the model returns validated,
schema-conform atoms rather than free-form prose. This is what preserves
detail and makes downstream dedupe/counting possible.
"""
from __future__ import annotations

import logging
import uuid

from backend.llm.chat import get_extraction_llm, structured
from backend.schemas import (
    AtomExtraction,
    KnowledgeAtom,
    Transcript,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a knowledge-extraction engine. Given a transcript excerpt from a "
    "video about a topic, extract every distinct, useful unit of knowledge as an "
    "atom. Optimize for INFORMATION RETENTION, not brevity. Capture advice, "
    "concrete examples, failure modes, tools, resources, and frameworks. "
    "Each atom's `claim` must be self-contained and understandable on its own. "
    "Do not merge multiple ideas into one atom. Do not invent content that is "
    "not supported by the transcript. Set `confidence` based on how strongly "
    "the speaker asserts the point. Include a short verbatim `quote` only when "
    "it is especially striking."
)


def _chunk_transcript(transcript: Transcript, max_chars: int = 6000) -> list[tuple[float, str]]:
    """Chunk snippets into ~max_chars windows, tracking each window's start time."""
    chunks: list[tuple[float, str]] = []
    buf: list[str] = []
    buf_len = 0
    start_ts: float | None = None

    for snip in transcript.snippets:
        if start_ts is None:
            start_ts = snip.start
        buf.append(snip.text)
        buf_len += len(snip.text) + 1
        if buf_len >= max_chars:
            chunks.append((start_ts, " ".join(buf)))
            buf, buf_len, start_ts = [], 0, None

    if buf:
        chunks.append((start_ts or 0.0, " ".join(buf)))
    return chunks


def extract_atoms(topic: str, transcript: Transcript) -> list[KnowledgeAtom]:
    """Extract knowledge atoms from a single transcript."""
    llm = get_extraction_llm()
    model = structured(llm, AtomExtraction)

    atoms: list[KnowledgeAtom] = []
    for chunk_start, text in _chunk_transcript(transcript):
        prompt = (
            f"TOPIC: {topic}\n"
            f"VIDEO: {transcript.title}\n"
            f"CHUNK_START_SECONDS: {chunk_start:.0f}\n\n"
            f"TRANSCRIPT EXCERPT:\n{text}"
        )
        try:
            result: AtomExtraction = model.invoke(
                [("system", _SYSTEM), ("human", prompt)]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("extraction failed for %s @ %.0fs: %s",
                           transcript.video_id, chunk_start, exc)
            continue

        # with_structured_output yields None when the model returns no parseable
        # tool call, which local/weaker models do. Skipping the chunk keeps the
        # atoms already extracted from a paid-for transcription run.
        if result is None:
            logger.warning("extraction returned no structured output for %s @ %.0fs",
                           transcript.video_id, chunk_start)
            continue

        for raw in result.atoms:
            if not raw.claim or not raw.claim.strip():
                continue  # skip malformed/empty atoms
            ts = raw.approx_timestamp if raw.approx_timestamp is not None else chunk_start
            atoms.append(
                KnowledgeAtom(
                    id=str(uuid.uuid4()),
                    topic=topic,
                    type=raw.type,
                    claim=raw.claim,
                    actionable_step=raw.actionable_step,
                    confidence=raw.confidence,
                    quote=raw.quote,
                    video_id=transcript.video_id,
                    video_title=transcript.title,
                    channel=transcript.channel,
                    timestamp=ts,
                )
            )

    logger.info("extracted %d atoms from %s", len(atoms), transcript.video_id)
    return atoms
