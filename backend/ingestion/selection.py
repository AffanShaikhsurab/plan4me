"""Ranked video selection + transcript acquisition.

Strategy (per product spec):
  1. Search `num_candidates` (default 50), kept in YouTube relevance order.
  2. Build a final set of `target_videos` (default 10):
       - The top `force_deepgram_top_n` (default 5) by rank are ALWAYS included:
         use captions if present, else Deepgram.
       - Remaining slots are filled by the next highest-ranked candidates that
         already have captions (free path, no Deepgram).

This caps paid transcription at <= force_deepgram_top_n videos per report while
guaranteeing the most relevant videos are never dropped for lack of captions.
"""
from __future__ import annotations

import logging

from backend.config import get_settings
from backend.ingestion.transcripts import fetch_captions
from backend.schemas import Transcript, VideoMeta

logger = logging.getLogger(__name__)


def select_and_transcribe(
    candidates: list[VideoMeta],
    languages: list[str] | None = None,
    target: int | None = None,
) -> list[Transcript]:
    s = get_settings()
    target = target or s.target_videos
    force_n = min(s.force_deepgram_top_n, target)

    transcripts: list[Transcript] = []
    selected_ids: set[str] = set()

    # --- Phase 1: forced top-N (captions, else Deepgram) ---
    forced = candidates[:force_n]
    for v in forced:
        t = fetch_captions(v, languages)
        if t is None:
            t = _deepgram(v)
        if t and t.snippets:
            transcripts.append(t)
            selected_ids.add(v.video_id)
        else:
            logger.info("forced video %s produced no transcript; skipping", v.video_id)

    # --- Phase 2: fill remaining slots with caption-having candidates ---
    for v in candidates[force_n:]:
        if len(transcripts) >= target:
            break
        if v.video_id in selected_ids:
            continue
        t = fetch_captions(v, languages)  # fetch doubles as the caption check
        if t and t.snippets:
            transcripts.append(t)
            selected_ids.add(v.video_id)

    logger.info(
        "selected %d/%d transcripts from %d candidates (forced top %d)",
        len(transcripts), target, len(candidates), force_n,
    )
    return transcripts


def _deepgram(video: VideoMeta) -> Transcript | None:
    s = get_settings()
    if not s.deepgram_enabled:
        logger.info("no captions for top video %s and Deepgram disabled", video.video_id)
        return None
    # Lazy import so the core installs without the deepgram SDK.
    try:
        from backend.ingestion.deepgram_stt import transcribe_with_deepgram
    except ImportError as exc:  # pragma: no cover
        logger.warning("deepgram SDK unavailable: %s", exc)
        return None
    logger.info("using Deepgram for caption-less top video %s", video.video_id)
    return transcribe_with_deepgram(video)
