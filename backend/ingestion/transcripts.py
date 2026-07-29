"""Captions-first transcript acquisition.

Strategy (the key cost/accuracy decision):
  1. Try to fetch existing captions via youtube-transcript-api (free, instant).
  2. Only if none exist AND whisper fallback is enabled, download audio and
     transcribe locally.

youtube-transcript-api >= 1.0 uses an INSTANCE API:
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=[...])
The old static `get_transcript` is deprecated.
"""
from __future__ import annotations

import logging
from typing import Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

try:  # RequestBlocked was added in newer versions
    from youtube_transcript_api import RequestBlocked  # type: ignore
except ImportError:  # pragma: no cover
    class RequestBlocked(Exception):  # type: ignore
        ...

from backend.config import get_settings
from backend.schemas import Transcript, TranscriptSnippet, VideoMeta

logger = logging.getLogger(__name__)


def fetch_captions(video: VideoMeta, languages: Optional[list[str]] = None) -> Optional[Transcript]:
    """Fetch existing YouTube captions. Returns None if unavailable."""
    settings = get_settings()
    langs = languages or settings.transcript_language_list
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video.video_id, languages=langs)
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
        logger.info("no captions for %s: %s", video.video_id, type(exc).__name__)
        return None
    except RequestBlocked:
        logger.warning("YouTube blocked caption request for %s (consider proxies)", video.video_id)
        return None
    except Exception as exc:  # noqa: BLE001 - be resilient during ingestion
        logger.warning("caption fetch failed for %s: %s", video.video_id, exc)
        return None

    snippets = [
        TranscriptSnippet(
            text=s.text,
            start=float(s.start),
            duration=float(getattr(s, "duration", 0.0) or 0.0),
        )
        for s in fetched
    ]
    return Transcript(
        video_id=video.video_id,
        title=video.title,
        channel=video.channel,
        language=getattr(fetched, "language_code", None) or (langs[0] if langs else None),
        source="captions",
        snippets=snippets,
    )


def get_transcript(video: VideoMeta, languages: Optional[list[str]] = None) -> Optional[Transcript]:
    """Captions-first, optional Whisper fallback."""
    transcript = fetch_captions(video, languages)
    if transcript is not None:
        return transcript

    settings = get_settings()
    if not settings.enable_whisper_fallback:
        return None

    # Lazy import so the core backend installs without torch/ctranslate2.
    try:
        from backend.ingestion.whisper_fallback import transcribe_with_whisper
    except ImportError as exc:  # pragma: no cover
        logger.warning("whisper fallback requested but unavailable: %s", exc)
        return None

    logger.info("falling back to whisper for %s", video.video_id)
    return transcribe_with_whisper(video)
