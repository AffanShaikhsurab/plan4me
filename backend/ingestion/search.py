"""YouTube search via yt-dlp's Python API.

We use `extract_flat` so the search returns lightweight metadata for many
videos WITHOUT the expensive per-video format extraction. This is the cheapest
way to build a candidate list before deciding what to actually ingest.

Ref (verified via docs): yt_dlp.YoutubeDL(opts).extract_info("ytsearchN:query",
download=False) with extract_flat returns url_result entries (_type='url').
"""
from __future__ import annotations

import logging

import yt_dlp

from backend.schemas import VideoMeta

logger = logging.getLogger(__name__)


def search_videos(query: str, max_results: int = 10) -> list[VideoMeta]:
    """Return up to `max_results` candidate videos for a topic query."""
    search_term = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # flat = fast, metadata only, no format probing
        "skip_download": True,
    }

    results: list[VideoMeta] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_term, download=False)

    entries = (info or {}).get("entries", []) or []
    for e in entries:
        vid = e.get("id")
        if not vid:
            continue
        results.append(
            VideoMeta(
                video_id=vid,
                title=e.get("title") or "(untitled)",
                url=e.get("url") or f"https://www.youtube.com/watch?v={vid}",
                channel=e.get("channel") or e.get("uploader"),
                duration=_safe_int(e.get("duration")),
                view_count=_safe_int(e.get("view_count")),
            )
        )
    logger.info("search '%s' -> %d candidates", query, len(results))
    return results


def _safe_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
