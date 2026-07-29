"""Deepgram transcription fallback for caption-less videos.

Used only for the top-ranked videos (see selection.py) that lack captions.
Flow: yt-dlp downloads the raw best-audio stream (no ffmpeg needed) -> send
bytes to Deepgram Nova-3 with diarization + utterances -> map utterances into
our Transcript schema (preserving speaker labels and timestamps).

Deepgram v5 SDK:
    client = DeepgramClient(api_key=...)
    resp = client.listen.v1.media.transcribe_file(
        request=<bytes>, model="nova-3", diarize=True, utterances=True,
        smart_format=True, punctuate=True)
    resp.results.utterances -> [{start, end, transcript, speaker}, ...]
"""
from __future__ import annotations

import logging
import os
import tempfile

import yt_dlp

from backend.config import get_settings
from backend.schemas import Transcript, TranscriptSnippet, VideoMeta

logger = logging.getLogger(__name__)

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from deepgram import DeepgramClient

        s = get_settings()
        _CLIENT = DeepgramClient(api_key=s.deepgram_api_key)
    return _CLIENT


def _download_bestaudio(video: VideoMeta, out_dir: str) -> str | None:
    """Download the raw best-audio stream (no ffmpeg post-processing)."""
    out_tmpl = os.path.join(out_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video.url])
    for f in os.listdir(out_dir):
        if f.startswith(video.video_id):
            return os.path.join(out_dir, f)
    return None


def transcribe_with_deepgram(video: VideoMeta) -> Transcript | None:
    settings = get_settings()
    if not settings.deepgram_enabled:
        logger.warning("deepgram requested for %s but no API key set", video.video_id)
        return None

    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio_path = _download_bestaudio(video, tmp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio download failed for %s: %s", video.video_id, exc)
            return None
        if not audio_path:
            logger.warning("no audio file produced for %s", video.video_id)
            return None

        try:
            with open(audio_path, "rb") as fh:
                audio_bytes = fh.read()
            resp = _client().listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=settings.deepgram_model,
                diarize=True,
                utterances=True,
                smart_format=True,
                punctuate=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("deepgram transcription failed for %s: %s", video.video_id, exc)
            return None

    snippets = _utterances_to_snippets(resp)
    if not snippets:
        # Fall back to the flat transcript if utterances are absent.
        try:
            flat = resp.results.channels[0].alternatives[0].transcript
            if flat:
                snippets = [TranscriptSnippet(text=flat, start=0.0)]
        except (AttributeError, IndexError):
            pass

    if not snippets:
        return None

    return Transcript(
        video_id=video.video_id,
        title=video.title,
        channel=video.channel,
        language="en",
        source="deepgram",
        snippets=snippets,
    )


def _utterances_to_snippets(resp) -> list[TranscriptSnippet]:
    snippets: list[TranscriptSnippet] = []
    utterances = getattr(getattr(resp, "results", None), "utterances", None) or []
    for u in utterances:
        text = getattr(u, "transcript", None) or ""
        if not text.strip():
            continue
        start = float(getattr(u, "start", 0.0) or 0.0)
        end = float(getattr(u, "end", start) or start)
        spk = getattr(u, "speaker", None)
        snippets.append(
            TranscriptSnippet(
                text=text.strip(),
                start=start,
                duration=max(0.0, end - start),
                speaker=f"SPEAKER_{spk}" if spk is not None else None,
            )
        )
    return snippets
