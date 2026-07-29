"""Optional local transcription fallback using faster-whisper.

This module is imported LAZILY (only when captions are missing and the
fallback is enabled), so the core backend does not require torch/ctranslate2.

Pipeline: yt-dlp downloads audio-only (m4a) via ffmpeg -> faster-whisper
transcribes -> we map segments into our Transcript schema.
"""
from __future__ import annotations

import logging
import os
import tempfile

import yt_dlp

from backend.config import get_settings
from backend.schemas import Transcript, TranscriptSnippet, VideoMeta

logger = logging.getLogger(__name__)

_MODEL = None  # cached WhisperModel instance


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        s = get_settings()
        logger.info(
            "loading faster-whisper model=%s device=%s compute=%s",
            s.whisper_model_size, s.whisper_device, s.whisper_compute_type,
        )
        _MODEL = WhisperModel(
            s.whisper_model_size,
            device=s.whisper_device,
            compute_type=s.whisper_compute_type,
        )
    return _MODEL


def _download_audio(video: VideoMeta, out_dir: str) -> str:
    out_tmpl = os.path.join(out_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "m4a/bestaudio/best",
        "outtmpl": out_tmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video.url])
    path = os.path.join(out_dir, f"{video.video_id}.m4a")
    if not os.path.exists(path):  # codec may differ; grab whatever landed
        for f in os.listdir(out_dir):
            if f.startswith(video.video_id):
                return os.path.join(out_dir, f)
    return path


def transcribe_with_whisper(video: VideoMeta) -> Transcript | None:
    model = _get_model()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio_path = _download_audio(video, tmp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio download failed for %s: %s", video.video_id, exc)
            return None

        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        snippets = [
            TranscriptSnippet(text=seg.text.strip(), start=float(seg.start),
                              duration=float(seg.end - seg.start))
            for seg in segments
        ]

    return Transcript(
        video_id=video.video_id,
        title=video.title,
        channel=video.channel,
        language=getattr(info, "language", None),
        source="whisper",
        snippets=snippets,
    )
