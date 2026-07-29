"""Targeted test of the Deepgram fallback path.

Finds a caption-LESS video among the search candidates (the case where our
pipeline would invoke Deepgram) and runs it through transcribe_with_deepgram,
then prints the diarized result. Never prints the API key.
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

from backend.config import get_settings
from backend.ingestion.search import search_videos
from backend.ingestion.transcripts import fetch_captions
from backend.ingestion.deepgram_stt import transcribe_with_deepgram

TOPIC = "how to get a remote software job"


def main() -> int:
    s = get_settings()
    print(f"deepgram_enabled = {s.deepgram_enabled}  model = {s.deepgram_model}")
    if not s.deepgram_enabled:
        print("No Deepgram key detected in settings; aborting.")
        return 1

    cands = search_videos(TOPIC, s.num_candidates)
    print(f"searched {len(cands)} candidates")

    # Find caption-less candidates; prefer the shortest to minimize cost.
    caption_less = []
    for v in cands:
        if fetch_captions(v) is None:
            caption_less.append(v)
        if len(caption_less) >= 8:  # enough to choose a short one
            break

    if not caption_less:
        print("No caption-less candidate found; cannot exercise Deepgram path.")
        return 1

    caption_less.sort(key=lambda v: v.duration or 10**9)
    target = caption_less[0]
    print(f"chosen caption-less video: {target.video_id}  dur={target.duration}s  {target.title[:60]!r}")

    print("--- running Deepgram transcription ---")
    t = transcribe_with_deepgram(target)
    if not t or not t.snippets:
        print("Deepgram returned no transcript.")
        return 1

    speakers = sorted({sn.speaker for sn in t.snippets if sn.speaker})
    print(f"SUCCESS: source={t.source} snippets={len(t.snippets)} speakers={speakers}")
    print("first 5 utterances:")
    for sn in t.snippets[:5]:
        print(f"  [{sn.speaker}] {sn.start:.1f}s: {sn.text[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
