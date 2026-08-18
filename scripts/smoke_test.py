"""End-to-end smoke test that verifies each layer independently.

Run:  python -m scripts.smoke_test  [--topic "..."]  [--videos N]  [--full]

By default it runs the cheap layers (imports, search, captions). Add --full to
also exercise Bedrock (extraction + synthesis), which incurs token cost.
"""
from __future__ import annotations

import argparse
import sys
import traceback


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str, exc: Exception | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if exc:
        print("        " + "".join(traceback.format_exception_only(type(exc), exc)).strip())


def test_imports() -> bool:
    print("1) Imports")
    try:
        import yt_dlp  # noqa
        import youtube_transcript_api  # noqa
        import langgraph  # noqa
        import langchain_aws  # noqa
        from backend.pipeline.graph import build_graph  # noqa
        _ok("all core modules import")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("import failure", exc)
        return False


def test_search(topic: str):
    print("2) YouTube search (yt-dlp)")
    try:
        from backend.ingestion.search import search_videos
        vids = search_videos(topic, 5)
        if vids:
            _ok(f"got {len(vids)} candidates; first: {vids[0].title[:60]!r}")
        else:
            _fail("search returned 0 results")
        return vids
    except Exception as exc:  # noqa: BLE001
        _fail("search error", exc)
        return []


def test_captions(vids):
    print("3) Captions fetch (youtube-transcript-api)")
    if not vids:
        _fail("skipped (no videos)")
        return None
    try:
        from backend.ingestion.transcripts import fetch_captions
        for v in vids:
            t = fetch_captions(v)
            if t and t.snippets:
                _ok(f"{v.video_id}: {len(t.snippets)} snippets, lang={t.language}")
                return t
        _fail("no captions found on any candidate (try --videos with more)")
        return None
    except Exception as exc:  # noqa: BLE001
        _fail("caption error", exc)
        return None


def test_bedrock():
    """Reachability of whichever provider LLM_PROVIDER selects."""
    try:
        from backend.llm.chat import active_models, get_extraction_llm
        models = active_models()
        print(f"4) LLM reachability (provider={models['provider']}, "
              f"model={models['extraction_model']})")
        resp = get_extraction_llm().invoke("Reply with the single word: ok")
        _ok(f"model replied: {str(resp.content)[:40]!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("llm error", exc)
        return False


def test_extraction(topic, transcript):
    print("5) Atom extraction (structured output)")
    if not transcript:
        _fail("skipped (no transcript)")
        return []
    try:
        from backend.llm.extraction import extract_atoms
        atoms = extract_atoms(topic, transcript)
        if atoms:
            _ok(f"extracted {len(atoms)} atoms; e.g. {atoms[0].claim[:70]!r}")
        else:
            _fail("0 atoms extracted")
        return atoms
    except Exception as exc:  # noqa: BLE001
        _fail("extraction error", exc)
        return []


def test_cluster_and_synth(topic, atoms):
    print("6) Clustering + synthesis")
    if not atoms:
        _fail("skipped (no atoms)")
        return
    try:
        from backend.pipeline.clustering import cluster_atoms
        from backend.llm.synthesis import synthesize_report
        clusters = cluster_atoms(atoms)
        _ok(f"{len(atoms)} atoms -> {len(clusters)} clusters")
        md = synthesize_report(topic, clusters)
        _ok(f"report generated ({len(md)} chars)")
        print("\n----- REPORT PREVIEW -----")
        print(md[:1200])
        print("----- END PREVIEW -----\n")
    except Exception as exc:  # noqa: BLE001
        _fail("cluster/synth error", exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="how to get a remote software job")
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="also run Bedrock layers")
    args = ap.parse_args()

    print(f"\n=== plan4me smoke test | topic={args.topic!r} ===\n")

    if not test_imports():
        return 1

    vids = test_search(args.topic)[: args.videos]
    transcript = test_captions(vids)

    if args.full:
        if test_bedrock():
            atoms = test_extraction(args.topic, transcript)
            test_cluster_and_synth(args.topic, atoms)
    else:
        print("\n(Skipping Bedrock layers. Re-run with --full to test extraction/synthesis.)")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
