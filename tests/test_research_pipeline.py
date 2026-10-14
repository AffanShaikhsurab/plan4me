"""Unit tests for the two-depth research pipeline. No network."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.pipeline.jobs import ResearchJob, _run
from backend.research.agenda import apply_agenda_rules
from backend.research.deep import book_access, research_concept
from backend.research.final_doc import FINAL_HEADINGS, ensure_headings, write_final_document
from backend.research.pipeline import ResearchError, run_research, stream_research
from backend.research.render import render_video_brief
from backend.research.video_brief import produce_video_brief
from backend.research.web import url_allowed
from backend.schemas import (
    Agenda,
    AgendaBook,
    AgendaConcept,
    BookMention,
    ClaimItem,
    ReportRequest,
    TopicItem,
    Transcript,
    TranscriptSnippet,
    VideoBrief,
    VideoMeta,
    WebHit,
    FetchedPage,
)


def _brief(video_id: str, topic_name: str, what: str, claim: str, label: str, timestamp: float, book: BookMention | None = None) -> VideoBrief:
    return VideoBrief(
        video_id=video_id,
        title=video_id,
        channel="channel",
        topics=[TopicItem(name=topic_name, what_it_is=what)],
        claims=[ClaimItem(text=claim, timestamp_seconds=timestamp, label=label)],
        books=[book] if book else [],
    )


def _fixture_briefs() -> tuple[VideoBrief, VideoBrief, VideoBrief]:
    brief_a = _brief(
        "aaa",
        "Do not share the unbuilt roadmap",
        "do not announce features that are not shipped.",
        "Sharing the roadmap invites copies.",
        "heuristic",
        120,
    )
    brief_b = _brief(
        "bbb",
        "Keep unshipped features private",
        "do not post the feature plan before it exists.",
        "We lost a deal after posting the roadmap.",
        "self-reported",
        40,
        BookMention(title="Obviously Awesome", author="April Dunford"),
    )
    brief_c = _brief(
        "ccc",
        "Post a daily screenshot",
        "post one screenshot of the work every day.",
        "Post a screenshot every day.",
        "heuristic",
        10,
    )
    return brief_a, brief_b, brief_c


def test_merge_support_recomputes_two_videos():
    brief_a, brief_b, _brief_c = _fixture_briefs()
    raw = Agenda(
        topic="how to build in public",
        concepts=[
            AgendaConcept(
                name="Do not share the unbuilt roadmap",
                what_videos_agree="Do not announce features that are not shipped.",
                video_ids=["aaa", "bbb"],
                support_count=1,
                label="single-source",
                contested=False,
                operable_question="Would you post the unbuilt roadmap?",
                books=[
                    AgendaBook(
                        title="Obviously Awesome",
                        author="April Dunford",
                        video_id="bbb",
                    )
                ],
            )
        ],
    )
    agenda = apply_agenda_rules(raw, [brief_a, brief_b])
    concept = agenda.concepts[0]
    assert concept.support_count == 2
    assert concept.label == "multi-source"
    assert concept.contested is False
    assert len(concept.books) == 1
    assert concept.deep_research is True


def test_single_video_topic_is_not_deep_researched():
    brief_a, brief_b, brief_c = _fixture_briefs()
    raw = Agenda(
        topic="how to build in public",
        concepts=[
            AgendaConcept(
                name="Do not share the unbuilt roadmap",
                video_ids=["aaa", "bbb"],
                books=[
                    AgendaBook(
                        title="Obviously Awesome",
                        author="April Dunford",
                        video_id="bbb",
                    )
                ],
            ),
            AgendaConcept(
                name="Post a daily screenshot",
                video_ids=["ccc"],
                support_count=4,
                label="multi-source",
                deep_research=True,
            ),
        ],
    )
    agenda = apply_agenda_rules(raw, [brief_a, brief_b, brief_c])
    single = next(concept for concept in agenda.concepts if concept.name == "Post a daily screenshot")
    assert single.label == "single-source"
    assert single.deep_research is False


def test_cap_twelve_keeps_highest_support():
    briefs = [
        VideoBrief(
            video_id=f"v{index:02d}",
            title=f"v{index:02d}",
            topics=[TopicItem(name="n", what_it_is="w")],
            claims=[ClaimItem(text="c", timestamp_seconds=1)],
        )
        for index in range(20)
    ]
    concepts = []
    for index in range(15):
        video_ids = [f"v{n:02d}" for n in range(16 - index)]
        concepts.append(
            AgendaConcept(
                name=f"concept-{index:02d}",
                video_ids=video_ids,
                support_count=1,
                label="single-source",
            )
        )
    agenda = apply_agenda_rules(Agenda(topic="t", concepts=concepts), briefs)
    deep = [concept for concept in agenda.concepts if concept.deep_research]
    held_back = [
        concept
        for concept in agenda.concepts
        if concept.label == "multi-source" and not concept.deep_research
    ]
    assert len(deep) == 12
    assert len(held_back) == 3
    assert min(concept.support_count for concept in deep) >= max(
        concept.support_count for concept in held_back
    )


def test_model_cannot_inflate_support():
    brief_a, _brief_b, _brief_c = _fixture_briefs()
    raw = Agenda(
        topic="t",
        concepts=[
            AgendaConcept(
                name="Only here",
                video_ids=["aaa"],
                support_count=9,
                label="multi-source",
                deep_research=True,
            )
        ],
    )
    agenda = apply_agenda_rules(raw, [brief_a])
    concept = agenda.concepts[0]
    assert concept.support_count == 1
    assert concept.label == "single-source"
    assert concept.deep_research is False


def test_unknown_video_id_is_dropped():
    brief_a, brief_b, _brief_c = _fixture_briefs()
    raw = Agenda(
        topic="t",
        concepts=[AgendaConcept(name="Ghost", video_ids=["aaa", "not-a-real-video"])],
    )
    agenda = apply_agenda_rules(raw, [brief_a, brief_b])
    assert agenda.concepts == []


def test_blocked_host_is_not_fetched():
    assert url_allowed("https://libgen.example/book") is False
    fetched: list[str] = []

    def fake_search(query: str, max_results: int = 5) -> list[WebHit]:
        del query, max_results
        return [WebHit(title="book", url="https://libgen.example/book", snippet="")]

    def fake_fetch(url: str) -> FetchedPage:
        fetched.append(url)
        raise AssertionError("fetch_page should not be called")

    research_concept(
        "how to build in public",
        AgendaConcept(
            name="Keep a public log",
            video_ids=["aaa", "bbb"],
            support_count=2,
            label="multi-source",
            deep_research=True,
        ),
        search=fake_search,
        fetch=fake_fetch,
    )
    assert fetched == []


def test_pdf_url_is_not_allowed():
    assert url_allowed("https://example.com/file.pdf") is False
    assert url_allowed("https://example.com/notes") is True


def test_book_access_threshold():
    assert book_access(["a", "b"]) == "access-required"
    assert book_access(["a", "b", "c"]) == "summaries-mapped"


def test_empty_captions_stop_before_the_brief_model(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("backend.research.pipeline.default_runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "backend.research.pipeline.search_videos",
        lambda *args, **kwargs: [
            VideoMeta(video_id="zzz", title="No captions", url="https://youtu.be/zzz")
        ],
    )
    monkeypatch.setattr(
        "backend.research.pipeline.select_and_transcribe",
        lambda *args, **kwargs: [],
    )
    called = {"n": 0}

    def boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("brief model was called")

    monkeypatch.setattr("backend.research.pipeline.produce_video_brief", boom)
    with pytest.raises(ResearchError, match="no captions found for this topic"):
        run_research("how to build in public", 3, ["en"], "emptycaps001")
    assert called["n"] == 0


def test_ensure_headings_inserts_numbers():
    updated = ensure_headings("## What to do\n\nShip it.\n")
    assert "## Numbers" in updated
    numbers = updated.split("## Numbers", 1)[1]
    assert "Nothing in the sources opened." in numbers


def test_max_videos_rejects_51():
    with pytest.raises(ValidationError, match="max_videos must be 1 to 50"):
        ReportRequest(topic="how to build in public", max_videos=51)
    with pytest.raises(ResearchError, match="max_videos must be 1 to 50"):
        list(stream_research("how to build in public", 51, None, "job123456789"))


def test_render_video_brief_headings(tmp_path: Path):
    brief_a, _brief_b, _brief_c = _fixture_briefs()
    path = tmp_path / "aaa.md"
    path.write_text(render_video_brief(brief_a), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## Topics",
        "## Claims",
        "## Processes",
        "## Checklist",
        "## Books named",
        "## Numbers",
        "## Disagreements inside this video",
    ):
        assert heading in text
    assert "(aaa 120s, heuristic)" in text
    assert "No web page was opened for it." in text


def test_video_brief_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    fixed = VideoBrief(
        video_id="wrong",
        title="wrong",
        topics=[TopicItem(name="Do the thing", what_it_is="one action")],
        claims=[ClaimItem(text="A claim from the captions.", timestamp_seconds=5, label="heuristic")],
    )

    class _Runner:
        def invoke(self, messages):
            del messages
            return fixed

    monkeypatch.setattr("backend.research.video_brief.structured", lambda llm, schema: _Runner())
    monkeypatch.setattr("backend.research.video_brief.get_extraction_llm", lambda: object())
    transcript = Transcript(
        video_id="vid123",
        title="Real title",
        channel="Chan",
        snippets=[TranscriptSnippet(text="hello from the video", start=5, duration=1)],
    )
    brief = produce_video_brief("how to build in public", transcript, tmp_path)
    saved = (tmp_path / "videos" / "vid123.md").read_text(encoding="utf-8")
    assert (tmp_path / "videos" / "vid123.json").exists()
    assert "## Claims" in saved
    assert brief.video_id == "vid123"
    assert brief.title == "Real title"
    assert brief.channel == "Chan"


def test_deep_pass_does_not_summarize_the_book_or_fetch_youtube():
    queries: list[str] = []
    fetched: list[str] = []

    def fake_search(query: str, max_results: int = 5) -> list[WebHit]:
        del max_results
        queries.append(query)
        return [
            WebHit(title="Talk", url="https://www.youtube.com/watch?v=abc", snippet=""),
            WebHit(title="Notes", url="https://example.com/notes", snippet="short"),
        ]

    def fake_fetch(url: str) -> FetchedPage:
        fetched.append(url)
        return FetchedPage(url=url, title="Notes", text="too short to keep", word_count=4, error=None)

    note = research_concept(
        "how to build in public",
        AgendaConcept(
            name="Do not share the unbuilt roadmap",
            video_ids=["aaa", "bbb"],
            support_count=2,
            label="multi-source",
            deep_research=True,
            books=[
                AgendaBook(title="Obviously Awesome", author="April Dunford", video_id="bbb")
            ],
        ),
        search=fake_search,
        fetch=fake_fetch,
    )
    assert fetched
    assert all("youtube.com" not in url and "youtu.be" not in url for url in fetched)
    for query in queries:
        lowered = query.lower()
        assert not ("summary of" in lowered and "obviously awesome" in lowered)
    what_to_do = note.markdown.split("## What to do", 1)[1].split("## What not to do", 1)[0]
    assert "1." not in what_to_do
    assert "no public page opened that passed the keep rules" in note.markdown


def test_final_document_fills_missing_headings(monkeypatch):
    class _FakeModel:
        def invoke(self, messages):
            del messages
            return "## What to do\n\nShip it.\n"

    monkeypatch.setattr("backend.research.final_doc.get_synthesis_llm", lambda: _FakeModel())
    text = write_final_document(
        "how to build in public",
        Agenda(topic="how to build in public", concepts=[]),
        [],
        [],
        job_id="abc123abc123",
        candidates_searched=50,
        transcripts_used=0,
        skipped_ids=["zzz"],
        opened_on="2026-10-14",
    )
    for heading in FINAL_HEADINGS:
        assert heading in text
    assert "Video figures are self-reported or heuristic unless a row in Numbers says otherwise." in text
    assert "No concept was multi-source." in text
    assert "## Numbers" in text
    assert "Nothing in the sources opened." in text


def test_legacy_flag_enters_stream_pipeline(monkeypatch):
    from backend.config import Settings

    entered: list[str] = []

    def fake_legacy(*args, **kwargs):
        del args, kwargs
        entered.append("legacy")
        yield "search", {"videos": []}

    def fake_new(*args, **kwargs):
        del args, kwargs
        entered.append("new")
        if False:
            yield "search", {}

    monkeypatch.setattr("backend.pipeline.jobs.stream_pipeline", fake_legacy)
    monkeypatch.setattr("backend.pipeline.jobs.stream_research", fake_new)
    monkeypatch.setattr("backend.pipeline.jobs.save_atoms", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.pipeline.jobs.save_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "backend.pipeline.jobs.get_settings",
        lambda: Settings(_env_file=None, legacy_atom_pipeline=True),
    )
    job = ResearchJob(job_id="abc123abc123", topic="how to build in public", max_videos=1)
    _run(job, None)
    assert entered == ["legacy"]
    assert job.status == "done"


@pytest.mark.live
def test_live_build_in_public():
    from dotenv import load_dotenv

    from backend.config import get_settings
    from backend.llm.chat import reset_provider_cache
    from backend.research.pipeline import default_runs_root
    from backend.schemas import Agenda

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local", override=True)
    get_settings.cache_clear()
    reset_provider_cache()

    report = run_research("how to build in public", max_videos=3, languages=["en"])
    assert report.transcripts_found >= 1
    run_dir = default_runs_root() / report.job_id
    print(f"LIVE_RUN {run_dir}")
    final = run_dir / "FINAL.md"
    assert final.exists()
    text = final.read_text(encoding="utf-8")
    for heading in FINAL_HEADINGS:
        assert heading in text
    agenda = Agenda.model_validate_json((run_dir / "agenda.json").read_text(encoding="utf-8"))
    concept_dir = run_dir / "concepts"
    notes = list(concept_dir.glob("*.json")) if concept_dir.exists() else []
    deep = [concept for concept in agenda.concepts if concept.deep_research]
    assert len(notes) == len(deep)
    for concept in deep:
        assert concept.label == "multi-source"
        assert concept.support_count >= 2
    if not deep:
        assert "No concept was multi-source." in text
    pages = run_dir / "pages"
    if pages.exists():
        import json

        for path in pages.glob("*.json"):
            url = str(json.loads(path.read_text(encoding="utf-8")).get("url", ""))
            assert not url.lower().split("?", 1)[0].endswith(".pdf")
