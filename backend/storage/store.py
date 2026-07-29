"""Minimal SQLite persistence for atoms and reports.

Deliberately simple for the prototype. The schema mirrors KnowledgeAtom so we
can later swap SQLite for Postgres + pgvector (store the embedding alongside
each atom) without changing call sites.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

from backend.config import get_settings
from backend.schemas import KnowledgeAtom, KnowledgeReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS atoms (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    type TEXT NOT NULL,
    claim TEXT NOT NULL,
    actionable_step TEXT,
    confidence REAL,
    quote TEXT,
    video_id TEXT,
    video_title TEXT,
    channel TEXT,
    timestamp REAL,
    cluster_id INTEGER,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    markdown TEXT NOT NULL,
    meta TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_atoms_topic ON atoms(topic);
"""


@contextmanager
def _conn():
    settings = get_settings()
    con = sqlite3.connect(settings.db_path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)


def save_atoms(atoms: list[KnowledgeAtom]) -> None:
    if not atoms:
        return
    now = time.time()
    rows = [
        (a.id, a.topic, a.type.value, a.claim, a.actionable_step, a.confidence,
         a.quote, a.video_id, a.video_title, a.channel, a.timestamp, a.cluster_id, now)
        for a in atoms
    ]
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO atoms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )


def get_latest_report() -> dict | None:
    """Return the most recently saved report (topic, markdown, meta) or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT topic, markdown, meta, created_at FROM reports "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    topic, markdown, meta, created_at = row
    return {
        "topic": topic,
        "markdown": markdown,
        "meta": json.loads(meta) if meta else {},
        "created_at": created_at,
    }


def save_report(report: KnowledgeReport) -> int:
    meta = json.dumps({
        "videos_processed": report.videos_processed,
        "transcripts_found": report.transcripts_found,
        "atoms_extracted": report.atoms_extracted,
        "clusters": report.clusters,
    })
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO reports (topic, markdown, meta, created_at) VALUES (?,?,?,?)",
            (report.topic, report.markdown, meta, time.time()),
        )
        return int(cur.lastrowid)
