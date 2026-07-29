"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import {
  buildReport,
  getHealth,
  getLatestReport,
  searchVideos,
  type HealthInfo,
  type KnowledgeReport,
  type VideoMeta,
} from "./api";

export default function Home() {
  const [topic, setTopic] = useState("how to get a remote software job");
  const [maxVideos, setMaxVideos] = useState(10);

  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [candidates, setCandidates] = useState<VideoMeta[] | null>(null);
  const [report, setReport] = useState<KnowledgeReport | null>(null);

  const [searching, setSearching] = useState(false);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Interactive checklist state, keyed by source line of the markdown checkbox.
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    // Show the last generated report on load so the page isn't empty.
    getLatestReport()
      .then((r) => {
        if (r && r.markdown) {
          setReport(r);
          setTopic(r.topic);
        }
      })
      .catch(() => {});
  }, []);

  async function onPreview() {
    setError(null);
    setSearching(true);
    setCandidates(null);
    try {
      setCandidates(await searchVideos(topic, maxVideos));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }

  async function onBuild() {
    setError(null);
    setBuilding(true);
    setCandidates(null);
    setChecked({});
    try {
      setReport(await buildReport(topic, maxVideos));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBuilding(false);
    }
  }

  // Custom renderer: make GFM task-list checkboxes interactive.
  // Checkboxes render in stable document order, so a render-scoped counter
  // gives each one a consistent key across re-renders.
  let checkboxIndex = 0;
  const mdComponents: Components = {
    input(props) {
      const anyProps = props as { type?: string };
      if (anyProps.type !== "checkbox") return <input {...props} />;
      const key = String(checkboxIndex++);
      return (
        <input
          type="checkbox"
          className="check"
          checked={!!checked[key]}
          onChange={() => setChecked((s) => ({ ...s, [key]: !s[key] }))}
        />
      );
    },
  };

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <h1 className="title">
            plan<span className="thin">4me</span>
          </h1>
          <p className="subtitle">
            Turn dozens of hours of interviews and talks into one comprehensive,
            evidence-backed guide — every insight traced back to its sources.
          </p>
        </div>
        {health && (
          <span className="model-chip" data-testid="health-badge">
            {health.extraction_model} · {health.synthesis_model}
          </span>
        )}
      </header>

      <div className="searchbar">
        <input
          className="topic"
          data-testid="topic-input"
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Teach me everything about…"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !building) onBuild();
          }}
        />
        <span className="videos-input">
          <input
            data-testid="max-videos-input"
            type="number"
            min={1}
            max={50}
            value={maxVideos}
            onChange={(e) => setMaxVideos(Number(e.target.value))}
            title="Number of videos to process"
          />
          videos
        </span>
        <button
          className="btn-ghost"
          data-testid="preview-btn"
          onClick={onPreview}
          disabled={searching || building || !topic.trim()}
        >
          {searching ? "Searching…" : "Preview"}
        </button>
        <button
          className="btn-primary"
          data-testid="build-btn"
          onClick={onBuild}
          disabled={building || searching || !topic.trim()}
        >
          {building ? "Building…" : "Build knowledge"}
        </button>
      </div>

      {error && (
        <p className="error" data-testid="error">
          {error}
        </p>
      )}

      {building && (
        <div className="loading" data-testid="building-status">
          <span className="spinner" />
          Searching YouTube → pulling transcripts → extracting knowledge atoms →
          clustering → synthesizing. This can take a minute or two.
        </div>
      )}

      {candidates && (
        <div className="candidates" data-testid="candidates">
          {candidates.map((c) => (
            <div className="candidate" key={c.video_id}>
              <a href={c.url} target="_blank" rel="noreferrer">
                {c.title}
              </a>
              {c.channel && <div className="ch">{c.channel}</div>}
            </div>
          ))}
        </div>
      )}

      {report && (
        <>
          <div className="stats" data-testid="stats">
            <span className="stat-pill">
              <b>{report.videos_processed}</b> videos
            </span>
            <span className="stat-pill">
              <b>{report.transcripts_found}</b> transcripts
            </span>
            <span className="stat-pill">
              <b>{report.atoms_extracted}</b> knowledge atoms
            </span>
            <span className="stat-pill">
              <b>{report.clusters}</b> deduplicated insights
            </span>
          </div>

          <article className="report" data-testid="report">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
              {report.markdown}
            </ReactMarkdown>
          </article>
        </>
      )}
    </main>
  );
}
