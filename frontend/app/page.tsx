"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import {
  getHealth,
  getLatestReport,
  getResearchJob,
  searchVideos,
  startResearch,
  type HealthInfo,
  type KnowledgeReport,
  type ResearchJob,
  type ResearchStage,
  type VideoMeta,
} from "./api";

// The Three.js desk scene in ./ResearchScene.tsx is kept for reference but is
// deliberately not mounted: the research view reads better as quiet progress.

type Phase = "idle" | "portal" | "research" | "report";

type PortalRect = { top: number; left: number; width: number; height: number };

const STAGES: Array<{
  id: Exclude<ResearchStage, "done">;
  label: string;
  countKey: "videos" | "transcripts" | "atoms" | "clusters" | null;
  countLabel: string;
}> = [
  {
    id: "search",
    label: "Gathering sources",
    countKey: "videos",
    countLabel: "found",
  },
  {
    id: "transcribe",
    label: "Reading transcripts",
    countKey: "transcripts",
    countLabel: "read",
  },
  {
    id: "extract",
    label: "Analysing insights",
    countKey: "atoms",
    countLabel: "notes",
  },
  {
    id: "cluster",
    label: "Connecting ideas",
    countKey: "clusters",
    countLabel: "themes",
  },
  {
    id: "synthesize",
    label: "Writing your guide",
    countKey: null,
    countLabel: "",
  },
];

const PORTAL_DURATION = 1000;

function slugify(value: string) {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "research"
  );
}

export default function Home() {
  const [topic, setTopic] = useState("");
  const [maxVideos, setMaxVideos] = useState(10);

  const [phase, setPhase] = useState<Phase>("idle");
  const [portalRect, setPortalRect] = useState<PortalRect | null>(null);
  const [portalOpen, setPortalOpen] = useState(false);

  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [candidates, setCandidates] = useState<VideoMeta[] | null>(null);
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [report, setReport] = useState<KnowledgeReport | null>(null);
  const [lastReport, setLastReport] = useState<KnowledgeReport | null>(null);

  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [copied, setCopied] = useState(false);

  const composerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    getLatestReport()
      .then((r) => {
        if (r && r.markdown) setLastReport(r);
      })
      .catch(() => {});
  }, []);

  // Poll the running job so the progress view reflects real pipeline stages.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const jobId = job.job_id;
    const timer = window.setInterval(async () => {
      try {
        const next = await getResearchJob(jobId);
        setJob(next);
        if (next.status === "done" && next.report) {
          setReport(next.report);
          setLastReport(next.report);
        }
        if (next.status === "error") {
          setError(next.error ?? "Research failed.");
        }
      } catch (e) {
        setError((e as Error).message);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  // Give the finished guide a beat to land before switching surfaces.
  useEffect(() => {
    if (phase !== "research" || job?.status !== "done" || !report) return;
    const timer = window.setTimeout(() => setPhase("report"), 2200);
    return () => window.clearTimeout(timer);
  }, [phase, job?.status, report]);

  const onSubmit = useCallback(async () => {
    const trimmed = topic.trim();
    if (!trimmed) return;

    const rect = composerRef.current?.getBoundingClientRect();
    if (rect) {
      setPortalRect({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
    }

    setError(null);
    setCandidates(null);
    setReport(null);
    setChecked({});
    setPortalOpen(false);
    setPhase("portal");

    // Two frames so the portal paints at the composer's size before expanding.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => setPortalOpen(true)),
    );
    window.setTimeout(() => {
      setPhase((current) => (current === "portal" ? "research" : current));
    }, PORTAL_DURATION);

    try {
      setJob(await startResearch(trimmed, maxVideos));
    } catch (e) {
      setError((e as Error).message);
      setPhase("idle");
      setPortalOpen(false);
    }
  }, [topic, maxVideos]);

  async function onPreview() {
    const trimmed = topic.trim();
    if (!trimmed) return;
    setError(null);
    setSearching(true);
    setCandidates(null);
    try {
      setCandidates(await searchVideos(trimmed, maxVideos));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }

  function backToSearch() {
    setPhase("idle");
    setPortalOpen(false);
    setPortalRect(null);
    setJob(null);
    setCopied(false);
  }

  async function copyMarkdown() {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(report.markdown);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Clipboard access was blocked by the browser.");
    }
  }

  function downloadMarkdown() {
    if (!report) return;
    const blob = new Blob([report.markdown], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${slugify(report.topic)}.md`;
    link.click();
    URL.revokeObjectURL(url);
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

  const stage = job?.stage ?? "search";
  const completed = job?.completed_stages ?? [];
  const counts = job?.counts ?? {};
  const finished = job?.status === "done";
  const failed = job?.status === "error";

  const doneCount = STAGES.filter((s) => completed.includes(s.id)).length;
  // Half a step of credit for the stage in flight, so the bar always moves.
  const progress = finished
    ? 1
    : Math.min(1, (doneCount + 0.5) / STAGES.length);
  const activeStage = STAGES.find((s) => s.id === stage);

  return (
    <main className={`app phase-${phase}`}>
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-background" />
        <div className="hero-shade" />

        <nav className="glass-nav" aria-label="Primary">
          <span className="nav-brand">plan4me</span>
          <span className="nav-purpose">Collective knowledge from video</span>
          {health && (
            <span className="model-chip" data-testid="health-badge">
              <span className="status-dot" />
              Research engine online
            </span>
          )}
        </nav>

        <div className="hero-content">
          <p className="eyebrow">Your research companion</p>
          <h1 className="title" id="hero-title">
            Learn from <em>every voice.</em>
          </h1>
          <p className="subtitle">
            Turn hours of expert interviews and talks into one clear,
            evidence-backed guide.
          </p>

          <div className="composer" ref={composerRef}>
            <label className="sr-only" htmlFor="topic">
              What would you like to research?
            </label>
            <textarea
              id="topic"
              className="topic"
              data-testid="topic-input"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Teach me everything about…"
              rows={3}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSubmit();
                }
              }}
            />

            <div className="composer-footer">
              <div className="research-options">
                <label className="videos-input">
                  <input
                    data-testid="max-videos-input"
                    type="number"
                    min={1}
                    max={50}
                    value={maxVideos}
                    onChange={(e) => setMaxVideos(Number(e.target.value))}
                  />
                  <span>videos</span>
                </label>
                <button
                  className="preview-button"
                  data-testid="preview-btn"
                  onClick={onPreview}
                  disabled={searching || !topic.trim()}
                >
                  {searching ? "Finding sources…" : "Preview sources"}
                </button>
              </div>

              <button
                className="send-button"
                data-testid="build-btn"
                onClick={onSubmit}
                disabled={!topic.trim()}
                aria-label="Begin research"
                title="Begin research"
              >
                <span aria-hidden="true">↑</span>
              </button>
            </div>
          </div>

          {error && phase === "idle" && (
            <p className="error glass-message" data-testid="error">
              {error}
            </p>
          )}

          {lastReport && phase === "idle" && (
            <button
              className="last-guide"
              onClick={() => {
                setReport(lastReport);
                setPhase("report");
              }}
            >
              Open your last guide: {lastReport.topic} →
            </button>
          )}

          {candidates && phase === "idle" && (
            <div className="preview-panel" data-testid="candidates">
              <div className="preview-panel-head">
                <span>{candidates.length} sources found</span>
                <button onClick={() => setCandidates(null)}>Hide</button>
              </div>
              {candidates.length > 0 ? (
                <ul>
                  {candidates.map((candidate, index) => (
                    <li key={candidate.video_id}>
                      <span className="candidate-index">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <a
                        href={candidate.url}
                        target="_blank"
                        rel="noreferrer"
                        title={candidate.title}
                      >
                        {candidate.title}
                        {candidate.channel && (
                          <span className="ch">{candidate.channel}</span>
                        )}
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">
                  No matching videos. Try a more specific topic.
                </p>
              )}
            </div>
          )}
        </div>
      </section>

      {phase === "portal" && portalRect && (
        <div
          className={`portal ${portalOpen ? "is-open" : ""}`}
          style={
            {
              "--portal-top": `${portalRect.top}px`,
              "--portal-left": `${portalRect.left}px`,
              "--portal-width": `${portalRect.width}px`,
              "--portal-height": `${portalRect.height}px`,
            } as React.CSSProperties
          }
          aria-hidden="true"
        >
          <span className="portal-topic">{topic.trim()}</span>
        </div>
      )}

      {phase === "research" && (
        <section className="progress-view" aria-live="polite">
          <div className="progress-card" data-testid="building-status">
            <p className="section-kicker">Researching</p>
            <h2 className="progress-topic">{job?.topic ?? topic}</h2>

            <div
              className="progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress * 100)}
            >
              <span
                className="progress-fill"
                style={{ width: `${progress * 100}%` }}
              />
            </div>

            <p className="progress-now">
              {failed
                ? "Research stopped"
                : finished
                  ? "Your guide is ready"
                  : (activeStage?.label ?? "Starting up") + "…"}
            </p>

            <ol className="progress-trace">
              {STAGES.map((item) => {
                const isDone = completed.includes(item.id);
                const isActive = !isDone && !finished && stage === item.id;
                const value = item.countKey ? counts[item.countKey] : undefined;
                return (
                  <li
                    key={item.id}
                    className={`trace-row ${isDone ? "is-done" : ""} ${
                      isActive ? "is-active" : ""
                    }`}
                  >
                    <span className="trace-dot" aria-hidden="true" />
                    <span className="trace-label">{item.label}</span>
                    {typeof value === "number" && value > 0 && (
                      <span className="trace-count">
                        {value} {item.countLabel}
                      </span>
                    )}
                  </li>
                );
              })}
            </ol>

            {failed && (
              <p className="error" data-testid="error">
                {job?.error ?? error}
              </p>
            )}

            <div className="progress-foot">
              {finished && report ? (
                <button
                  className="arrival-button"
                  onClick={() => setPhase("report")}
                >
                  Open your guide
                </button>
              ) : (
                <>
                  <span className="progress-elapsed">
                    {job ? `${Math.round(job.elapsed_seconds)}s` : "—"}
                  </span>
                  <button className="quiet-link" onClick={backToSearch}>
                    Cancel
                  </button>
                </>
              )}
            </div>
          </div>
        </section>
      )}

      {phase === "report" && report && (
        <section className="report-surface" id="research">
          <header className="report-bar">
            <div className="report-bar-title">
              <p className="section-kicker">Your guide</p>
              <h2>{report.topic}</h2>
            </div>
            <div className="report-actions">
              {report.atoms_extracted > 0 && (
                <>
                  <button className="ghost-button" onClick={copyMarkdown}>
                    {copied ? "Copied" : "Copy markdown"}
                  </button>
                  <button className="ghost-button" onClick={downloadMarkdown}>
                    Download .md
                  </button>
                </>
              )}
              <button className="ghost-button is-primary" onClick={backToSearch}>
                New research
              </button>
            </div>
          </header>

          <div className="report-scroll">
            <div className="stats" data-testid="stats">
              <span className="stat-pill">
                <b>{report.videos_processed}</b> candidates scanned
              </span>
              <span className="stat-pill">
                <b>{report.transcripts_found}</b> videos transcribed
              </span>
              <span className="stat-pill">
                <b>{report.atoms_extracted}</b> notes extracted
              </span>
              <span className="stat-pill">
                <b>{report.clusters}</b> themes connected
              </span>
            </div>

            {report.atoms_extracted > 0 ? (
              <article className="report" data-testid="report">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={mdComponents}
                >
                  {report.markdown}
                </ReactMarkdown>
              </article>
            ) : (
              <div className="report-empty" data-testid="report">
                <h3>No usable transcripts came back</h3>
                <p>
                  {report.videos_processed} videos matched this topic, but none
                  of them returned a transcript we could read, so there was
                  nothing to analyse. Try rephrasing it, or pick a topic with
                  more recorded talks and interviews.
                </p>
                <button className="ghost-button" onClick={backToSearch}>
                  Try another topic
                </button>
              </div>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
