export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface VideoMeta {
  video_id: string;
  title: string;
  url: string;
  channel?: string | null;
  duration?: number | null;
  view_count?: number | null;
}

export type AtomType =
  | "advice"
  | "example"
  | "failure_mode"
  | "tool"
  | "resource"
  | "framework"
  | "claim";

export interface ClusterInsight {
  type: AtomType;
  claim: string;
  support_count: number;
  actionable_step?: string | null;
  quote?: string | null;
  confidence: number;
  sources: string[];
}

export interface KnowledgeReport {
  topic: string;
  videos_processed: number;
  transcripts_found: number;
  atoms_extracted: number;
  clusters: number;
  markdown: string;
  insights: ClusterInsight[];
}

export interface HealthInfo {
  /** "ok", or "degraded" when the provider could not be resolved. */
  status: string;
  region: string;
  /** null when status is "degraded" — the provider never resolved. */
  extraction_model: string | null;
  synthesis_model: string | null;
  whisper_fallback: boolean;
  /** Active chat provider: bedrock | openai | anthropic | gemini | ollama | moonshot. */
  llm_provider: string;
  /** Active embedding provider: bedrock | openai | local. */
  embedding_provider: string;
  /** Present only when status is "degraded". */
  llm_error?: string;
}

export type ResearchStage =
  | "search"
  | "transcribe"
  | "extract"
  | "cluster"
  | "synthesize"
  | "done";

export interface ResearchCounts {
  videos?: number;
  transcripts?: number;
  atoms?: number;
  clusters?: number;
}

export interface ResearchJob {
  job_id: string;
  topic: string;
  status: "running" | "done" | "error";
  stage: ResearchStage;
  completed_stages: string[];
  stage_order: string[];
  counts: ResearchCounts;
  elapsed_seconds: number;
  report: KnowledgeReport | null;
  error: string | null;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<HealthInfo> {
  return jsonOrThrow<HealthInfo>(await fetch(`${API_BASE}/health`));
}

export async function getLatestReport(): Promise<KnowledgeReport | null> {
  const res = await fetch(`${API_BASE}/reports/latest`);
  if (!res.ok) return null;
  return (await res.json()) as KnowledgeReport | null;
}

export async function searchVideos(
  query: string,
  maxResults: number
): Promise<VideoMeta[]> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_results: maxResults }),
  });
  return jsonOrThrow<VideoMeta[]>(res);
}

export async function buildReport(
  topic: string,
  maxVideos: number
): Promise<KnowledgeReport> {
  const res = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, max_videos: maxVideos }),
  });
  return jsonOrThrow<KnowledgeReport>(res);
}

export async function startResearch(
  topic: string,
  maxVideos: number
): Promise<ResearchJob> {
  const res = await fetch(`${API_BASE}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, max_videos: maxVideos }),
  });
  return jsonOrThrow<ResearchJob>(res);
}

export async function getResearchJob(jobId: string): Promise<ResearchJob> {
  return jsonOrThrow<ResearchJob>(await fetch(`${API_BASE}/research/${jobId}`));
}
