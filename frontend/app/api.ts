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
  status: string;
  region: string;
  extraction_model: string;
  synthesis_model: string;
  whisper_fallback: boolean;
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
