/** Training-tab specific REST + WebSocket helpers. */

import { getJson, postJson } from "@/api/http";

export interface TrainingRunSummary {
  run_id: string;
  status: string;
  current_epoch?: number;
  best_metric?: number;
  output_dir?: string;
  config_summary?: Record<string, unknown>;
}

export interface MetricRow {
  epoch?: number;
  step?: number;
  [metric: string]: number | string | undefined;
}

export const trainingApi = {
  validate: (config: unknown) =>
    postJson<{ valid: boolean; issues: string[] }>("/api/training/validate", { config }),

  launch: (config: unknown, output_dir: string) =>
    postJson<{ run_id?: string; [k: string]: unknown }>("/api/training/launch", {
      config,
      output_dir,
    }),

  listRuns: () => getJson<{ runs: TrainingRunSummary[] }>("/api/training/runs"),

  getRun: (run_id: string) => getJson<TrainingRunSummary>(`/api/training/runs/${run_id}`),

  getMetrics: (project_root: string, run_id: string) =>
    getJson<{ metrics: MetricRow[]; exists: boolean }>(
      `/api/training/runs/${run_id}/metrics?project_root=${encodeURIComponent(project_root)}`,
    ),

  compare: (experiment_ids: string[]) =>
    postJson<unknown>("/api/training/compare", { experiment_ids }),

  registerModel: (body: {
    project_path: string;
    model_name: string;
    checkpoint_path: string;
    tag?: string | null;
    metadata?: Record<string, unknown> | null;
  }) => postJson<unknown>("/api/training/register_model", body),
};

export function openTrainingStream(
  project_root: string,
  run_id: string,
  onMessage: (msg: { type: string; run_id: string; row?: MetricRow; status?: unknown }) => void,
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/api/training/runs/${encodeURIComponent(
    run_id,
  )}/stream?project_root=${encodeURIComponent(project_root)}`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      const m = JSON.parse(ev.data);
      onMessage(m);
    } catch {
      /* ignore */
    }
  };
  return () => ws.close();
}
