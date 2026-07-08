/** Training-tab specific REST + WebSocket helpers. */

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
    fetch("/api/training/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }).then((r) => r.json()),

  launch: (config: unknown, output_dir: string) =>
    fetch("/api/training/launch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, output_dir }),
    }).then((r) => r.json()),

  listRuns: () =>
    fetch("/api/training/runs").then((r) => r.json()) as Promise<{ runs: TrainingRunSummary[] }>,

  getRun: (run_id: string) => fetch(`/api/training/runs/${run_id}`).then((r) => r.json()),

  getMetrics: (project_root: string, run_id: string) =>
    fetch(
      `/api/training/runs/${run_id}/metrics?project_root=${encodeURIComponent(project_root)}`,
    ).then((r) => r.json()) as Promise<{ metrics: MetricRow[]; exists: boolean }>,

  compare: (experiment_ids: string[]) =>
    fetch("/api/training/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_ids }),
    }).then((r) => r.json()),

  registerModel: (body: {
    project_path: string;
    model_name: string;
    checkpoint_path: string;
    tag?: string | null;
    metadata?: Record<string, unknown> | null;
  }) =>
    fetch("/api/training/register_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
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
