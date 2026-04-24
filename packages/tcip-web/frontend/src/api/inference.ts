/** Inference + Results API helpers for the Inference and Results tabs. */

export interface InferenceJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  done: number;
  total: number;
  images_dir: string;
  output_dir: string;
  error: string | null;
}

export interface LaunchInferenceBody {
  checkpoint_path: string;
  images_dir: string;
  output_dir: string;
  sahi?: boolean;
  conf?: number;
  iou?: number;
  slice_h?: number;
  slice_w?: number;
  overlap?: number;
}

export const inferenceApi = {
  launch: (body: LaunchInferenceBody) =>
    fetch("/api/inference/launch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json() as Promise<{ status: string; job_id: string }>),

  listJobs: () =>
    fetch("/api/inference/jobs").then((r) => r.json() as Promise<{ jobs: InferenceJob[] }>),

  getJob: (jobId: string) =>
    fetch(`/api/inference/jobs/${jobId}`).then((r) => r.json()),

  getPreview: (jobId: string, limit = 12) =>
    fetch(`/api/inference/jobs/${jobId}/preview?limit=${limit}`).then((r) => r.json()),
};

export function openInferenceStream(
  jobId: string,
  onMessage: (msg: Record<string, unknown>) => void,
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/api/inference/jobs/${encodeURIComponent(
    jobId,
  )}/stream`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch {
      /* ignore */
    }
  };
  return () => ws.close();
}

export interface PerPlantRow {
  plant_id: string;
  accession: string | null;
  date: string;
  n_images: number;
  n_total: number;
  n_elongated: number;
  ratio: number;
}

export interface OnsetRow {
  plant_id: string;
  accession: string | null;
  n_datapoints: number;
  catkin_05per_date: string | null;
  catkin_50per_date: string | null;
  catkin_95per_date: string | null;
}

export const resultsApi = {
  registeredModels: (project_path: string) =>
    fetch(`/api/results/models/registered?project_path=${encodeURIComponent(project_path)}`).then(
      (r) => r.json() as Promise<{ models: { name: string; path: string; tag?: string }[] }>,
    ),

  buildPlantMapping: (body: {
    images_root: string;
    plant_csv_paths: string[];
    dates?: string[];
    nn_tolerance_m?: number;
    persist_path?: string;
  }) =>
    fetch("/api/results/plant_mapping/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),

  loadPlantMapping: (persist_path: string) =>
    fetch("/api/results/plant_mapping/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist_path }),
    }).then((r) => r.json()),

  perPlantCurves: (body: {
    project_root: string;
    mapping_path: string;
    predictions_by_date: Record<string, string>;
    elongation_height?: number;
  }) =>
    fetch("/api/results/per_plant_curves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json() as Promise<{ rows: PerPlantRow[]; n_plants: number }>),

  onsetDates: (curves: PerPlantRow[]) =>
    fetch("/api/results/onset_dates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ curves }),
    }).then((r) => r.json() as Promise<{ rows: OnsetRow[] }>),

  exportCsv: async (rows: unknown[], filename: string): Promise<Blob> => {
    const resp = await fetch("/api/results/export_csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows, filename }),
    });
    if (!resp.ok) throw new Error(`export_csv failed: ${resp.status}`);
    return await resp.blob();
  },
};
