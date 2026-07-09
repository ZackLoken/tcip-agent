/** Inference + Results API helpers for the Inference and Results tabs. */

import { getJson, postJson } from "@/api/http";

export interface RegisteredModel {
  name: string;
  checkpoint_path: string;
  tags?: string[];
}

export type InferenceStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  // Rehydrated after a restart: the job's worker thread is gone (not resumable).
  | "interrupted";

export interface InferenceJob {
  job_id: string;
  status: InferenceStatus;
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
    postJson<{ status: string; job_id: string }>("/api/inference/launch", body),

  listJobs: () => getJson<{ jobs: InferenceJob[] }>("/api/inference/jobs"),

  getJob: (jobId: string) => getJson<InferenceJob>(`/api/inference/jobs/${jobId}`),

  getPreview: (jobId: string, limit = 12) =>
    getJson<unknown>(`/api/inference/jobs/${jobId}/preview?limit=${limit}`),

  cancel: (jobId: string) =>
    postJson<{ job_id: string; status: string; cancel_requested: boolean }>(
      `/api/inference/jobs/${encodeURIComponent(jobId)}/cancel`,
      {},
    ),
};

/**
 * Open a live progress stream for an inference job, auto-reconnecting with capped
 * backoff if the socket drops mid-run. The server sends a single ``final`` frame at a
 * terminal state and closes; once seen we stop reconnecting (the job is done, not lost).
 */
export function openInferenceStream(
  jobId: string,
  onMessage: (msg: Record<string, unknown>) => void,
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/api/inference/jobs/${encodeURIComponent(
    jobId,
  )}/stream`;
  let ws: WebSocket | null = null;
  let closedByClient = false;
  let terminated = false;
  let backoff = 500;

  const connect = () => {
    if (closedByClient) return;
    ws = new WebSocket(url);
    ws.onopen = () => {
      backoff = 500;
    };
    ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "final") terminated = true;
      onMessage(msg);
    };
    ws.onclose = () => {
      if (closedByClient || terminated) return;
      const delay = backoff;
      backoff = Math.min(backoff * 2, 15_000);
      setTimeout(connect, delay);
    };
  };

  connect();
  return () => {
    closedByClient = true;
    ws?.close();
  };
}

export interface PlantMappingSummary {
  [date: string]: { n_images: number; n_mapped: number; avg_distance_m: number };
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
    getJson<{ models: RegisteredModel[] }>(
      `/api/results/models/registered?project_path=${encodeURIComponent(project_path)}`,
    ),

  buildPlantMapping: (body: {
    images_root: string;
    plant_csv_paths: string[];
    dates?: string[];
    nn_tolerance_m?: number;
    persist_path?: string;
  }) =>
    postJson<{ summary: PlantMappingSummary; mapping: unknown }>(
      "/api/results/plant_mapping/build",
      body,
    ),

  loadPlantMapping: (persist_path: string) =>
    postJson<{ mapping: unknown }>("/api/results/plant_mapping/load", { persist_path }),

  perPlantCurves: (body: {
    project_root: string;
    mapping_path: string;
    predictions_by_date: Record<string, string>;
    elongation_height?: number;
  }) => postJson<{ rows: PerPlantRow[]; n_plants: number }>("/api/results/per_plant_curves", body),

  onsetDates: (curves: PerPlantRow[]) =>
    postJson<{ rows: OnsetRow[] }>("/api/results/onset_dates", { curves }),

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
