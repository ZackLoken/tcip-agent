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
  tile?: boolean;
  conf?: number;
  iou?: number;
  slice_h?: number;
  slice_w?: number;
  overlap?: number;
  // Cross-tile merge (tiled runs only): "nms" suppresses overlaps, "nmm" unions boxes split
  // across a tile seam. Only meaningful when tile=true.
  postprocess?: "nms" | "nmm";
}

export const inferenceApi = {
  launch: (body: LaunchInferenceBody) =>
    postJson<{ status: string; job_id: string }>("/api/inference/launch", body),

  listJobs: () => getJson<{ jobs: InferenceJob[] }>("/api/inference/jobs"),

  getJob: (jobId: string) => getJson<InferenceJob>(`/api/inference/jobs/${jobId}`),

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
  n_positive: number;
  n_unclassified: number;
  n_missing: number;
  // null when this date was not fully classified/observed (K4/K5) — never a fabricated ratio.
  ratio: number | null;
}

// K4/K5: milestone column names are derived from the threaded trait's own spec, not hardcoded to
// catkin — the fixed fields below are the columns every phenology delivery carries regardless of
// trait; the trait-specific milestone/date columns (e.g. catkin_05per_date) arrive as additional
// keys and are read generically (see ResultsTab.tsx's milestoneColumns helper).
export interface OnsetRow {
  plant_id: string;
  accession: string | null;
  n_datapoints: number;
  n_dates_unclassified: number;
  n_dates_missing_images: number;
  // Dates with a real, non-zero-detection observation (stage-6 review N6) — a plant can be fully
  // classified AND fully observed (0 unclassified, 0 missing) while still never having detected
  // anything, e.g. before emergence; that reads as "no observations", not "valid".
  n_observed_dates: number;
  [milestoneColumn: string]: string | number | null;
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
    trait: string;
  }) =>
    postJson<{
      rows: PerPlantRow[];
      n_plants: number;
      positive_class_id: number | null;
      // False when nothing was ever classified along the trait's positive-class axis — the ratios
      // are then not a valid bloom measurement (run + validate the classifier first).
      elongation_classified: boolean;
    }>("/api/results/per_plant_curves", body),

  onsetDates: (curves: PerPlantRow[], trait: string) =>
    postJson<{ rows: OnsetRow[] }>("/api/results/onset_dates", { curves, trait }),

  // `exportKind` states what the export IS. The backend gates on this declaration rather than
  // guessing from the rows' column names — three successive guessing rules were each defeated,
  // because the caller controls the column names (see the export_csv route's own comment).
  exportCsv: async (
    rows: unknown[],
    filename: string,
    exportKind: "phenology" | "diagnostic",
    predictions_by_date?: Record<string, string>,
  ): Promise<Blob> => {
    const resp = await fetch("/api/results/export_csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows,
        filename,
        export_kind: exportKind,
        predictions_by_date,
      }),
    });
    if (!resp.ok) {
      // K15: surface the server's actual reason instead of discarding the response body.
      let detail = `export_csv failed: ${resp.status}`;
      try {
        const body = (await resp.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        // response body wasn't JSON — keep the status-only message
      }
      throw new Error(detail);
    }
    return await resp.blob();
  },
};
