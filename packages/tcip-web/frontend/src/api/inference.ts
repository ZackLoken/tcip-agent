/** Inference + Results API helpers for the Inference and Results tabs. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";

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
  warning: string | null;
}

/** A run is named, not spelled: the backend resolves both the images dir and the prediction
 *  bucket from (dataset_root, model_name, date) through its own layout resolver. Everything the
 *  platform derives per checkpoint (conf, IoU, tile geometry, cross-tile merge) is left off. */
export interface LaunchInferenceBody {
  checkpoint_path: string;
  dataset_root: string;
  model_name: string;
  date?: string | null;
  tile?: boolean;
}

export const inferenceApi = {
  launch: (body: LaunchInferenceBody) =>
    postJson<{
      status: string;
      job_id: string;
      images_dir: string;
      output_dir: string;
      bucket_redirected: boolean;
      requested_output_dir: string | null;
    }>(ROUTES.postInferenceLaunch, body),

  listJobs: () => getJson<{ jobs: InferenceJob[] }>(ROUTES.getInferenceJobs),

  getJob: (jobId: string) => getJson<InferenceJob>(ROUTES.getInferenceJobsByJobId(jobId)),

  cancel: (jobId: string) =>
    postJson<{ job_id: string; status: string; cancel_requested: boolean }>(
      ROUTES.postInferenceJobsByJobIdCancel(jobId),
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
  const url = wsUrl(ROUTES.socketInferenceJobsByJobIdStream(jobId));
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
  // null when this date was not fully classified/observed, never a fabricated ratio.
  ratio: number | null;
}

// Milestone column names are derived from the threaded trait's own spec, not hardcoded to
// any one trait: the fixed fields below are the columns every phenology delivery carries
// regardless of trait; the trait-specific milestone/date columns (e.g. <trait>_50per_date)
// arrive as additional keys and are read generically (see ResultsTab.tsx's milestoneColumns
// helper).
export interface OnsetRow {
  plant_id: string;
  accession: string | null;
  n_dates: number;
  n_dates_unclassified: number;
  n_dates_missing_images: number;
  // Dates with a real, non-zero-detection observation: a plant can be fully classified and fully
  // observed (0 unclassified, 0 missing) while still never having detected anything, e.g. before
  // emergence; that reads as "no observations", not "valid".
  n_observed_dates: number;
  [milestoneColumn: string]: string | number | null;
}

// The inputs a phenology measurement is computed from. Every Results door takes this same shape; none
// accepts rows, so no caller-composed table can be mistaken for (or declared to be) a delivery.
export interface PhenologyRequest {
  project_root: string;
  mapping_path: string;
  predictions_by_date: Record<string, string>;
  trait: string;
  // Show provisional numbers instead of refusing. The server still marks them provisional, and it
  // never applies to a CSV.
  acknowledge_unvalidated?: boolean;
}

// Every door returns the evidence that qualifies its numbers alongside them, so no surface can
// render a phenology measurement bare.
export interface PhenologyResponse<Row> {
  rows: Row[];
  // Per-dimension reconciled state, e.g. { operating_point: "validated_held_out", classifier: "false" }.
  validated: Record<string, string>;
  // True when any dimension lacked on-disk evidence, including when the caller acknowledged it,
  // which is exactly when these numbers must not be rendered as valid.
  provisional: boolean;
  validity_detail: Record<string, unknown>;
  // False when nothing was ever classified along the trait's positive-class axis: the ratios are
  // then not a valid phenology measurement (run + validate the classifier first).
  positive_class_assessed: boolean;
  n_plants?: number;
  positive_class_id?: number | null;
}

export const resultsApi = {
  registeredModels: (project_path: string) =>
    getJson<{ models: RegisteredModel[] }>(
      `${ROUTES.getResultsModelsRegistered}?project_path=${encodeURIComponent(project_path)}`,
    ),

  // The project's own registered traits, so the Results tab resolves which trait it is
  // computing for from the project's registry instead of assuming one. Each trait's declared
  // milestone fractions come along, so the tab can tell what there is to compute for it.
  traits: (project_root: string) =>
    getJson<{
      traits: string[];
      milestone_fractions_by_trait: Record<string, number[]>;
      invalid_specs: { file: string; reason: string }[];
    }>(`${ROUTES.getResultsTraits}?project_root=${encodeURIComponent(project_root)}`),

  buildPlantMapping: (body: {
    images_root: string;
    plant_csv_paths: string[];
    dates?: string[];
    nn_tolerance_m?: number;
    persist_path?: string;
  }) =>
    postJson<{ summary: PlantMappingSummary; mapping: unknown }>(
      ROUTES.postResultsPlantMappingBuild,
      body,
    ),

  loadPlantMapping: (persist_path: string) =>
    postJson<{ mapping: unknown }>(ROUTES.postResultsPlantMappingLoad, { persist_path }),

  perPlantCurves: (body: PhenologyRequest) =>
    postJson<PhenologyResponse<PerPlantRow>>(ROUTES.postResultsPerPlantCurves, body),

  onsetDates: (body: PhenologyRequest) =>
    postJson<PhenologyResponse<OnsetRow>>(ROUTES.postResultsOnsetDates, body),

  // The server computes what it exports: this sends the inputs a phenology measurement is derived from
  // plus which computation to run, never a table of rows: sending rows would let the caller
  // control both the columns and any accompanying kind declaration, so the backend could no
  // longer trust either.
  exportCsv: async (
    body: PhenologyRequest,
    payload: "curves" | "milestones",
    filename: string,
  ): Promise<Blob> => {
    const resp = await fetch(ROUTES.postResultsExportCsv, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // acknowledge_unvalidated is deliberately dropped: it may reveal provisional numbers on
      // screen, never write them to a file that leaves the platform.
      body: JSON.stringify({ ...body, acknowledge_unvalidated: false, payload, filename }),
    });
    if (!resp.ok) {
      // Surface the server's actual reason instead of discarding the response body.
      let detail = `export_csv failed: ${resp.status}`;
      try {
        const body = (await resp.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        // response body wasn't JSON, keep the status-only message
      }
      throw new Error(detail);
    }
    return await resp.blob();
  },
};
