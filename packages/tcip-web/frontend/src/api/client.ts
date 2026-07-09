/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import type {
  Box,
  Detection,
  DatasetSelection,
  GuiState,
  ImageLabels,
  MatchesResponse,
  PolygonShape,
  PredictionReference,
} from "@/store/types";

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return (await resp.json()) as T;
}

function q(params: Record<string, string | number | boolean | null | undefined>) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined) continue;
    u.set(k, String(v));
  }
  return u.toString();
}

/** Per-file label mtimes (ns) used as optimistic-concurrency version tokens. */
export interface Mtimes {
  detect: number | null;
  segment: number | null;
}

export type LoadedLabels = ImageLabels & { base_mtimes: Mtimes };

export interface SaveLabelsBody {
  image_path: string;
  detect_path?: string | null;
  segment_path?: string | null;
  boxes: Box[];
  polygons: PolygonShape[];
  project_root?: string | null;
  /** Echo the loaded mtimes so the backend can 409 a stale (lost-update) write. */
  base_mtimes?: Mtimes | null;
}

export type SaveResult = { status: "ok"; base_mtimes: Mtimes } | { status: "conflict" };

export const api = {
  state: (): Promise<GuiState> => call("/api/state"),

  dataset: {
    tree: (dataset_root: string) =>
      call<{
        dataset_root: string;
        dates_with_images: string[];
        annotation_types: string[];
        model_names: string[];
      }>(`/api/dataset/tree?${q({ dataset_root })}`),

    listImages: (dataset_root: string, date: string) =>
      call<{ images: string[]; count: number; dataset_root: string; date: string }>(
        `/api/dataset/images?${q({ dataset_root, date })}`,
      ),

    select: (body: {
      project_root: string;
      dataset_root: string;
      annotation_type?: string | null;
      date?: string | null;
      model_name?: string | null;
    }) =>
      call<{ status: string; selection: DatasetSelection }>("/api/dataset/select", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  images: {
    url: (path: string, max_width?: number) => `/api/images?${q({ path, max_width })}`,

    dimensions: (path: string) =>
      call<{ path: string; width: number; height: number }>(
        `/api/images/dimensions?${q({ path })}`,
      ),
  },

  annotate: {
    load: (image_path: string, detect_path?: string | null, segment_path?: string | null) =>
      call<LoadedLabels>(`/api/annotate/labels?${q({ image_path, detect_path, segment_path })}`),

    // Not routed through call(): a 409 (the label file changed underneath the
    // client) is an expected outcome the caller resolves by reloading, not an error.
    save: async (body: SaveLabelsBody): Promise<SaveResult> => {
      const resp = await fetch("/api/annotate/labels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.status === 409) return { status: "conflict" };
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
      }
      const data = (await resp.json()) as { base_mtimes: Mtimes };
      return { status: "ok", base_mtimes: data.base_mtimes };
    },

    openImage: (body: {
      image_path: string;
      image_index?: number;
      scale?: number;
      offset_x?: number;
      offset_y?: number;
      mode?: string;
      pred_reference?: PredictionReference | null;
    }) =>
      call<{ status: string; image_path: string }>("/api/annotate/open", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  review: {
    // `signal` lets the caller cancel an in-flight recompute so a slower earlier
    // response can't land after (and clobber) a newer one when sliders are dragged.
    matches: (
      body: {
        project_root: string;
        image_name: string;
        image_path: string;
        gt_detect_path?: string | null;
        gt_segment_path?: string | null;
        pred_detect_path?: string | null;
        pred_segment_path?: string | null;
        iou_threshold?: number;
        conf_threshold?: number;
        filter_type?: string;
        filter_class?: string | number;
        status_filter?: string;
      },
      signal?: AbortSignal,
    ) =>
      call<MatchesResponse>("/api/review/matches", {
        method: "POST",
        body: JSON.stringify(body),
        signal,
      }),

    action: (body: {
      project_root: string;
      image_name: string;
      image_path: string;
      gt_detect_path?: string | null;
      gt_segment_path?: string | null;
      pred_detect_path?: string | null;
      pred_segment_path?: string | null;
      det_type: string;
      class_id: number;
      conf?: number | null;
      iou?: number | null;
      gt_type?: string | null;
      gt_idx?: number | null;
      pred_type?: string | null;
      pred_idx?: number | null;
      bbox: [number, number, number, number];
      action: "accepted" | "rejected" | "edited";
      iou_threshold?: number;
      conf_threshold?: number;
    }) =>
      call<{ status: string; image_status: MatchesResponse["image_status"] }>(
        "/api/review/action",
        {
          method: "POST",
          body: JSON.stringify(body),
        },
      ),

    markComplete: (project_root: string, image_name: string) =>
      call<{ status: string; image_status: MatchesResponse["image_status"] }>(
        "/api/review/mark_complete",
        {
          method: "POST",
          body: JSON.stringify({ project_root, image_name }),
        },
      ),

    backupLabels: (project_root: string, label_dirs: string[]) =>
      call<{ status: string; labels_backed_up: boolean }>("/api/review/backup_labels", {
        method: "POST",
        body: JSON.stringify({ project_root, label_dirs }),
      }),

    saveGt: (body: {
      project_root: string;
      image_name: string;
      image_path: string;
      detect_path?: string | null;
      segment_path?: string | null;
      boxes: Box[];
      polygons: PolygonShape[];
    }) =>
      call<{ status: string }>("/api/review/save_gt", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    imageStatus: (project_root: string, image_name: string) =>
      call<{ status: string }>(`/api/review/image_status?${q({ project_root, image_name })}`),

    materialize: (body: {
      project_root: string;
      source_images_dir: string;
      output_dir: string;
      experiment_id?: string;
      include_hard_negatives?: boolean;
      only_completed?: boolean;
      copy_files?: boolean;
    }) =>
      call<MaterializeResult>("/api/review/materialize", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    launchQueue: (body: {
      project_root: string;
      checkpoint_path: string;
      images_dir: string;
      method?: string;
      task?: string;
      budget?: number;
      skip_reviewed?: boolean;
    }) =>
      call<{ status: string; job_id: string }>("/api/review/queue/launch", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    getQueue: (job_id: string) => call<ReviewQueueJob>(`/api/review/queue/${job_id}`),
  },
};

export interface MaterializeResult {
  positive: number;
  hard_negative: number;
  total_boxes: number;
  output_dir: string;
  manifest?: string;
  experiment_id?: string;
  [k: string]: unknown;
}

export interface ReviewQueueEntry {
  image: string;
  score: number;
}

export interface ReviewQueueResult {
  method: string;
  task: string;
  total_candidates: number;
  reviewed_skipped: number;
  selected_count: number;
  queue: ReviewQueueEntry[];
}

export interface ReviewQueueJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  error: string | null;
  result: ReviewQueueResult | Record<string, never>;
}

export type { Detection };
