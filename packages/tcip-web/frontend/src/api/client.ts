/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import { asJson } from "@/api/http";
import type { CanvasStateBody } from "@/lib/canvasSync";
import type {
  Box,
  Detection,
  DatasetSelection,
  ImageLabels,
  MatchesResponse,
  PolygonShape,
  PredictionReference,
  ReviewImageStatus,
} from "@/store/types";

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  // One error-surfacing path shared with getJson/postJson: throw the backend's clean `detail`
  // (or "<status> <statusText>"), not a raw JSON blob — so toasts are consistent everywhere.
  return asJson<T>(resp);
}

function q(params: Record<string, string | number | boolean | null | undefined>) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined) continue;
    u.set(k, String(v));
  }
  return u.toString();
}

export interface FsEntry {
  name: string;
  path: string;
  is_dataset_root: boolean;
}

export interface FsListing {
  path: string;
  parent: string | null;
  is_dataset_root?: boolean;
  has_tcip?: boolean;
  entries: FsEntry[];
}

/** Per-file label version tokens (stringified mtime ns), echoed back opaquely on save.
 *  Strings because the ns value exceeds 2**53 — as a number, JSON.parse would round it
 *  and every save would 409. */
export interface Mtimes {
  detect: string | null;
  segment: string | null;
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
  /** GUI-set annotator identity; stamped as created_by ("user:<name>") on saved GT. */
  user?: string | null;
}

export type SaveResult = { status: "ok"; base_mtimes: Mtimes } | { status: "conflict" };

/**
 * Cap the width of the image served to the canvas. A 20MP drone frame is ~5500px wide;
 * capping bounds GPU texture memory + decode time while staying oversampled at fit-zoom
 * (narrower images are served untouched). It IS a quality/perf tradeoff — at extreme
 * zoom the image softens. Raise it, or set it undefined (full-res), to tune.
 */
export const IMAGE_MAX_WIDTH = 4096;

export interface ProjectSummary {
  name: string;
  path: string;
  created: number;
  modified: number;
  dates: string[];
  traits: string[];
  models: string[];
  // Per-date availability: traits with labels / models with predictions on each date.
  // The trait/model pickers filter to these so a date with no catkin labels doesn't
  // offer "catkin" (which would open an empty canvas).
  traits_by_date: Record<string, string[]>;
  models_by_date: Record<string, string[]>;
  image_count: number;
  is_active: boolean;
}

export const api = {
  projects: {
    list: () =>
      call<{ workspace: string; active: string | null; projects: ProjectSummary[] }>(
        "/api/projects",
      ),
    getActive: () => call<{ name: string | null; path: string | null }>("/api/projects/active"),
    setActive: (name: string) =>
      call<{ name: string; path: string }>("/api/projects/active", {
        method: "POST",
        body: JSON.stringify({ name }),
      }),
  },

  dataset: {
    tree: (dataset_root: string) =>
      call<{
        dataset_root: string;
        dates_with_images: string[];
        annotation_types: string[];
        model_names: string[];
        traits_by_date: Record<string, string[]>;
        models_by_date: Record<string, string[]>;
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
      call<{
        status: string;
        selection: DatasetSelection;
        // Advisory: whether the resolved (trait,date) has labels / (model,date) has
        // predictions. False → the canvas will start empty (not an error).
        annotations_present?: boolean;
        predictions_present?: boolean;
      }>("/api/dataset/select", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // Persist the current image position so the agent (get_active_context) sees the last
    // image the human looked at. Debounced by the caller; fire-and-forget on the FE side.
    nav: (current_image_index: number) =>
      call<{ status: string; current_image_index: number }>("/api/dataset/nav", {
        method: "POST",
        body: JSON.stringify({ current_image_index }),
      }),
  },

  fs: {
    // List sub-directories of `path` (omit for the top-level drives/roots view).
    list: (path?: string) => call<FsListing>(`/api/fs/list?${q({ path })}`),
  },

  canvas: {
    // Live canvas-state push (heartbeat or full geometry) — fire-and-forget from the tabs.
    pushState: (body: CanvasStateBody) =>
      call<{ status: string; shapes_stored: boolean }>("/api/canvas/state", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  images: {
    url: (path: string, max_width?: number, quality?: number) =>
      `/api/images?${q({ path, max_width, quality })}`,
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
      // Only for action "edited": the shape the user adjusted on the Review canvas.
      edited_box?: [number, number, number, number] | null;
      edited_polygon?: number[][] | null;
      iou_threshold?: number;
      conf_threshold?: number;
      // Active filters, so the fresh matches the server returns are scoped like the current view.
      filter_type?: string;
      filter_class?: string | number;
      /** GUI-set reviewer identity; stamped as accepted_by/created_by ("user:<name>") on GT. */
      user?: string | null;
    }) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Per-image annotation status after the GT write (null when unchanged), for the client to sync.
        annotation_status: "complete" | "partial" | "negative" | "unannotated" | null;
        // Fresh matches recomputed against the written GT — install these instead of re-fetching.
        matches: MatchesResponse;
      }>("/api/review/action", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    markComplete: (body: {
      project_root: string;
      image_name: string;
      gt_detect_path?: string | null;
      gt_segment_path?: string | null;
      completed?: boolean;
    }) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Derived server-side from the GT files — never from a stale client snapshot.
        annotation_status: "complete" | "partial" | "negative" | "unannotated";
      }>("/api/review/mark_complete", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    backupLabels: (project_root: string, label_dirs: string[]) =>
      call<{ status: string; files_backed_up: number }>("/api/review/backup_labels", {
        method: "POST",
        body: JSON.stringify({ project_root, label_dirs }),
      }),

    // Batch review status + detection presence for a whole (trait, date): drives the image-level
    // Reviewed/Unreviewed nav filter and lets the tab skip images with nothing to review.
    imageStatuses: (params: {
      project_root: string;
      gt_dir?: string | null;
      pred_dir?: string | null;
    }) => {
      const q = new URLSearchParams({ project_root: params.project_root });
      if (params.gt_dir) q.set("gt_dir", params.gt_dir);
      if (params.pred_dir) q.set("pred_dir", params.pred_dir);
      return call<{ statuses: Record<string, ReviewImageStatus>; detection_stems: string[] }>(
        `/api/review/image_statuses?${q.toString()}`,
      );
    },
  },
};

export type { Detection };
