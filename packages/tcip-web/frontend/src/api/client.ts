/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import { asJson } from "@/api/http";
import type { CanvasStateBody } from "@/lib/canvasSync";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type {
  Annotation,
  AnnotationPayload,
  DatasetSelection,
  ImageLabels,
  MatchesResponse,
  PredictionReference,
  ReviewImageStatus,
} from "@/store/types";

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  // One error-surfacing path shared with getJson/postJson: throw the backend's clean `detail`
  // (or "<status> <statusText>"), not a raw JSON blob, so toasts are consistent everywhere.
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

/** The unified per-image label version token (stringified mtime ns), echoed back opaquely on save.
 *  A string because the ns value exceeds 2**53: as a number, JSON.parse would round it and every
 *  save would 409. */
export type LoadedLabels = ImageLabels & { base_mtime: string | null };

export interface SaveLabelsBody {
  image_path: string;
  label_path?: string | null;
  annotations: AnnotationPayload[];
  project_root?: string | null;
  /** Echo the loaded mtime token so the backend can 409 a stale (lost-update) write. */
  base_mtime?: string | null;
  /** GUI-set annotator identity; stamped as created_by ("user:<name>") on saved GT. */
  user?: string | null;
}

export type SaveResult = { status: "ok"; base_mtime: string | null } | { status: "conflict" };

/**
 * Cap the width of the image served to the canvas. A 20MP drone frame is ~5500px wide;
 * capping bounds GPU texture memory + decode time while staying oversampled at fit-zoom
 * (narrower images are served untouched). It is a quality/perf tradeoff: at extreme
 * zoom the image softens. Raise it, or set it undefined (full-res), to tune.
 */
export const IMAGE_MAX_WIDTH = 4096;

/** One band's symbology, as `GET /api/images/bands` reports it: a declared name where the
 *  source has one (else its 0-index as a string), the sensor's own wavelength when known. */
export interface ImageBandInfo {
  name: string;
  wavelength_nm: number | null;
  dtype: string;
  min: number;
  max: number;
}

export interface ImageBandsResponse {
  band_count: number;
  bands: ImageBandInfo[];
}

export interface ProjectSummary {
  name: string;
  path: string;
  created: number;
  modified: number;
  dates: string[];
  subjects: string[];
  models: string[];
  // Per-date availability: subjects with labels / models with predictions on each date.
  // The subject/model pickers filter to these so a date with no bush labels doesn't
  // offer "bush" (which would open an empty canvas).
  subjects_by_date: Record<string, string[]>;
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
        subjects: string[];
        model_names: string[];
        subjects_by_date: Record<string, string[]>;
        models_by_date: Record<string, string[]>;
      }>(`/api/dataset/tree?${q({ dataset_root })}`),

    listImages: (dataset_root: string, date: string) =>
      call<{ images: string[]; count: number; dataset_root: string; date: string }>(
        `/api/dataset/images?${q({ dataset_root, date })}`,
      ),

    select: (body: {
      project_root: string;
      dataset_root: string;
      subject?: string | null;
      date?: string | null;
      model_name?: string | null;
    }) =>
      call<{
        status: string;
        selection: DatasetSelection;
        // Advisory: whether the resolved (subject,date) has labels / (model,date) has
        // predictions. False → the canvas will start empty (not an error).
        annotations_present?: boolean;
        predictions_present?: boolean;
      }>("/api/dataset/select", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // Persist the current image position so the agent (view_gui_state) sees the last
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
    // Live canvas-state push (heartbeat or full geometry): fire-and-forget from the tabs.
    pushState: (body: CanvasStateBody) =>
      call<{ status: string; shapes_stored: boolean }>("/api/canvas/state", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  images: {
    url: (path: string, max_width?: number, quality?: number, bands?: string, stretch?: string) =>
      `/api/images?${q({ path, max_width, quality, bands, stretch })}`,

    // Per-band symbology plus the one fact that gates the band picker's visibility
    // (band_count > 3), never shown for a standard RGB dataset.
    bands: (path: string) => call<ImageBandsResponse>(`/api/images/bands?${q({ path })}`),
  },

  annotate: {
    // Read the one unified per-image label file, splitting the annotation list into the canvas'
    // box / polygon / point / geometry-less buckets (shared with save via labelSerde).
    load: async (image_path: string, label_path?: string | null): Promise<LoadedLabels> => {
      const raw = await call<{
        image_path: string;
        img_width: number;
        img_height: number;
        annotations: Annotation[];
        base_mtime: string | null;
      }>(`/api/annotate/labels?${q({ image_path, label_path })}`);
      const { boxes, polygons, points, imageAnnotations } = annotationsToCanvas(
        raw.annotations ?? [],
      );
      return {
        image_path: raw.image_path,
        img_width: raw.img_width,
        img_height: raw.img_height,
        boxes,
        polygons,
        points,
        imageAnnotations,
        base_mtime: raw.base_mtime,
      };
    },

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
      const data = (await resp.json()) as { base_mtime: string | null };
      return { status: "ok", base_mtime: data.base_mtime };
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
        gt_path?: string | null;
        pred_path?: string | null;
        iou_threshold?: number;
        conf_threshold?: number;
        filter_type?: string;
        filter_class?: string;
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
      gt_path?: string | null;
      pred_path?: string | null;
      det_type: string;
      class_name: string;
      conf?: number | null;
      iou?: number | null;
      gt_idx?: number | null;
      pred_idx?: number | null;
      bbox: [number, number, number, number];
      // "swept": no geometry, no gt/pred index; an explicit "I checked this image for missed
      // objects and found none" attestation, distinct from "edited" (which always writes a real
      // GT box). Never mutates ground truth.
      action: "accepted" | "rejected" | "edited" | "swept";
      // Only for action "edited": the shape the user adjusted on the Review canvas.
      edited_box?: [number, number, number, number] | null;
      edited_points?: number[][] | null;
      iou_threshold?: number;
      conf_threshold?: number;
      // Active filters, so the fresh matches the server returns are scoped like the current view.
      filter_type?: string;
      filter_class?: string;
      /** GUI-set reviewer identity; stamped as accepted_by/created_by ("user:<name>") on GT. */
      user?: string | null;
    }) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Per-image annotation status after the GT write (null when unchanged), for the client to sync.
        annotation_status: "complete" | "partial" | "negative" | "unannotated" | null;
        // Fresh matches recomputed against the written GT; install these instead of re-fetching.
        matches: MatchesResponse;
      }>("/api/review/action", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    markComplete: (body: {
      project_root: string;
      image_name: string;
      gt_path?: string | null;
      // The prediction bucket loaded for this image: a confirmed negative carries zero verdicts,
      // so it has nowhere else to record which model it was reviewed against.
      pred_dir?: string | null;
      completed?: boolean;
    }) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Derived server-side from the GT file, never from a stale client snapshot.
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

    // Promote a completed review into a validation reference for its (model, trait, date). Runs the
    // same disjoint + count-bias gate the backend uses and returns an honest validated / not-yet result.
    validateReference: (body: { project_root: string; trait: string; pred_dir?: string | null }) =>
      call<{
        validated: boolean;
        reference: string | null;
        reviewed_image_count: number;
        conf: number | null;
        reason: string;
        buckets_stamped: string[];
      }>("/api/review/validate_reference", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // The prediction bucket's own generation confidence: read-only, no gate run, no stamp. Lets
    // the Review tab warn as soon as the "Conf >=" filter is raised above it, rather than
    // only after clicking "Use review as validation reference".
    generationConf: (pred_dir: string) =>
      call<{ generation_conf: number | null }>(
        `/api/review/generation_conf?${new URLSearchParams({ pred_dir }).toString()}`,
      ),

    // Batch review status + detection presence for a whole (subject, date): drives the image-level
    // Reviewed/Unreviewed nav filter and lets the tab skip images with nothing to review.
    imageStatuses: (params: {
      project_root: string;
      gt_dir?: string | null;
      pred_dir?: string | null;
    }) => {
      const qs = new URLSearchParams({ project_root: params.project_root });
      if (params.gt_dir) qs.set("gt_dir", params.gt_dir);
      if (params.pred_dir) qs.set("pred_dir", params.pred_dir);
      return call<{ statuses: Record<string, ReviewImageStatus>; detection_stems: string[] }>(
        `/api/review/image_statuses?${qs.toString()}`,
      );
    },

    // Launch the active-learning priority queue (informativeness ranking only, never the
    // confidence_triage/auto-accept-as-GT strategy, which stays agent-only) as a background job;
    // poll launchPriorityQueue's job_id via priorityQueueJob until status is a terminal value.
    launchPriorityQueue: (body: {
      project_root: string;
      checkpoint_path: string;
      images_dir: string;
      method?: string;
      budget?: number;
    }) =>
      call<{ status: string; job_id: string }>("/api/review/queue/launch", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    priorityQueueJob: (jobId: string) =>
      call<{
        job_id: string;
        status: "pending" | "running" | "completed" | "failed";
        error: string | null;
        queue: { image: string; score: number }[];
        total_candidates: number;
        reviewed_skipped: number;
      }>(`/api/review/queue/${jobId}`),
  },
};

export type { Detection } from "@/store/types";
