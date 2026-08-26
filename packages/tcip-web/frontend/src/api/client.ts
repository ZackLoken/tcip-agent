/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import type { ImageStatus } from "@/api/classes";
import { asJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import {
  RENDER_CACHE_VERSION,
  type ActionPayload,
  type CoveragePayload,
  type CoverageRecord,
} from "@/api/types.generated";
import type { CanvasStateBody } from "@/lib/canvasSync";
import type {
  CompletenessRecord,
  CompletenessTogglePostBody,
  CoverageGridResponse,
} from "@/lib/coverage";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type {
  Annotation,
  AnnotationPayload,
  DatasetSelection,
  ImageLabels,
  MatchesResponse,
  Mode,
  PredictionReference,
  ReviewImageStatus,
  TabName,
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
  // Non-empty: the backend refuses a save with nowhere to write (422); resolve that locally.
  label_path: string;
  annotations: AnnotationPayload[];
  project_root?: string | null;
  /** Echo the loaded mtime token so the backend can 409 a stale (lost-update) write. */
  base_mtime?: string | null;
  /** GUI-set annotator identity; stamped as created_by ("user:<name>") on saved GT. */
  user?: string | null;
}

export type SaveResult = { status: "ok"; base_mtime: string | null } | { status: "conflict" };

/** One band's symbology, as `GET /api/images/bands` reports it: a declared name where the
 *  source has one (else its 0-index as a string), the sensor's own wavelength when known. */
export interface ImageBandInfo {
  name: string;
  wavelength_nm: number | null;
  dtype: string;
  min: number;
  max: number;
  /** What the band holds ("red", "alpha", and the rest), where the server read it from the file.
   *  Absent where nothing knows, which is not the same as a band with no interpretation. */
  interpretation?: string;
}

export interface ImageBandsResponse {
  band_count: number;
  bands: ImageBandInfo[];
  /** Whether the reported ranges came from part of the raster's pixels rather than all of them.
   *  Absent for the <=3-band early return, which reports no per-band stats at all. */
  sampled?: boolean;
  /** The share of the raster's pixels those stats were read from (1.0 when they are exact). */
  pixel_fraction?: number;
  /** The seed that chose the sample, so the same numbers can be reproduced. */
  seed?: number;
  /** Present when the ranges were read off an overview level instead of native pixels: the
   *  served/native resolution ratio they were read at. Those bounds describe display scale. */
  overview_scale?: number;
}

/** A raster's overview build, as the build/status endpoints report it. Without the pyramid, a
 *  raster past the server's display bound has no resolution a whole view can be served at. */
export interface OverviewJob {
  job_id: string;
  path: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  error: string | null;
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
  // Exactly one is set (the backend's site_fields never raises), so a recordless or damaged
  // project still lists.
  site: string | null;
  site_problem: string | null;
  // The first date's labels that would not read, naming the file; the project still lists.
  label_problem: string | null;
}

export const api = {
  projects: {
    list: () =>
      call<{
        workspace: string;
        active: string | null;
        active_path: string | null;
        projects: ProjectSummary[];
      }>(ROUTES.getProjects),
    setActive: (name: string) =>
      call<{ name: string; path: string }>(ROUTES.postProjectsActive, {
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
        // date -> model -> the dir that model's predictions for that date live in, resolved by
        // the backend's own layout resolver. Index it; never reassemble the path here.
        prediction_dirs: Record<string, Record<string, string>>;
        // The first date's labels that would not read, naming the file; the tree still lists
        // every other date.
        label_problem: string | null;
      }>(`${ROUTES.getDatasetTree}?${q({ dataset_root })}`),

    listImages: (dataset_root: string, date: string) =>
      call<{ images: string[]; count: number; dataset_root: string; date: string }>(
        `${ROUTES.getDatasetImages}?${q({ dataset_root, date })}`,
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
        // Set when annotations_present read false because the label document would not read,
        // naming the file; the selection still succeeds.
        label_problem?: string | null;
      }>(ROUTES.postDatasetSelect, {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // Persist the current image position so the agent (view_gui_state) sees the last
    // image the human looked at. Debounced by the caller; fire-and-forget on the FE side.
    nav: (current_image_index: number) =>
      call<{ status: string; current_image_index: number }>(ROUTES.postDatasetNav, {
        method: "POST",
        body: JSON.stringify({ current_image_index }),
      }),
  },

  fs: {
    // List sub-directories of `path` (omit for the top-level drives/roots view).
    list: (path?: string) => call<FsListing>(`${ROUTES.getFsList}?${q({ path })}`),
  },

  canvas: {
    // Live canvas-state push (heartbeat or full geometry): fire-and-forget from the tabs.
    pushState: (body: CanvasStateBody) =>
      call<{ status: string; shapes_stored: boolean }>(ROUTES.postCanvasState, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  images: {
    /** An image serve URL. The served width is the server's own display bound unless a caller
     *  names a narrower max_width; x0/y0/x1/y1 (all four or none) request a half-open
     *  native-pixel region of the raster. Every URL carries the render cache's own version, so a
     *  browser cache entry from before a version bump is never the response to a request built
     *  after it. */
    url: (
      path: string,
      opts: {
        max_width?: number;
        quality?: number;
        bands?: string;
        stretch?: string;
        x0?: number;
        y0?: number;
        x1?: number;
        y1?: number;
      } = {},
    ) => `${ROUTES.getImages}?${q({ path, ...opts, v: RENDER_CACHE_VERSION })}`,

    // Per-band symbology plus the one fact that gates the band picker's visibility
    // (band_count > 3), never shown for a standard RGB dataset.
    bands: (path: string) => call<ImageBandsResponse>(`${ROUTES.getImagesBands}?${q({ path })}`),

    // Build the reduced-resolution pyramid a whole view of an oversized raster is served from.
    // One build per raster: a request for one already running joins it.
    buildOverviews: (path: string) =>
      call<OverviewJob>(ROUTES.postImagesOverviews, {
        method: "POST",
        body: JSON.stringify({ path }),
      }),

    overviewJob: (job_id: string) =>
      call<OverviewJob>(`${ROUTES.getImagesOverviewsStatus}?${q({ job_id })}`),
  },

  coverage: {
    // The coverage lattice for one raster, cells included. The one geometry implementation lives
    // server-side: clients index the served cells and never re-derive them.
    grid: (path: string) => call<CoverageGridResponse>(`${ROUTES.getCoverageGrid}?${q({ path })}`),

    // The stored per-image record for a (subject, date) bucket; date omitted = dateless bucket.
    // The route wraps the record as {coverage}; unwrapped here so consumers get the bare record.
    get: (path: string, subject: string, date: string | null) =>
      call<{ coverage: CoverageRecord | null }>(
        `${ROUTES.getCoverage}?${q({ path, subject, date })}`,
      ).then((body) => body.coverage),

    // Union-merged server-side on a matching grid; a mismatched grid replaces the record.
    push: (body: CoveragePayload) =>
      call<{ status: string }>(ROUTES.postCoverage, { method: "POST", body: JSON.stringify(body) }),

    // Every subject's region-completeness record for the raster at `path`, so the minimap can
    // tell the active subject's attestations apart from another subject's.
    completeness: (path: string, dataset_root: string | null) =>
      call<{ by_subject: Record<string, CompletenessRecord> }>(
        `${ROUTES.getCoverageCompleteness}?${q({ path, dataset_root })}`,
      ),

    // Toggles one cell's completeness for a subject; the server stamps or clears its content
    // digest so a later edit inside the cell is told apart from an unedited attestation.
    toggleCompleteness: (body: CompletenessTogglePostBody) =>
      call<{ status: string; complete: boolean; cells_complete: string[] }>(
        ROUTES.postCoverageCompleteness,
        { method: "POST", body: JSON.stringify(body) },
      ),
  },

  state: {
    // Mirror the active tab into the backend GUI state (debounced by the caller) so
    // view_gui_state reports the tab the human actually sees.
    tab: (active_tab: TabName) =>
      call<{ status: string }>(ROUTES.postStateTab, {
        method: "POST",
        body: JSON.stringify({ active_tab }),
      }),
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
      }>(`${ROUTES.getAnnotateLabels}?${q({ image_path, label_path })}`);
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
      const resp = await fetch(ROUTES.postAnnotateLabels, {
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
      mode?: Mode;
      pred_reference?: PredictionReference | null;
    }) =>
      call<{ status: string; image_path: string }>(ROUTES.postAnnotateOpen, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  review: {
    // `signal` lets the caller cancel an in-flight recompute so a slower earlier
    // response can't land after (and clobber) a newer one when sliders are dragged.
    matches: (
      body: {
        dataset_root: string;
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
      call<MatchesResponse>(ROUTES.postReviewMatches, {
        method: "POST",
        body: JSON.stringify(body),
        signal,
      }),

    action: (body: ActionPayload) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Per-image annotation status after the GT write (null when unchanged), for the client to sync.
        annotation_status: ImageStatus | null;
        // Fresh matches recomputed against the written GT; install these instead of re-fetching.
        matches: MatchesResponse;
      }>(ROUTES.postReviewAction, {
        method: "POST",
        body: JSON.stringify(body),
      }),

    markComplete: (body: {
      dataset_root: string;
      image_name: string;
      gt_path?: string | null;
      // The prediction bucket loaded for this image: a confirmed negative carries zero verdicts,
      // so it has nowhere else to record which model it was reviewed against.
      pred_dir?: string | null;
      completed?: boolean;
      // The subject this Complete confirms; omitted, the completion is recorded with no
      // subject-scoped status derived.
      subject?: string | null;
    }) =>
      call<{
        status: string;
        image_status: MatchesResponse["image_status"];
        // Derived server-side from the GT file, scoped to subject; null when no subject was named.
        annotation_status: ImageStatus | null;
      }>(ROUTES.postReviewMarkComplete, {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // Its files land in the label directory, but its root opens the review engine: dataset-scoped.
    backupLabels: (dataset_root: string, label_dirs: string[]) =>
      call<{ status: string; files_backed_up: number }>(ROUTES.postReviewBackupLabels, {
        method: "POST",
        body: JSON.stringify({ dataset_root, label_dirs }),
      }),

    // Promote a completed review into a validation reference for its (model, trait, date). Runs the
    // same disjoint + count-bias gate the backend uses and returns an honest validated / not-yet result.
    validateReference: (body: {
      dataset_root: string;
      trait: string;
      pred_dir?: string | null;
      // The object identity this reference validates; the door refuses a request naming none.
      subject: string;
    }) =>
      call<{
        validated: boolean;
        reference: string | null;
        reviewed_image_count: number;
        conf: number | null;
        reason: string;
        buckets_stamped: string[];
      }>(ROUTES.postReviewValidateReference, {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // The prediction bucket's own generation confidence: read-only, no gate run, no stamp. Lets
    // the Review tab warn as soon as the "Conf >=" filter is raised above it, rather than
    // only after clicking "Use review as validation reference".
    generationConf: (pred_dir: string) =>
      call<{ generation_conf: number | null }>(
        `${ROUTES.getReviewGenerationConf}?${new URLSearchParams({ pred_dir }).toString()}`,
      ),

    // Batch review status + detection presence for a whole (subject, date): drives the image-level
    // Reviewed/Unreviewed nav filter and lets the tab skip images with nothing to review.
    imageStatuses: (params: {
      dataset_root: string;
      gt_dir?: string | null;
      pred_dir?: string | null;
    }) => {
      const qs = new URLSearchParams({ dataset_root: params.dataset_root });
      if (params.gt_dir) qs.set("gt_dir", params.gt_dir);
      if (params.pred_dir) qs.set("pred_dir", params.pred_dir);
      return call<{
        statuses: Record<string, ReviewImageStatus>;
        detection_stems: string[];
        unreadable: string[];
      }>(`${ROUTES.getReviewImageStatuses}?${qs.toString()}`);
    },

    // Launch the active-learning priority queue (informativeness ranking only, never the
    // confidence_triage/auto-accept-as-GT strategy, which stays agent-only) as a background job;
    // poll launchPriorityQueue's job_id via priorityQueueJob until status is a terminal value.
    launchPriorityQueue: (body: {
      dataset_root: string;
      checkpoint_path: string;
      images_dir: string;
      method?: string;
      budget?: number;
    }) =>
      call<{ status: string; job_id: string }>(ROUTES.postReviewQueueLaunch, {
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
      }>(ROUTES.getReviewQueueByJobId(jobId)),
  },
};

export type { Detection } from "@/store/types";
