/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import { AUDIT_ENTRY_NOT_WRITTEN, asJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import { stateSocket } from "@/api/ws";
import {
  RENDER_CACHE_VERSION,
  type JobStatus,
  type ProjectSummary,
  type RemovalRequest,
  type RenameRequest,
  type ServingGrid,
} from "@/api/types.generated";
import type { CanvasStateBody } from "@/lib/canvasSync";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type {
  Annotation,
  AnnotationPayload,
  DatasetSelection,
  ImageLabels,
  Proposal,
  SubjectCompletion,
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

/** One save of the one save door: the annotations and the gestures it adjudicates beside them,
 *  by the person ``user`` names. */
export interface SaveLabelsBody {
  image_path: string;
  // Non-empty: the backend refuses a save with nowhere to write (422); resolve that locally.
  label_path: string;
  annotations: AnnotationPayload[];
  /** Echo the loaded mtime token so the backend can 409 a stale (lost-update) write. */
  base_mtime?: string | null;
  user: string;
  /** The bucket whose proposals ``accept`` and ``reject`` name by index. */
  bucket?: string | null;
  accept?: number[];
  reject?: number[];
  /** Each subject marked complete (true) or its marks withdrawn (false). */
  complete?: Record<string, boolean>;
  /** The pixel ``[x, y, w, h]`` a mark covers; the whole image when absent. */
  rect?: [number, number, number, number] | null;
  proposals_hidden?: boolean;
}

/** What a landed save answers: the new version token and the completion it left. */
interface Saved {
  base_mtime: string | null;
  completion: Record<string, SubjectCompletion>;
}

export type SaveResult =
  | ({ status: "ok" } & Saved)
  | { status: "conflict" }
  // The save committed but its audit line did not: the client heals exactly as it does on "ok";
  // message names the gap for a toast.
  | ({ status: "unrecorded"; message: string } & Saved);

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
  status: JobStatus;
  progress: number;
  error: string | null;
}

export type { ProjectSummary };

export const api = {
  projects: {
    list: () =>
      call<{
        workspace: string;
        // The id of the project the backend has open, or null.
        open_id: string | null;
        // Names a last-opened project no longer in the workspace while nothing is open.
        last_opened_problem: string | null;
        projects: ProjectSummary[];
      }>(ROUTES.getProjects),
    open: (id: string) =>
      call<{ id: string; display_name: string; path: string }>(ROUTES.postProjectsOpen, {
        method: "POST",
        body: JSON.stringify({ id }),
      }),
    remove: (body: RemovalRequest) =>
      call<{ archive_path: string; moved_to: string }>(ROUTES.postProjectsRemove, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    rename: (body: RenameRequest) =>
      call<{ id: string; display_name: string; previous_display_name: string }>(
        ROUTES.postProjectsRename,
        {
          method: "POST",
          body: JSON.stringify(body),
        },
      ),
  },

  dataset: {
    tree: (dataset_root: string) =>
      call<{
        dataset_root: string;
        dates_with_images: string[];
        subjects: string[];
        subjects_by_date: Record<string, string[]>;
        // date -> each published bucket's name (its path under predictions/) -> its directory.
        // Index it; never reassemble the path here.
        prediction_dirs: Record<string, Record<string, string>>;
        // The first date's labels that would not read, naming the file; the tree still lists
        // every other date.
        label_problem: string | null;
      }>(`${ROUTES.getDatasetTree}?${q({ dataset_root })}`),

    select: (body: {
      dataset_root: string;
      subject?: string | null;
      date?: string | null;
      predictions_dir?: string | null;
    }) =>
      call<{
        status: string;
        selection: DatasetSelection;
        // Advisory: whether the resolved (subject,date) has labels / the bucket has
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
    // Not routed through call(): a 409 (another project open) resolves by resync.
    pushState: async (
      body: CanvasStateBody,
    ): Promise<{ status: string; shapes_written: boolean } | { status: "conflict" }> => {
      const resp = await fetch(ROUTES.postCanvasState, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.status === 409) {
        stateSocket.resync();
        return { status: "conflict" };
      }
      return asJson<{ status: string; shapes_written: boolean }>(resp);
    },
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

    // The region-serving grid over a raster: index its cells, never re-derive them.
    servingGrid: (path: string) =>
      call<ServingGrid>(`${ROUTES.getImagesServingGrid}?${q({ path })}`),
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
        completion: Record<string, SubjectCompletion>;
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
        completion: raw.completion,
        base_mtime: raw.base_mtime,
      };
    },

    // The chosen bucket's proposals for the image, each paired, decided and admitted server-side.
    proposals: (image_path: string, bucket: string, label_path?: string | null) =>
      call<{ bucket: string; proposals: Proposal[] }>(
        `${ROUTES.getAnnotateProposals}?${q({ image_path, bucket, label_path })}`,
      ),

    // Not routed through call(): a 409 (the label file changed underneath the
    // client) is an expected outcome the caller resolves by reloading, not an error.
    save: async (body: SaveLabelsBody): Promise<SaveResult> => {
      const resp = await fetch(ROUTES.postAnnotateLabels, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.status === 409) {
        try {
          const detail = ((await resp.json()) as { detail?: unknown })?.detail;
          const parsed =
            typeof detail === "object" && detail !== null
              ? (detail as { error?: unknown; message?: unknown; committed?: unknown })
              : null;
          if (parsed?.error === AUDIT_ENTRY_NOT_WRITTEN) {
            const committed = parsed.committed as Saved;
            return {
              status: "unrecorded",
              base_mtime: committed.base_mtime,
              completion: committed.completion,
              message: typeof parsed.message === "string" ? parsed.message : "",
            };
          }
        } catch {
          /* an unparseable 409 body stays conflict below */
        }
        return { status: "conflict" };
      }
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
      }
      const data = (await resp.json()) as Saved;
      return { status: "ok", base_mtime: data.base_mtime, completion: data.completion };
    },

    // Launch the review queue as a background job; poll its job_id via queueJob until terminal.
    launchQueue: (body: {
      checkpoint_path: string;
      images_dir: string;
      subject?: string | null;
      method?: string;
      budget?: number;
    }) =>
      call<{ status: string; job_id: string }>(ROUTES.postAnnotateQueueLaunch, {
        method: "POST",
        body: JSON.stringify(body),
      }),

    // reference_member is present only when the run was bound to a selection.
    queueJob: (jobId: string) =>
      call<{
        job_id: string;
        status: JobStatus;
        error: string | null;
        queue: { image: string; score: number; reference_member?: boolean }[];
        total_candidates: number;
        reviewed_skipped: number;
      }>(ROUTES.getAnnotateQueueByJobId(jobId)),
  },
};
