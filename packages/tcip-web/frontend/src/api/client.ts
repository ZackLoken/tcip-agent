/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import { getJson, postJson, StructuredRefusalError } from "@/api/http";
import { ROUTES } from "@/api/routes";
import { stateSocket } from "@/api/ws";
import {
  RENDER_CACHE_VERSION,
  type JobStatus,
  type LaunchPriorityQueuePayload,
  type OpenRequest,
  type ProjectSummary,
  type RemovalRequest,
  type RenameRequest,
  type ViewReads,
} from "@/api/types.generated";
import type { CanvasStateBody } from "@/lib/canvasSync";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type { PixelRect } from "@/lib/viewGeometry";
import type {
  Annotation,
  AnnotationPayload,
  DatasetSelection,
  Flag,
  FlagRequest,
  ImageLabels,
  ServedProposals,
  SubjectState,
  TabName,
} from "@/store/types";

/** Whether `e` is the backend answering 409, an outcome a caller resolves rather than an error. */
function isConflict(e: unknown): e is StructuredRefusalError {
  return e instanceof StructuredRefusalError && e.status === 409;
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

/** The unified per-image label version token (stringified mtime ns), echoed back opaquely on
 *  save. */
export type LoadedLabels = ImageLabels & { base_mtime: string | null };

/** One save of the one save door: the annotations and the gestures it adjudicates beside them,
 *  by the person ``user`` names. */
export interface SaveLabelsBody {
  image_path: string;
  annotations: AnnotationPayload[];
  /** Echo the loaded mtime token so the backend can 409 a stale (lost-update) write. */
  base_mtime?: string | null;
  user: string;
  /** The name of the bucket whose proposals ``accept`` and ``reject`` name by index. */
  bucket?: string | null;
  accept?: number[];
  reject?: number[];
  /** Each annotation the person confirms as their own call, by its position in ``annotations``. */
  confirm?: number[];
  /** Each subject marked complete (true) or its marks withdrawn (false). */
  complete?: Record<string, boolean>;
  /** The pixel ``[x, y, w, h]`` a mark covers; the whole image when absent. */
  rect?: [number, number, number, number] | null;
  proposals_hidden?: boolean;
  /** Each flag the save raises. */
  flag?: FlagRequest[];
  /** Each open flag the save resolves, by id, with the reply given. */
  resolve?: Record<string, string>;
}

/** The image's label document as the load and the save routes answer it, with the version token
 *  that names it. */
interface LabelsBody {
  image_path: string;
  img_width: number;
  img_height: number;
  annotations: Annotation[];
  completion: Record<string, SubjectState>;
  flags: Flag[];
  base_mtime: string | null;
}

/** The document split into the canvas' buckets (shared with save via labelSerde). */
function loadedLabels(raw: LabelsBody): LoadedLabels {
  return {
    image_path: raw.image_path,
    img_width: raw.img_width,
    img_height: raw.img_height,
    ...annotationsToCanvas(raw.annotations ?? []),
    completion: raw.completion,
    flags: raw.flags,
    base_mtime: raw.base_mtime,
  };
}

/** A landed save answers the document it wrote, whole, so the editor adopts content and token
 *  together. */
export type SaveResult =
  | {
      status: "ok";
      labels: LoadedLabels;
      /** Each accepted proposal's index, as a string key, to its annotation's document index. */
      accepted: Record<string, number>;
    }
  | { status: "conflict" };

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
   *  [width, height] they were read at. Those bounds describe display scale. */
  overview_size?: [number, number];
}

/** A raster's overview build, as the build/status endpoints report it: the build the server names
 *  (X-TCIP-Image-Error) when a read of a raster that can have a pyramid needs one. */
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
      getJson<{
        workspace: string;
        // The id of the project the backend has open, or null.
        open_id: string | null;
        // Names a last-opened project no longer in the workspace while nothing is open.
        last_opened_problem: string | null;
        projects: ProjectSummary[];
      }>(ROUTES.getProjects),
    open: (body: OpenRequest) =>
      postJson<{ id: string; display_name: string; path: string }>(ROUTES.postProjectsOpen, body),
    remove: (body: RemovalRequest) =>
      postJson<{ archive_path: string; moved_to: string }>(ROUTES.postProjectsRemove, body),
    rename: (body: RenameRequest) =>
      postJson<{ id: string; display_name: string; previous_display_name: string }>(
        ROUTES.postProjectsRename,
        body,
      ),
  },

  dataset: {
    tree: (dataset_root: string) =>
      getJson<{
        dataset_root: string;
        dates_with_images: string[];
        subjects: string[];
        subjects_by_date: Record<string, string[]>;
        // date -> the name of each bucket published under the dataset root for it.
        buckets_by_date: Record<string, string[]>;
        // The first date's labels that would not read, naming the document; the tree still
        // lists every other date.
        label_problem: string | null;
      }>(`${ROUTES.getDatasetTree}?${q({ dataset_root })}`),

    select: (body: {
      dataset_root: string;
      subject?: string | null;
      date: string;
      bucket?: string | null;
    }) =>
      postJson<{
        status: string;
        selection: DatasetSelection;
        // Advisory: whether the resolved (subject,date) has labels / the bucket has
        // predictions. False → the canvas will start empty (not an error).
        annotations_present?: boolean;
        predictions_present?: boolean;
        // Set when annotations_present read false because a label document would not read,
        // naming it; the selection still succeeds.
        label_problem?: string | null;
      }>(ROUTES.postDatasetSelect, body),

    // Persist the current image position so the agent (view_gui_state) sees the last
    // image the human looked at. Debounced by the caller; fire-and-forget on the FE side.
    nav: (current_image_index: number) =>
      postJson<{ status: string; current_image_index: number }>(ROUTES.postDatasetNav, {
        current_image_index,
      }),
  },

  fs: {
    // List sub-directories of `path` (omit for the top-level drives/roots view).
    list: (path?: string) => getJson<FsListing>(`${ROUTES.getFsList}?${q({ path })}`),
  },

  canvas: {
    // Live canvas-state push (heartbeat or full geometry): fire-and-forget from the tabs. A 409
    // (another project open) resolves by resync.
    pushState: async (
      body: CanvasStateBody,
    ): Promise<{ status: string; shapes_written: boolean } | { status: "conflict" }> => {
      try {
        return await postJson<{ status: string; shapes_written: boolean }>(
          ROUTES.postCanvasState,
          body,
        );
      } catch (e) {
        if (!isConflict(e)) throw e;
        stateSocket.resync();
        return { status: "conflict" };
      }
    },
  },

  images: {
    /** An image serve URL for the display `displayPixels` names (`useDisplayPixels`), and no size
     *  or encoding: the server derives what it serves from that count. x0/y0/x1/y1 (all four or
     *  none) request a half-open region of the raster, in the pixel grid of the overview `level`
     *  the server's view answer named (native when omitted). Every URL carries the render
     *  cache's own version, so a browser cache entry from before a version bump is never the
     *  response to a request built after it. */
    url: (
      path: string,
      displayPixels: number,
      opts: {
        bands?: string;
        stretch?: string;
        x0?: number;
        y0?: number;
        x1?: number;
        y1?: number;
        level?: number;
      } = {},
    ) =>
      `${ROUTES.getImages}?${q({ path, ...opts, display_pixels: displayPixels, v: RENDER_CACHE_VERSION })}`,

    // Per-band symbology plus the one fact that gates the band picker's visibility
    // (band_count > 3), never shown for a standard RGB dataset.
    bands: (path: string) => getJson<ImageBandsResponse>(`${ROUTES.getImagesBands}?${q({ path })}`),

    // Build the reduced-resolution pyramid a whole view of an oversized raster is served from.
    // One build per raster: a request for one already running joins it.
    buildOverviews: (path: string) => postJson<OverviewJob>(ROUTES.postImagesOverviews, { path }),

    overviewJob: (job_id: string) =>
      getJson<OverviewJob>(`${ROUTES.getImagesOverviewsStatus}?${q({ job_id })}`),

    // The reads the server serves a native-pixel view of a raster by, for one display.
    viewReads: (path: string, displayPixels: number, view: PixelRect) =>
      getJson<ViewReads>(
        `${ROUTES.getImagesView}?${q({ path, display_pixels: displayPixels, ...view })}`,
      ),
  },

  state: {
    // Mirror the active tab into the backend GUI state (debounced by the caller) so
    // view_gui_state reports the tab the human actually sees.
    tab: (active_tab: TabName) => postJson<{ status: string }>(ROUTES.postStateTab, { active_tab }),
  },

  annotate: {
    // Read the image's one label document, splitting the annotation list into the canvas'
    // box / polygon / point / geometry-less buckets (shared with save via labelSerde).
    load: async (image_path: string): Promise<LoadedLabels> =>
      loadedLabels(await getJson<LabelsBody>(`${ROUTES.getAnnotateLabels}?${q({ image_path })}`)),

    // The chosen bucket's proposals for the image, each paired and decided server-side, with the
    // bucket's validated operating point.
    proposals: (image_path: string, bucket: string) =>
      getJson<ServedProposals>(`${ROUTES.getAnnotateProposals}?${q({ image_path, bucket })}`),

    // A 409 (the label document changed underneath the client) is an expected outcome the
    // caller resolves, not an error.
    save: async (body: SaveLabelsBody, signal?: AbortSignal): Promise<SaveResult> => {
      try {
        const answer = await postJson<LabelsBody & { accepted: Record<string, number> }>(
          ROUTES.postAnnotateLabels,
          body,
          signal,
        );
        return { status: "ok", labels: loadedLabels(answer), accepted: answer.accepted };
      } catch (e) {
        if (!isConflict(e)) throw e;
        return { status: "conflict" };
      }
    },

    // Launch the review queue as a background job; poll its job_id via queueJob until terminal.
    launchQueue: (body: LaunchPriorityQueuePayload) =>
      postJson<{ status: string; job_id: string }>(ROUTES.postAnnotateQueueLaunch, body),

    // reference_member is present only when the run was bound to a selection.
    queueJob: (jobId: string) =>
      getJson<{
        job_id: string;
        status: JobStatus;
        error: string | null;
        queue: { image: string; score: number; reference_member?: boolean }[];
        total_candidates: number;
        reviewed_skipped: number;
      }>(ROUTES.getAnnotateQueueByJobId(jobId)),
  },
};
