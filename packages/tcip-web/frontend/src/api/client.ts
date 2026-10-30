/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import {
  RENDER_CACHE_VERSION,
  type ActiveTabPayload,
  type CaptureChoice,
  type DatasetTree,
  type ImageBands,
  type InvariancesRequest,
  type LabelDocumentAnswer,
  type LaunchPriorityQueuePayload,
  type NavRequest,
  type OpenRequest,
  type OverviewBuildPayload,
  type OverviewJob,
  type ProjectRemoval,
  type ProjectsListing,
  type RecordCorrection,
  type ProjectSummary,
  type FindAnswer,
  type FindPayload,
  type ImageStagings,
  type ImageStates,
  type ProposePayload,
  type RemovalRequest,
  type PriorityQueueJob,
  type RenameRequest,
  type OpenAnswer,
  type SavedLabels,
  type SavePayload,
  type SelectAnswer,
  type ServedProposals,
  type Staged,
  type StretchMode,
  type ViewReads,
} from "@/api/types.generated";
import type { CompositeBands } from "@/lib/bandSelection";
import type { CanvasStateBody } from "@/lib/canvasSync";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type { PixelRect } from "@/lib/viewGeometry";
import type { CanvasContent } from "@/store/types";

function q(
  params: Record<
    string,
    string | number | boolean | readonly (string | number)[] | null | undefined
  >,
) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined) continue;
    if (typeof v === "object") for (const item of v) u.append(k, String(item));
    else u.set(k, String(v));
  }
  return u.toString();
}

/** A half-open pixel region ``[x0, y0, x1, y1]``, the right and bottom edges exclusive. */
export type PixelRegion = readonly [number, number, number, number];

/** The image's label document on the canvas, its annotations split by geometry kind, with the
 *  version token a save echoes back opaquely. */
export type LoadedLabels = Omit<LabelDocumentAnswer, "annotations"> & CanvasContent;

/** The document, its annotations split into the canvas' buckets (shared with save via
 *  labelSerde). */
export function loadedLabels({ annotations, ...document }: LabelDocumentAnswer): LoadedLabels {
  return { ...document, ...annotationsToCanvas(annotations) };
}

/** A landed save's `SavedLabels`, the document it wrote decoded for the canvas, so the editor
 *  adopts content and token together. */
export type LoadedSave = Omit<SavedLabels, "document"> & { document: LoadedLabels };

/** The save answer `answer`, its document decoded for the canvas. */
export function loadedSave(answer: SavedLabels): LoadedSave {
  return { ...answer, document: loadedLabels(answer.document) };
}

export type { ProjectSummary };

export const api = {
  projects: {
    list: () => getJson<ProjectsListing>(ROUTES.getProjects),
    open: (body: OpenRequest) => postJson<OpenAnswer>(ROUTES.postProjectsOpen, body),
    remove: (body: RemovalRequest) => postJson<ProjectRemoval>(ROUTES.postProjectsRemove, body),
    rename: (body: RenameRequest) => postJson<RecordCorrection>(ROUTES.postProjectsRename, body),
    invariances: (body: InvariancesRequest) =>
      postJson<RecordCorrection>(ROUTES.postProjectsInvariances, body),
  },

  dataset: {
    tree: (dataset_root: string) =>
      getJson<DatasetTree>(`${ROUTES.getDatasetTree}?${q({ dataset_root })}`),

    select: (body: CaptureChoice) => postJson<SelectAnswer>(ROUTES.postDatasetSelect, body),

    // Persist the current image position so the agent (view_gui_state) sees the last
    // image the human looked at. Debounced by the caller; fire-and-forget on the FE side.
    nav: (current_image_index: number) =>
      postJson<null>(ROUTES.postDatasetNav, { current_image_index } satisfies NavRequest),
  },

  canvas: {
    // Live canvas-state push (heartbeat or full geometry): fire-and-forget from the tabs. Another
    // project open refuses (a `ProjectNotOpenRefusal`), which the caller resolves by resync.
    pushState: (body: CanvasStateBody) => postJson<null>(ROUTES.postCanvasState, body),
  },

  images: {
    /** An image serve URL for the display `displayPixels` names (`useDisplayPixels`), and no size
     *  or encoding: the server derives what it serves from that count. A `region` requests a
     *  half-open region of the raster, in the pixel grid of the overview `level` the server's
     *  view answer named (native when omitted). Every URL carries the render
     *  cache's own version, so a browser cache entry from before a version bump is never the
     *  response to a request built after it. */
    url: (
      path: string,
      displayPixels: number,
      opts: {
        bands?: CompositeBands;
        stretch?: StretchMode;
        region?: PixelRegion;
        level?: number;
      } = {},
    ) =>
      `${ROUTES.getImages}?${q({ path, ...opts, display_pixels: displayPixels, v: RENDER_CACHE_VERSION })}`,

    // Per-band symbology plus the one fact that gates the band picker's visibility (`plain`,
    // never shown for a frame displayed as its own pixels).
    bands: (path: string) => getJson<ImageBands>(`${ROUTES.getImagesBands}?${q({ path })}`),

    // Build the reduced-resolution pyramid a whole view of an oversized raster is served from.
    // One build per raster: a request for one already running joins it.
    buildOverviews: (body: OverviewBuildPayload) =>
      postJson<OverviewJob>(ROUTES.postImagesOverviews, body),

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
    tab: (body: ActiveTabPayload) => postJson<null>(ROUTES.postStateTab, body),
  },

  annotate: {
    // Read the image's one label document, splitting the annotation list into the canvas'
    // box / polygon / point / geometry-less buckets (shared with save via labelSerde).
    load: async (image_path: string): Promise<LoadedLabels> =>
      loadedLabels(
        await getJson<LabelDocumentAnswer>(`${ROUTES.getAnnotateLabels}?${q({ image_path })}`),
      ),

    // The chosen bucket's proposals for the image, each paired and decided server-side, with the
    // bucket's validated operating point.
    proposals: (image_path: string, bucket: string) =>
      getJson<ServedProposals>(`${ROUTES.getAnnotateProposals}?${q({ image_path, bucket })}`),

    // Every bucket an engine or the agent staged that holds a document for the image.
    stagings: (image_path: string) =>
      getJson<ImageStagings>(`${ROUTES.getAnnotateStagings}?${q({ image_path })}`),

    // What each image of the selected capture's label document says about the subject.
    states: (subject: string) =>
      getJson<ImageStates>(`${ROUTES.getAnnotateStates}?${q({ subject })}`),

    // The label document changing underneath the client refuses (a `VersionConflictRefusal`), an
    // outcome the caller resolves.
    save: async (body: SavePayload, signal?: AbortSignal): Promise<LoadedSave> =>
      loadedSave(await postJson<SavedLabels>(ROUTES.postAnnotateLabels, body, signal)),

    // Send the held points and boxes to the segmenter; the answer names the bucket it staged.
    propose: (body: ProposePayload) => postJson<Staged | null>(ROUTES.postAnnotatePropose, body),

    // Ask the detector for the boxes a phrase describes; they come back as prompts, unstaged.
    find: (body: FindPayload) => postJson<FindAnswer>(ROUTES.postAnnotateFind, body),

    // Launch the review queue as a background job; poll its job_id via queueJob until terminal.
    launchQueue: (body: LaunchPriorityQueuePayload) =>
      postJson<PriorityQueueJob>(ROUTES.postAnnotateQueueLaunch, body),

    queueJob: (jobId: string) => getJson<PriorityQueueJob>(ROUTES.getAnnotateQueueByJobId(jobId)),
  },
};
