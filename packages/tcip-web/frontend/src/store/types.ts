/**
 * The GUI state's and the label routes' shapes are generated from the backend's own models and
 * imported below; the canvas-local shapes are the editor's own drawing model.
 */

import {
  FINISHED_STATES,
  type GuiState,
  type ServedAnnotation,
  type ServedProposal,
  type SubjectState,
} from "@/api/types.generated";

export type { DatasetSelection, GuiState, SubjectState, ViewState } from "@/api/types.generated";

export type TabName = GuiState["active_tab"];

export type Mode = GuiState["mode"];

/* ── Canvas-local shapes (the drawing model; not synced to server) ────────── */

/** What every canvas shape is and carries: its subject, attribute values and crowd flag (COCO's
 *  iscrowd, false for a shape drawn here, which is one object), and, for a loaded one, the load
 *  route's authorship classification, which drives the canvas symbology and is never sent on save,
 *  and its index in the document. A geometry-less (image- or plant-level) rating is this alone. */
export type CanvasAnnotation = Pick<ServedAnnotation, "subject" | "attributes" | "iscrowd"> &
  Partial<Pick<ServedAnnotation, "authorship" | "index">>;

export type Box = CanvasAnnotation & {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

/** The box being dragged out: its two corners, in drag order, and the subject it is drawn for. */
export type DrawingBox = Pick<Box, "x1" | "y1" | "x2" | "y2" | "subject">;

/** A polygon on the canvas: every ring of one annotation. The drawing tool authors exactly one ring
 *  (a person draws one contour); a loaded occlusion-split shape can carry several, and all of them
 *  are drawn, hit-tested and saved together as the single annotation they are. */
export type PolygonShape = CanvasAnnotation & {
  rings: [number, number][][];
};

/** A point on the canvas: one labeled location, the whole annotation, placed, moved and deleted
 *  as a single coordinate. */
export type PointShape = CanvasAnnotation & {
  x: number;
  y: number;
};

/** Which overlays the Annotate canvas shows beside the labels: the proposals staged on the image,
 *  and the selected bucket's predictions. */
export interface AnnotateView {
  proposals: boolean;
  predictions: boolean;
}

/** Whether a subject in `state` on its image is finished (the backend's FINISHED_STATES). */
export function isFinished(state: SubjectState | null): boolean {
  return state !== null && (FINISHED_STATES as readonly string[]).includes(state);
}

/** The canvas' saved content, empty: each tool's shapes, and the geometry-less (image/plant-level)
 *  ratings, kept so they round-trip losslessly on save. Its keys are the content's only list. */
export const NO_CONTENT = {
  boxes: [] as Box[],
  polygons: [] as PolygonShape[],
  points: [] as PointShape[],
  imageAnnotations: [] as CanvasAnnotation[],
};

export type CanvasContent = typeof NO_CONTENT;

/** The content arrays of any record carrying them, and nothing else. */
export const contentOf = (record: CanvasContent): CanvasContent =>
  Object.fromEntries(
    Object.keys(NO_CONTENT).map((key) => [key, record[key as keyof CanvasContent]]),
  ) as CanvasContent;

/** The content array each tool's shapes live in. */
export const TOOL_ARRAY = {
  box: "boxes",
  polygon: "polygons",
  point: "points",
} as const satisfies Record<Mode, keyof CanvasContent>;

/** The content arrays the tools draw and edit: every one but the geometry-less ratings. */
export type ToolContent = Pick<CanvasContent, (typeof TOOL_ARRAY)[Mode]>;

/** One proposal a bucket offers for the image, with the bucket it was served from. */
export type Proposal = ServedProposal & { bucket: string };
