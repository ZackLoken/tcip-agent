/**
 * The GUI state's shape is generated from the backend's own model (types.generated.ts). The
 * label schema below mirrors routes/annotate.py.
 */

import {
  FINISHED_STATES,
  type GuiState,
  type SubjectState,
  type VerdictAction,
} from "@/api/types.generated";

export type { DatasetSelection, GuiState, SubjectState, ViewState } from "@/api/types.generated";

export type TabName = GuiState["active_tab"];

export type Mode = GuiState["mode"];

/** The project the backend has open, as its state envelope names it. */
export interface OpenProject {
  id: string;
  path: string;
}

/* ── Label schema (name-based, one unified file per image) ───────────────── */

/** One annotation as it lives on disk / crosses the wire: a subject, an optional geometry
 *  (a box OR a polygon OR a point, or none for an image/plant-level rating), and its attribute
 *  values by name. A prediction is the same shape with ``score`` set. */
export interface Annotation extends CarriedFields {
  subject: string;
  bbox?: [number, number, number, number] | null; // [x1, y1, x2, y2], pixel
  // Every ring of a polygon, pixel: an occlusion-split instance_seg shape is genuinely more than
  // one region, and both load routes (annotation_dict in annotate.py) always send them all.
  rings?: [number, number][][] | null;
  // A single labeled location, pixel: a placed prompt or a keypoint/landmark. Geometry is a union
  // server-side (tcip_annotation.state.Annotation), so this never arrives alongside bbox/rings, and
  // a point carries no extent: never derive a box from it (see bbox_of, which refuses one).
  point?: [number, number] | null;
  attributes: Record<string, string>;
  score?: number | null;
  // Always stated by the load routes.
  iscrowd: boolean;
  // One of "person" | "tool" | "tool_accepted" | "unattributed", from the load route's
  // authorship_of. A load-response fact, never sent back on save (AnnotationPayload carries none).
  authorship?: string | null;
}

/** The fact an annotation carries through the canvas unchanged and back on save: its crowd flag
 *  (COCO's iscrowd, a region of unseparated objects, never one instance; a shape drawn here
 *  states none, which the save route reads as no crowd). Provenance is the save's own. */
export interface CarriedFields {
  iscrowd?: boolean;
}

/** The wire shape the Annotate save route accepts (mirrors AnnotationPayload in annotate.py):
 *  ``points`` for one hand-drawn/edited contour, ``rings`` for a multi-ring shape round-tripping
 *  through save. Send one or the other, never both (the backend prefers ``rings``). ``point`` is a
 *  third, separate geometry: one coordinate pair, no ring/multi-part concept. */
export interface AnnotationPayload extends CarriedFields {
  subject: string;
  bbox?: number[] | null;
  points?: number[][] | null;
  rings?: number[][][] | null;
  point?: number[] | null;
  attributes: Record<string, string>;
}

/* ── Canvas-local shapes (the drawing model; not synced to server) ────────── */

/** What every canvas shape is and carries: its subject, its attribute values, and the load
 *  route's authorship classification, which drives the canvas symbology and is never sent on
 *  save. */
interface CanvasShape extends CarriedFields {
  subject: string;
  attributes: Record<string, string>;
  authorship?: string | null;
}

export interface Box extends CanvasShape {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A polygon on the canvas: every ring of one annotation. The drawing tool authors exactly one ring
 *  (a person draws one contour); a loaded occlusion-split shape can carry several, and all of them
 *  are drawn, hit-tested and saved together as the single annotation they are. */
export interface PolygonShape extends CanvasShape {
  rings: [number, number][][];
}

/** A point on the canvas: one labeled location, the whole annotation. No extent, so no derived
 *  box and no vertices; it is placed, moved and deleted as a single coordinate. */
export interface PointShape extends CanvasShape {
  x: number;
  y: number;
}

/** Whether a subject in `state` on its image is finished (the backend's FINISHED_STATES). */
export function isFinished(state: SubjectState | null | undefined): boolean {
  return (FINISHED_STATES as readonly string[]).includes(state ?? "");
}

/** The Annotate canvas' load payload, split from the unified annotation list by geometry kind. */
export interface ImageLabels {
  image_path: string;
  img_width: number;
  img_height: number;
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  // Geometry-less (image/plant-level) ratings, kept so they round-trip losslessly on save.
  imageAnnotations: Annotation[];
  // Each subject the document holds or marks, by state; a subject absent here is unannotated.
  completion: Record<string, SubjectState>;
}

/** One proposal the chosen bucket offers for the image (GET /api/annotate/proposals): its index
 *  in the bucket's document, the annotation it pairs with, the last decision on it, and whether
 *  the bucket's assessment admits it. */
export interface Proposal extends Annotation {
  index: number;
  paired: number | null;
  decision: VerdictAction | null;
  admitted: boolean;
}
