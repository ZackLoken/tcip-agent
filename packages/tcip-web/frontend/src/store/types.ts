/**
 * The GUI state's shape is generated from the backend's own model (types.generated.ts). The
 * label schema below mirrors routes/{annotate,review,classes}.py.
 */

import type { ActionPayload, GuiState } from "@/api/types.generated";

export type { DatasetSelection, GuiState, ReviewFilters, ViewState } from "@/api/types.generated";

export type TabName = GuiState["active_tab"];

export type Mode = GuiState["mode"];

/** The project the backend has open, as its state envelope names it. */
export interface OpenProject {
  id: string;
  path: string;
}

/** Per-image review completion status (from ReviewEngine.get_image_review_status). */
export type ReviewImageStatus = "not_started" | "started" | "completed";

/** Image-level Reviewed/Unreviewed navigation filter (drives which images the Review tab walks). */
export type ReviewStatusFilter = "all" | "reviewed" | "unreviewed";

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

/** The facts an annotation carries through the canvas unchanged and back on save, so a re-save
 *  never re-stamps the original creator: its crowd flag (COCO's iscrowd, a region of unseparated
 *  objects, never one instance; a shape drawn here states none, which the save route reads as no
 *  crowd) and its provenance. ``accepted_by_rule`` names the validation record a rule-based
 *  admission was verified against ("<experiment_id>:<record_digest>"), set only by the Review
 *  accept that verified the claim. */
export interface CarriedFields {
  iscrowd?: boolean;
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
  accepted_by_rule?: string | null;
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

/** A review detection: an outcome (TP/FP/FN) referencing a GT and/or a prediction annotation by
 *  index. The class is named by ``class_name`` (a subject), never an integer id; the geometry to
 *  render is looked up from the referenced annotation's own bbox/rings, never inferred here. */
export interface Detection {
  det_type: "tp" | "fp" | "fn";
  class_name: string;
  conf: number | null;
  iou: number | null;
  gt_idx: number | null;
  pred_idx: number | null;
  bbox: [number, number, number, number];
  reviewed: boolean;
  reviewed_action: ActionPayload["action"] | null;
}

export interface MatchesResponse {
  img_width: number;
  img_height: number;
  n_tp: number;
  n_fp: number;
  n_fn: number;
  detections: Detection[];
  // Every GT / prediction annotation, each carrying its own geometry (bbox, rings or point).
  gt: Annotation[];
  preds: Annotation[];
  image_status: "not_started" | "started" | "completed";
  // Current detections with a stored verdict, and the current total, from review_progress;
  // n_total counts the whole image regardless of the active detection filter.
  n_reviewed: number;
  n_total: number;
  // The bucket's own resolved review scope; both null means a bare directory, nothing else.
  subject: string | null;
  attribute: string | null;
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
}
