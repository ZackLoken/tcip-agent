/**
 * Types mirroring the Python backend's GuiState and the name-based label schema.
 * Keep in sync with packages/tcip-web/src/tcip_web/state.py and routes/{annotate,review,classes}.py.
 */

export type TabName =
  "annotate" | "review" | "training" | "tuning" | "inference" | "results" | "meta";

export type Mode = "box" | "polygon" | "point";

export interface DatasetSelection {
  project_root: string | null;
  dataset_root: string | null;
  subject: string | null;
  date: string | null;
  image_list: string[];
  current_image_index: number;
  // One file per image now holds every subject, so the label/prediction dirs carry no subject
  // or task segment: annotations/<date>/ and predictions/<model>/<date>/.
  annotations_dir: string | null;
  predictions_dir: string | null;
}

export interface ViewState {
  scale: number;
  offset_x: number;
  offset_y: number;
}

/** Dashed reference overlay shown in the Annotate tab (Review→Edit flow). Display-only:
 *  a name-based label carries its subject, but the reference itself just needs geometry. */
export interface PredictionReference {
  type: "box" | "polygon";
  coords: number[] | number[][];
  confidence: number | null;
}

export interface ReviewFilters {
  iou_threshold: number;
  conf_threshold: number;
  filter_type: "all" | "tp" | "fp" | "fn";
  filter_class: string; // a subject name or "all"
  detection_idx: number;
}

/** Per-image review completion status (from ReviewEngine.get_image_review_status). */
export type ReviewImageStatus = "not_started" | "started" | "completed";

/** Image-level Reviewed/Unreviewed navigation filter (drives which images the Review tab walks). */
export type ReviewStatusFilter = "all" | "reviewed" | "unreviewed";

export interface GuiState {
  active_tab: TabName;
  dataset: DatasetSelection;
  view: ViewState;
  mode: Mode;
  // The subject a new shape is authored for (was an integer active_class). Client-owned.
  active_subject: string | null;
  review: ReviewFilters;
  pred_reference: PredictionReference | null;
}

/* ── Label schema (name-based, one unified file per image) ───────────────── */

/** One annotation as it lives on disk / crosses the wire: a subject, an optional geometry
 *  (a box OR a polygon OR a point, or none for an image/plant-level rating), and its attribute
 *  values by name. A prediction is the same shape with ``score`` set. */
export interface Annotation {
  subject: string;
  bbox?: [number, number, number, number] | null; // [x1, y1, x2, y2], pixel
  // Every ring of a polygon, pixel: an occlusion-split instance_seg shape is genuinely more than
  // one region, and both load routes (_ann_dict in annotate.py / review.py) always send them all.
  rings?: [number, number][][] | null;
  // A single labelled location, pixel: a placed prompt or a keypoint/landmark. Geometry is a union
  // server-side (tcip_annotation.state.Annotation), so this never arrives alongside bbox/rings, and
  // a point carries no extent: never derive a box from it (see bbox_of, which refuses one).
  point?: [number, number] | null;
  attributes: Record<string, string>;
  score?: number | null;
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
}

/** The wire shape the Annotate save route accepts (mirrors AnnotationPayload in annotate.py):
 *  ``points`` for one hand-drawn/edited contour, ``rings`` for a multi-ring shape round-tripping
 *  through save. Send one or the other, never both (the backend prefers ``rings``). ``point`` is a
 *  third, separate geometry: one coordinate pair, no ring/multi-part concept. */
export interface AnnotationPayload {
  subject: string;
  bbox?: number[] | null;
  points?: number[][] | null;
  rings?: number[][][] | null;
  point?: number[] | null;
  attributes: Record<string, string>;
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
}

/* ── Canvas-local shapes (the drawing model; not synced to server) ────────── */

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  subject: string;
  attributes: Record<string, string>;
  // Provenance round-trips through the canvas: loaded shapes carry it back on save so a
  // re-save never re-stamps the original creator (new shapes omit it and get stamped).
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
}

/** A polygon on the canvas: every ring of one annotation. The drawing tool authors exactly one ring
 *  (a person draws one contour); a loaded occlusion-split shape can carry several, and all of them
 *  are drawn, hit-tested and saved together as the single annotation they are. */
export interface PolygonShape {
  rings: [number, number][][];
  subject: string;
  attributes: Record<string, string>;
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
}

/** A point on the canvas: one labelled location, the whole annotation. No extent, so no derived
 *  box and no vertices; it is placed, moved and deleted as a single coordinate. */
export interface PointShape {
  x: number;
  y: number;
  subject: string;
  attributes: Record<string, string>;
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
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
  reviewed_action: string | null;
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
