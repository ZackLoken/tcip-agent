/**
 * Types mirroring the Python backend's GuiState.
 * Keep in sync with packages/tcip-web/src/tcip_web/state.py.
 */

export type TabName =
  | "annotate"
  | "review"
  | "training"
  | "tuning"
  | "inference"
  | "results"
  | "meta";

export type Mode = "box" | "polygon";

export interface DatasetSelection {
  project_root: string | null;
  dataset_root: string | null;
  annotation_type: string | null;
  date: string | null;
  image_list: string[];
  current_image_index: number;
  annotations_detect_dir: string | null;
  annotations_segment_dir: string | null;
  predictions_detect_dir: string | null;
  predictions_segment_dir: string | null;
}

export interface ViewState {
  scale: number;
  offset_x: number;
  offset_y: number;
}

export interface PredictionReference {
  type: "box" | "polygon";
  coords: number[] | number[][];
  class_id: number;
  confidence: number | null;
}

export interface ReviewFilters {
  iou_threshold: number;
  conf_threshold: number;
  filter_type: "all" | "tp" | "fp" | "fn";
  filter_class: string | number;
  status_filter: "all" | "reviewed" | "not_reviewed";
  detection_idx: number;
}

export interface TrainingRun {
  run_id: string;
  status: "pending" | "running" | "completed" | "failed";
  epoch: number;
  best_fitness: number | null;
}

export interface InferenceJob {
  job_id: string;
  status: string;
  done: number;
  total: number;
}

export interface GuiState {
  active_tab: TabName;
  dataset: DatasetSelection;
  view: ViewState;
  class_names: Record<number, string>;
  class_colors: Record<number, string>;
  mode: Mode;
  active_class: number;
  review: ReviewFilters;
  pred_reference: PredictionReference | null;
  training_runs: TrainingRun[];
  active_run_id: string | null;
  inference_jobs: InferenceJob[];
}

/* ── Canvas-local types (not synced to server) ──────────────────────── */

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  class_id: number;
}

export interface PolygonShape {
  points: [number, number][];
  class_id: number;
}

export interface PredBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  class_id: number;
  confidence: number;
}

export interface PredPolygon {
  points: [number, number][];
  class_id: number;
  confidence: number;
}

export interface Detection {
  det_type: "tp" | "fp" | "fn";
  class_id: number;
  conf: number | null;
  iou: number | null;
  gt_type: "box" | "polygon" | null;
  gt_idx: number | null;
  pred_type: "box" | "polygon" | null;
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
  gt_boxes: Box[];
  gt_polygons: PolygonShape[];
  pred_boxes: PredBox[];
  pred_polygons: PredPolygon[];
  image_status: "not_started" | "started" | "completed";
}

export interface ImageLabels {
  image_path: string;
  img_width: number;
  img_height: number;
  boxes: Box[];
  polygons: PolygonShape[];
}
