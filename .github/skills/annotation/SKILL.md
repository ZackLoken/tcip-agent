---
name: annotation
description: "Annotation and review workflows for TCIP's native per-image JSON labels, with import/export to YOLO, COCO, PASCAL VOC, and LabelMe. Covers SAM-assisted labeling, review cycles with IoU matching, active learning scoring, and quality metrics. Load when labeling or reviewing image annotations, scoring unlabeled images for active learning, running SAM-assisted labeling, or preparing/QCing training data."
---

# Annotation Workflow

## Canonical format — per-image JSON with provenance

The on-disk default for both GT and predictions is **one per-image, COCO-shaped `.json`**
(`tcip_annotation.json_io`), carrying `created_by` / `created_at` / `accepted_by` /
`accepted_at` provenance per object. `stage_proposals`, `accept_candidates`, and
`export_predictions` all read/write this schema; a dataset-level COCO training set is
assembled from these per-image files (`datasets.py`'s `to_coco_dataset`), not authored
directly. An unspecified format resolves to `.json` (`dataset_layout.py`'s `label_ext()`).

## Import/export formats

| Format | Files | Coordinates | Auto-detected by |
|--------|-------|------------|------------------|
| **YOLO** | One `.txt` per image | Normalized [0,1] | `.txt` extension |
| **COCO** | Single `.json` for dataset | Pixel coordinates | `.json` + content keys |
| **PASCAL VOC** | One `.xml` per image | Pixel coordinates | `.xml` extension |
| **LabelMe** | One `.json` per image | Pixel coordinates | `.json` + `"shapes"` key |

These are explicit import/export paths via `tcip_annotation.format_io`, not the default
write path. Use `detect_format()` from `format_io` to auto-detect.
Use `load_annotations` / `save_annotations` from `format_io` for format-agnostic I/O.

## Coordinate frame — upright, EXIF applied once

Every coordinate (normalized or pixel) lives in the **EXIF-upright** frame. Images are
decoded through one door — `load_image` (`image_utils.py`) / `get_image_dimensions`, both
via `auto_orient_image` — which applies the EXIF orientation exactly once, so the GUI
canvas, the model, tiling, and viz all share one pixel space. This matters most for
Orientation-6 phone/camera JPEGs whose stored frame is transposed (e.g. 5712×4284 ↔
4284×5712): denormalizing an upright-authored box against the raw sensor frame scatters
every box. Do not re-open images with a bare `PIL.Image.open` for anything coordinate-
bearing (denormalizing, cropping, drawing) — go through `load_image` so orientation isn't
applied twice or skipped.

## Stages

1. **Initial labeling** — Manual or SAM-assisted bounding box and polygon annotation
2. **Review** — IoU matching between predictions and ground truth to accept/correct/reject
3. **Active learning** — Score unlabeled images by model uncertainty to prioritize annotation effort
4. **Quality audit** — Per-class AP, coverage analysis, inter-annotator agreement

## Tools

| Tool | Purpose |
|------|---------|
| `load_annotations` | Load labels for a set of images (auto-detects format) |
| `save_annotations` | Write annotations to any supported format |
| `sam_predict` | SAM-assisted polygon generation from point/box prompts |
| `evaluate_detections` | Compute precision/recall/AP; `detail=True` adds per-detection TP/FP/FN match data |
| `push_panel_data` | Send images + annotations to the annotation or review panel |
| `prioritize_review_queue` | Rank unlabeled images by uncertainty/diversity (`strategy="informativeness"`, default), or `strategy="confidence_triage"` to partition by confidence |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set (accepted/edited → labels, rejected → hard negatives) with experiment lineage |

## SAM-Assisted Labeling

### Manual Prompting
1. User clicks a point or draws a rough box on the annotation canvas
2. `sam_predict` generates a precise polygon mask
3. Polygon is saved in the project's configured annotation format
4. Supports point prompts (positive/negative) and box prompts

### Vision-Guided Auto-Labeling (SAM as "Hands", Agent as "Eyes")

The agent can autonomously label images by using SAM for geometry generation
and its multimodal vision for classification and QA.

**Full workflow:**
1. `sam_auto_label(image_path)` → SAM generates candidate masks, renders numbered overlay
2. Agent `view_image` on overlay → identifies and classifies each candidate
3. `accept_candidates(image_path, assignments=[{candidate_id: 0, class_id: 1}, ...])` → stages
   accepted candidates as SAM predictions (`created_by="sam"`) in the predictions tree for
   human review on the Review canvas — never writes GT directly
4. Agent `view_image` on the staged result → visual QA pass

**Corrective loop (for missed objects):**
1. `visualize_grid_overlay(image_path)` → labeled grid (A1–H6) for spatial reference
2. Agent `view_image` → identifies missed regions by grid cell
3. `sam_predict(image_path, grid_cells=["B3", "D5"])` → SAM segments at those locations
4. Save new annotations via `save_annotations`

**Grid cell system:**
- 8 columns (A–H) × 6 rows (1–6) by default
- Agent references cells like "B3" or "F5" instead of pixel coordinates
- `grid_to_pixel()` in `sam_wrapper.py` converts to center pixel coords

| Tool | Role | Phase |
|------|------|-------|
| `sam_auto_label` | Generate all candidate masks | Discovery |
| `accept_candidates` | Stage classified candidates as predictions | Classification |
| `visualize_grid_overlay` | Spatial reference for corrections | Correction |
| `sam_predict(grid_cells=...)` | Targeted segmentation | Correction |
| `view_image` | Agent visual review | All phases |

## Review Protocol

1. Load ground truth with `load_annotations`
2. Load predictions (from inference or prior annotation)
3. `evaluate_detections` pairs predictions to GT by IoU (default threshold: 0.5) and returns
   aggregate TP/FP/FN; `detail=True` adds a per-detection breakdown (each TP/FP/FN tagged with
   its class id, box/polygon, IoU, and confidence)
4. Review in panel: accept correct predictions, correct errors, add missed objects

### The review channel — propose on canvas, never write GT blind

The agent must **never write ground truth the human hasn't seen**. Stage proposals to the
*predictions* tree and drive the human to review them:

- **`stage_proposals(dataset_root, model_name, date, stem, boxes)`** writes agent-proposed
  detections to `predictions/<model>/<date>/detect/<stem>.json` (per-image COCO/JSON) — the
  predictions tree, **not** `annotations/`. They render on the Review canvas as predictions for
  the human to accept/reject/edit. `model_name` is stamped as each object's `created_by`, so name
  the real producer (`sam`, `claude`, `groundingdino`, `model:<run>`) — not a generic placeholder.
  A bucket that already carries review verdicts is immutable: a stage into it is redirected to a
  fresh `<model>@r2` bucket (the response's `bucket` field is the one actually written), so a
  re-run never overwrites reviewed predictions. Pass `overwrite=True` to force in-place, which is
  still refused when verdicts exist.
- **`focus_review(project_root, dataset_root, trait, date, model_name, image_index, detection_idx,
  filter_type, iou, conf)`** drives the live Review tab straight to a model's predictions on a
  frame/detection, so the human sees exactly what you flagged (a false positive, a missed catkin)
  without hunting. The Review analog of `focus_annotate`; a soft no-op if no GUI is running.

Flow: run inference (or `stage_proposals`) → `focus_review` the human to the weakest/flagged
frames → they accept on the canvas → only then does it become GT. See
`.github/skills/delivery` for what ships after sign-off.

## Quality Metrics

- **Per-class AP** at IoU 0.5 and 0.5:0.95
- **Coverage**: fraction of images with labels
- **Empty files**: valid negative samples — never delete without asking
- **Cohen's κ**: inter-annotator agreement (if multiple annotators)

## Active Learning

`prioritize_review_queue` ranks images by model uncertainty/diversity:
- High uncertainty = model unsure = most valuable to annotate
- Supports uncertainty, diversity, and combined scoring; can skip already-reviewed images
- `prioritize_review_queue` returns a prioritized list for the annotator
