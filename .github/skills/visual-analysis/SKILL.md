---
name: visual-analysis
description: "Visual inspection of images, annotations, predictions, and training artifacts using the agent's multimodal vision capability. Load when inspecting an image or its labels, checking predictions on the canvas, reading a rendered visualization, or verifying a result visually before acting on it."
---

# Visual Analysis

The agent can visually inspect images using `view_image` after rendering annotations/predictions with visualization tools.

## Pattern

1. Call a visualization MCP tool → returns `image_path` in the result dict
2. Use `view_image` on that path → model sees the rendered image
3. Describe findings and recommend actions

## Tools

| Tool | Purpose |
|------|---------|
| `visualize(source="annotations", path=<image>)` | Render GT labels on a single image |
| `visualize(source="predictions", path=<image>)` | Render model predictions on a single image |
| `visualize(source="dataset", path=<folder>, n=16)` | Random grid of annotated dataset samples |
| `visualize_comparison` | Overlay GT (green) vs predictions (red) with match stats |
| `visualize_worst_predictions` | Grid of top-K failure cases |
| `sam_auto_label` | SAM candidate masks rendered with numbered overlay |
| `accept_candidates` | Save classified candidates, render final result |
| `visualize_grid_overlay` | Labeled grid overlay for spatial referencing |

`visualize` is one tool with a `source` of `annotations` / `predictions` / `dataset`
(it replaced the former `visualize_annotations` / `visualize_predictions` /
`visualize_dataset_sample`). All tools save renders to `.tcip/artifacts/viz/` and
return the file path.

## Workflows

### Annotation QA

Goal: Verify annotation quality before training.

1. `load_dataset` → understand image inventory and format
2. `visualize(source="dataset", path=folder_path, n=16)` → grid overview
3. `view_image` on the grid → assess overall annotation consistency
4. For flagged images: `visualize(source="annotations", path=image_path)` → `view_image`
5. Report: missed objects, wrong classes, sloppy box placement, inconsistent labeling

### Prediction Review

Goal: Assess model quality after inference.

1. `run_inference` or verify predictions exist
2. `visualize_worst_predictions(predictions_dir, labels_dir, top_k=10)`
3. `view_image` on grid → categorize failure types
4. For specific failures: `visualize_comparison(image_path)` → `view_image`
5. Categorize: false positives, false negatives, localization errors, class confusion
6. Recommend corrective actions

### Training Artifact Review

Goal: Diagnose training issues from worst-case analysis.

1. `check_training_status` → verify training completed, review loss curves
2. `get_worst_predictions` → identify failure cases
3. `visualize_worst_predictions` → render failures → `view_image`
4. Cross-reference visual findings with per-class metrics from `evaluate_dataset`
5. Recommend: more data, augmentation changes, architecture changes, longer training

### GT vs Prediction Comparison

Goal: Detailed per-image quality assessment.

1. `visualize_comparison(image_path, iou_threshold=0.5)` → `view_image`
2. Assess: IoU quality, missed detections, false positives, localization drift
3. Check multiple images to identify systematic patterns

## Visual Assessment Checklist

When inspecting rendered annotations/predictions, evaluate:

- [ ] **Coverage** — Are all visible objects annotated?
- [ ] **Tightness** — Do boxes closely fit the objects?
- [ ] **Class correctness** — Are classes assigned correctly?
- [ ] **Consistency** — Is the same object type labeled the same way across images?
- [ ] **Edge cases** — Occluded, overlapping, or ambiguous objects handled?
- [ ] **Scale variation** — Are small and large objects both captured?

## Rendering Details

- Images resized to max 1024px longest edge for manageable renders
- 20-class color palette consistent across all visualizations
- GT boxes: green outline. Prediction boxes: red outline
- Match lines: yellow center-to-center connections
- Labels include class name and confidence score where applicable
- Candidate masks: numbered with colored semi-transparent fills
- Grid overlay: yellow lines with A1–H6 cell labels

## Vision-Guided Auto-Labeling

Goal: Autonomously label images using SAM geometry + agent classification.

1. `sam_auto_label(image_path)` → numbered candidate overlay
2. `view_image` → agent identifies objects, assigns class per candidate
3. `accept_candidates(image_path, [{candidate_id: 0, class_id: 1}, ...])` → saves
4. `view_image` on result → QA pass
5. If objects were missed: `visualize_grid_overlay(image_path)` → `view_image`
6. Identify missed regions by grid cell → `sam_predict(grid_cells=["C4"])` → save
7. Repeat until coverage is satisfactory
