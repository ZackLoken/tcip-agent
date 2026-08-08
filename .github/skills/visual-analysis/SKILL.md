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
| `visualize(source="comparison", path=<image>)` | Overlay GT (green) vs predictions (red) with match stats |
| `render_failure_cases` | Grid of top-K failure cases |
| `propose_annotations` | Engine-proposed candidate masks rendered with numbered overlay (`engine='sam'` default) |
| `accept_proposals` | Stage classified candidates as predictions (created_by=<engine>) for human review |
| `overlay_reference_grid` | Labeled reference-grid overlay (square native-pixel cells); echoes its grid geometry (`tile_size`, `overlap`, `cols`, `rows`, `width`, `height`) for `segment_prompt(grid_cells=...)` |
| `capture_live_canvas` | The human's live GUI canvas: their image, viewport, and unsaved shapes in the GUI's own symbology (+ classes schema, TP/FP/FN legend on Review) |

`visualize` is one tool with a `source` of `annotations` / `predictions` / `dataset`
(it replaced the former `visualize_annotations` / `visualize_predictions` /
`visualize_dataset_sample`). All tools save renders to `.tcip/artifacts/viz/` and
return the file path.

`capture_live_canvas` is the live view: the GUI continuously pushes its canvas state
(hybrid: tiny heartbeats on pan/zoom, full display-resolved geometry on shape changes,
including unsaved edits and the in-progress drawing). The tool pings the GUI for fresh
state (`refresh=True`), renders the visible region (`crop_to_viewport`) or the full frame,
and returns the classes schema, review legend, per-tag / per-creator shape counts, and the
state's age. Use it to comment on work-in-progress before the human saves.

## Workflows

### Annotation QA

Goal: Verify annotation quality before training.

1. `scan_dataset` → understand image inventory and format
2. `visualize(source="dataset", path=folder_path, n=16)` → grid overview
3. `view_image` on the grid → assess overall annotation consistency
4. For flagged images: `visualize(source="annotations", path=image_path)` → `view_image`
5. Report: missed objects, wrong classes, sloppy box placement, inconsistent labeling

### Prediction Review

Goal: Assess model quality after inference.

1. `run_inference` or verify predictions exist
2. `render_failure_cases(predictions_dir, labels_dir, top_k=10)`
3. `view_image` on grid → categorize failure types
4. For specific failures: `visualize(source="comparison", path=image_path)` → `view_image`
5. Categorize: false positives, false negatives, localization errors, class confusion
6. Recommend corrective actions

### Training Artifact Review

Goal: Diagnose training issues from worst-case analysis.

1. `check_training_status` → verify training completed, review loss curves
2. `render_failure_cases` → surface + render failure cases → `view_image`
3. Cross-reference visual findings with `score_predictions`' per-image TP/FP/FN breakdown
   (no per-class breakdown for detection today, see the `evaluation` skill)
4. Recommend: more data, augmentation changes, architecture changes, longer training

### GT vs Prediction Comparison

Goal: Detailed per-image quality assessment.

1. `visualize(source="comparison", path=image_path, iou_threshold=0.5)` → `view_image`
2. Assess: IoU quality, missed detections, false positives, localization drift
3. Check multiple images to identify systematic patterns

## Visual Assessment Checklist

When inspecting rendered annotations/predictions, evaluate:

- [ ] Coverage: Are all visible objects annotated?
- [ ] Tightness: Do boxes closely fit the objects?
- [ ] Class correctness: Are classes assigned correctly?
- [ ] Consistency: Is the same object type labeled the same way across images?
- [ ] Edge cases: Occluded, overlapping, or ambiguous objects handled?
- [ ] Scale variation: Are small and large objects both captured?

## Rendering Details

- Images resized to the artifact bound (`display_bounds.VIZ_ARTIFACT_MAX_EDGE`, 1024px longest
  edge) for manageable renders; the reference grid's derived cell size targets this bound
- 20-class color palette consistent across all visualizations
- GT boxes: green outline. Prediction boxes: red outline
- Match lines: yellow center-to-center connections
- Labels include class name and confidence score where applicable
- Candidate masks: numbered with colored semi-transparent fills
- Grid overlay: yellow lines on the true cell boundaries (edge cells clip to the frame), with
  spreadsheet-style cell labels ('A1' top-left); labels decimate where a rendered cell is too
  small to hold one legibly

## Vision-Guided Auto-Labeling

Full workflow, tool/role table, the method-neutral engine seam, and the corrective (grid-cell) loop
live in `.github/skills/annotation` (it owns the format/write semantics; `accept_proposals` stages
to predictions, never GT directly). The visual-QA-specific angle here: after each `accept_proposals`
or `segment_prompt` call, `view_image` the staged result before moving on; catching a wrong class
or a sloppy mask before it reaches human review is cheaper than catching it after.
