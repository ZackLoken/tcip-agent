---
name: annotation
description: "Annotation and review workflows for TCIP's native per-image JSON labels, and the dataset-level COCO assembled from them. Covers engine-assisted auto-labeling (a method-neutral proposal seam; SAM is the built-in reference engine), review cycles with IoU matching, active learning scoring, and quality metrics. Load when labeling or reviewing image annotations, scoring unlabeled images for active learning, running engine-assisted auto-labeling, or preparing/QCing training data."
---

# Annotation Workflow

## Canonical format: per-image JSON with provenance

The on-disk default for both GT and predictions is one per-image, COCO-shaped `.json`
(`tcip_annotation.json_io`), carrying `created_by` / `created_at` / `accepted_by` /
`accepted_at` provenance per object. `stage_proposals`, `accept_proposals`, and
`export_predictions` all read/write this schema; a dataset-level COCO training set is
assembled from these per-image files (`datasets.py`'s `to_coco_dataset`), not authored
directly. An unspecified format resolves to `.json` (`dataset_layout.py`'s `label_ext()`).

## Import/export formats

| Format | Files | Coordinates | Recognized by |
|--------|-------|------------|---------------|
| json | One `.json` per image (canonical) | Pixel coordinates | an `annotations` key, no `images`/`categories` key |
| coco | Single `.json` for the dataset | Pixel coordinates | an `images`/`categories` key |

Both are read by `format_io.load_annotations` / written by `save_annotations`; the agent-facing
MCP tool wrapping the read side is `read_annotations` (see Tools below). `format_io.detect_format`
reads the format off the file's own keys and raises on anything else rather than guessing: a
misdetected format reads real annotations as empty, so no answer beats a wrong one.

A collaborator's delivery in some other schema is yours to convert: read a sample, write a
converter in `scripts/`, and emit the canonical per-image JSON. The platform carries no built-in
importers, so nothing constrains you to a format someone else's tool happened to use.

## Coordinate frame: upright, EXIF applied once

Every coordinate (normalized or pixel) lives in the EXIF-upright frame. Images are
decoded through one door, `load_image` (`image_utils.py`) / `get_image_dimensions`, both
via `auto_orient_image`, which applies the EXIF orientation exactly once, so the GUI
canvas, the model, tiling, and viz all share one pixel space. This matters most for
Orientation-6 phone/camera JPEGs whose stored frame is transposed (e.g. 5712×4284 ↔
4284×5712): denormalizing an upright-authored box against the raw sensor frame scatters
every box. Do not re-open images with a bare `PIL.Image.open` for anything coordinate-
bearing (denormalizing, cropping, drawing); go through `load_image` so orientation isn't
applied twice or skipped.

## Stages

1. Initial labeling: manual or engine-assisted bounding box and polygon annotation
2. Review: IoU matching between predictions and ground truth to accept/correct/reject
3. Active learning: score unlabeled images by model uncertainty to prioritize annotation effort
4. Quality audit: coverage analysis, inter-annotator agreement

## Tools

| Tool | Purpose |
|------|---------|
| `read_annotations` | Load labels for a set of images (auto-detects format) |
| `save_annotations` | Write annotations to any supported format |
| `segment_prompt` | Engine-assisted polygon generation from point/box/grid prompts (`engine='sam'` default) |
| `score_predictions` | Score predictions vs GT (image file or dataset dir); `detail=True` adds per-detection TP/FP/FN match data |
| `push_panel_data` | Send images + annotations to the annotation or review panel |
| `prioritize_review_queue` | Rank unlabeled images by uncertainty/diversity (`strategy="informativeness"`, default), or `strategy="confidence_triage"` to partition by confidence |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set (accepted/edited → labels, rejected → hard negatives) with experiment lineage |

## Engine-assisted auto-labeling (the engine is a capability, not a fixed method)

Auto-labeling runs through a method-neutral proposal seam (`tcip_mcp.pipelines.proposal`): the
agent names an `engine` and the platform runs it. `engine='sam'` is the built-in SAM2 reference; the
agent can register another engine (`register_proposal_engine`) or pass a dotted `module:factory` it
wrote, a Grounding DINO / open-vocab detector, or a bespoke proposer, exactly the way `model_source`
lets it bring a model. Engine-specific knobs travel in `engine_params`; the candidate schema is
neutral (`candidate_id` / `bbox` / `area` / `rings` / `score`), engine signals under `engine_meta`
(`rings` is `Polygon.rings`, one contour per connected region of the proposed mask, so an
occlusion-split object stays split from proposal through accept).
That bespoke proposer may be classical-analysis-based (an OpenCV / scikit-image pipeline the agent
writes), one option among engines, not a prescribed step; its proposals are soft and prove out only
by surviving review.

Trial and compare by review: pick the engine the data justifies. Don't assume one engine; wire
two or three, propose on the same images, and let the breeder's review decide: the useful engine is
the one whose high-confidence proposals *survive review* (high accept rate, few edits). That accept
rate is the measured comparison, the same signal that gates auto-accept in the review→retrain loop.
Do not promise an engine that isn't built; SAM is the one that ships as a runnable example.

### Manual/prompted segmentation
1. User clicks a point or draws a rough box on the annotation canvas (or the agent names a grid cell)
2. `segment_prompt(image_path, points=/box=/grid_cells=, engine='sam')` returns precise polygon
   `rings`, one per connected region of the mask, so a partly-occluded object comes back whole
3. The rings are saved as one polygon annotation in the project's configured annotation format
4. Supports point prompts (positive/negative), box prompts, and grid-cell references

### Vision-guided auto-labeling (the engine is the "hands", the agent's vision the "eyes")

The agent labels images by using a proposal engine for geometry and its multimodal vision for
classification and QA.

Full workflow:
1. `propose_annotations(image_path, engine='sam')` → the engine proposes candidates, renders a
   numbered overlay. `grid_cells=[...]` (with `tile_size`, `overlap` echoed by
   `overlay_reference_grid`) restricts the pass to the named cells' bounding rect instead of the
   whole frame, useful on a large or crowded image; the engine itself never sees a region, only a
   crop, and the returned candidates are already in full-frame coordinates. On a dataset image
   this stages the run (`staged: true`) keyed by the dataset, capture date and stem, alongside the
   content identity of the pixels the engine ran on; on a path outside any dataset's `images/`
   tree the render and candidates still come back (`staged: false`, naming why), but there is
   nothing for `accept_proposals` to read back later
2. Agent `view_image` on overlay → identifies and classifies each candidate
3. `accept_proposals(image_path, assignments=[{candidate_id: 0, subject: "leaf"}, ...])` → reads
   the proposals staged for that image's content, refusing if the image has changed since that
   run, then stages accepted candidates as predictions (`created_by=<engine>`) in the predictions
   tree for human review on the Review canvas, never writes GT directly, and through the
   verdict-guarded staging helper so a re-run never orphans recorded verdicts
4. Agent `view_image` on the staged result → visual QA pass

Corrective loop (for missed objects):
1. `overlay_reference_grid(image_path)` → labeled reference grid ('A1' top-left) for spatial reference
2. Agent `view_image` → identifies missed regions by grid cell
3. `segment_prompt(image_path, grid_cells=["B3", "D5"], tile_size=<echoed>)` → the engine segments
   at those locations
4. Save new annotations via `save_annotations`

Grid cell system:
- `overlay_reference_grid(image_path, tile_size=, overlap=)` renders square cells of `tile_size`
  native pixels (omitted, it derives a legible default from the image dims) and echoes the full
  grid geometry back in every response: `tile_size`, `overlap`, `cols`, `rows`, `width`, `height`
- Agent references cells like "B3" or "F5" instead of pixel coordinates
- A cell name is meaningless without its grid, so `segment_prompt(grid_cells=...)` requires the
  explicit `tile_size` (plus `overlap` if nonzero) the overlay echoed; it refuses rather than
  assume a grid, and recomputes the identical cells through the shared reference-grid geometry
- `grid_to_pixel()` in `sam_wrapper.py` looks a cell name up in a supplied cell list and returns
  its center pixel coords

| Tool | Role | Phase |
|------|------|-------|
| `propose_annotations` | Propose candidate masks with a chosen engine, whole-frame or `grid_cells`-scoped | Discovery |
| `accept_proposals` | Stage classified candidates as predictions | Classification |
| `overlay_reference_grid` | Spatial reference for corrections | Correction |
| `segment_prompt(grid_cells=...)` | Targeted segmentation | Correction |
| `view_image` | Agent visual review | All phases |

## Review Protocol

1. Load ground truth with `read_annotations`
2. Load predictions (from inference or prior annotation)
3. `score_predictions` pairs predictions to GT by IoU (default threshold: 0.5) and returns
   aggregate TP/FP/FN; `detail=True` adds a per-detection breakdown (each TP/FP/FN tagged with
   its class id, box/polygon, IoU, and confidence)
4. Review in panel: accept correct predictions, correct errors, add missed objects

### The review channel: propose on canvas, never write GT blind

The agent must never write ground truth the human hasn't seen. Stage proposals to the
*predictions* tree and drive the human to review them:

- `stage_proposals(dataset_root, model_name, date, stem, boxes)` writes agent-proposed
  detections to `predictions/<model>/<date>/<stem>.json` (per-image COCO/JSON), the
  predictions tree, not `annotations/`. They render on the Review canvas as predictions for
  the human to accept/reject/edit. `model_name` is stamped as each object's `created_by`, so name
  the real producer (`sam`, `claude`, `groundingdino`, `model:<run>`), not a generic placeholder.
  A bucket that already carries review verdicts is immutable: a stage into it is redirected to a
  fresh `<model>@r2` bucket (the response's `bucket` field is the one actually written), so a
  re-run never overwrites reviewed predictions. Pass `overwrite=True` to force in-place, which is
  still refused when verdicts exist.
- `focus(tab='review', project_root, dataset_root, subject, date, model_name, image_index,
  detection_idx, filter_type, iou_threshold, conf_threshold)` drives the live Review tab straight to a model's
  predictions on a frame/detection, so the human sees exactly what you flagged (a false positive, a
  missed catkin) without hunting. The Review analog of `focus(tab='annotate')`; a soft no-op if no
  GUI is running.

Flow: run inference (or `stage_proposals`) → `focus(tab='review')` the human to the weakest/flagged
frames → they accept on the canvas → only then does it become GT. See
`.github/skills/delivery` for what ships after sign-off.

## Quality Metrics

- Coverage: fraction of images with labels
- Negatives: a training negative is an empty label file (`"annotations": []`) plus a human
  Complete on that image, recorded as the status token `"negative"` in
  `.tcip/state/image_status.json`, scoped to the subject. Each entry there is a record, not a bare
  token: it carries `recorded_by` and `recorded_at`, so a person's own Complete is legible against
  one a review harvest transcribed. Write through `record_image_statuses` /
  `replace_image_status_store` rather than by hand. `to_coco_dataset` silently skips an
  empty file that is not in that set, treating it as unannotated. You cannot manufacture
  negatives; writing empty label files does not create them; only the human's Complete does.
  `python scripts/doctor.py <root>` flags every empty-label/status disagreement, a status entry in
  a shape it cannot read, and a stored `"complete"` whose label file holds no annotation of the
  confirmed subject. Never delete empty label files without asking.
- Cohen's κ: inter-annotator agreement (if multiple annotators)

## Active Learning

`prioritize_review_queue` ranks images by model uncertainty/diversity:
- High uncertainty = model unsure = most valuable to annotate
- Supports uncertainty, diversity, and combined scoring; can skip already-reviewed images
- `prioritize_review_queue` returns a prioritized list for the annotator
