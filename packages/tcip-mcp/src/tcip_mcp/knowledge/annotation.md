---
name: annotation
description: "Annotation and review workflows for TCIP's native per-image JSON labels, and the dataset-level COCO assembled from them. Covers engine-assisted auto-labeling (a method-neutral proposal seam; SAM is the built-in reference engine), review cycles with IoU matching, active learning scoring, and quality metrics. Load when labeling or reviewing image annotations, scoring unlabeled images for active learning, running engine-assisted auto-labeling, or preparing/QCing training data."
---

# Annotation Workflow

## Canonical format: per-image JSON with provenance

The on-disk default for both GT and predictions is one per-image, COCO-shaped `.json`
(`tcip_annotation.json_io`), carrying `created_by` / `created_at` / `accepted_by` /
`accepted_at` provenance per object. `stage_proposals` writes this schema without reading any
label document (its `assignments` regime reads back the proposal record `propose_annotations`
staged in a prior run, not a label file); `run_inference` reads it only
when it calibrates a confidence operating point. A dataset-level
COCO training set is assembled from these per-image files (`tcip_annotation.json_io`'s
`to_coco_dataset`, called from `pipelines/data/label_queries.py`'s `assemble_coco`), not authored
directly. An unspecified format resolves to `.json`
(`dataset_layout.py`'s `label_ext()`).

## Import/export formats

| Format | Files | Coordinates | Recognized by |
|--------|-------|------------|---------------|
| json | One `.json` per image (canonical) | Pixel coordinates | an `annotations` key, no `images`/`categories` key |
| coco | Single `.json` for the dataset | Pixel coordinates | an `images`/`categories` key |

Both are read by `format_io.load_annotations` / written by `save_annotations`; the read side is
wrapped for the agent by `annotation_tools.read_annotations`, a library call, not a tool of its
own. A missing label file
reads as no annotations; a present one either reader cannot make sense of raises
`json_io.UnreadableLabelDocument` naming the file or the malformed record's index, rather than
reading short. For json, the per-image reader (`json_io._annotations_of`) refuses: undecodable
text, a non-dict document, an `annotations` value that is not a list, a record that is not a
dict, a record with no string subject, and a record whose stored box has no positive extent. For
coco, `format_io.parse_coco_annotations` refuses on the same terms: a record whose `category_id`
will not coerce to `int`, or has no name in the document's own `categories`. An unreadable file
is not the same fact as no file.

`format_io.detect_format` raises its own `ValueError`, apart from the reader's refusals above,
for a missing path, a directory holding no label documents, or a present document that decodes
but matches neither format's shape, so a misdetected format never reads real annotations as
empty either; it also refuses the old `objects` label schema outright rather than sniffing it,
since that schema is converted once and never read in place.

A collaborator's delivery in some other schema is yours to convert: read a sample, write a
one-off converter script, and emit the canonical per-image JSON. The platform carries no built-in
importers.

## Coordinate frame: upright, EXIF applied once

Every coordinate (normalized or pixel) lives in the EXIF-upright frame. Images are
decoded through one entry point, `load_image` (`image_utils.py`) / `get_image_dimensions`, and
both orient through one shared EXIF orientation-tag read, so the GUI canvas, the model,
tiling, and viz all share one pixel space. This matters most for
Orientation-6 phone/camera JPEGs whose stored frame is transposed (e.g. 5712×4284 ↔
4284×5712): denormalizing an upright-authored box against the raw sensor frame scatters
every box. Do not re-open images with a bare `PIL.Image.open` for anything coordinate-
bearing (denormalizing, cropping, drawing); go through `load_image`.

## Stages

1. Initial labeling: manual or engine-assisted bounding box and polygon annotation
2. Review: IoU matching between predictions and ground truth to accept/correct/reject
3. Active learning: score unlabeled images by model uncertainty to prioritize annotation effort
4. Quality audit: coverage analysis

## Tools

| Tool | Purpose |
|------|---------|
| `save_annotations` | Write annotations to any supported format |
| `segment_prompt` | Engine-assisted polygon generation from point/box/grid prompts (`engine='sam'` default) |
| `push_panel_event` | Push an arbitrary event to a GUI panel over the tcip-web backend for a named `project_root`, not restricted to images/annotations; refuses when the GUI's open project does not agree |
| `prioritize_review_queue` | Rank unlabeled images by active-learning uncertainty/diversity for the next review batch |
| `materialize_review_dataset` | Turn human review verdicts into a curated training set (accepted/edited → labels, rejected → hard negatives) with experiment lineage; under a classified bucket's own recorded scope a rejected value call is never a hard negative (the model named the wrong state, not the object's absence), so it lands in `unconfirmed_negatives` instead |

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
   nothing for `stage_proposals`'s `assignments` regime to read back later
2. Agent reads the overlay with its own image-capable read tool → identifies and classifies each candidate
3. `stage_proposals(image_path, assignments=[{candidate_id: 0, subject: "leaf"}, ...])` → reads
   the proposals staged for that image's content, refusing if the image has changed since that
   run, then stages accepted candidates as predictions (`created_by=<engine>`) in the predictions
   tree for human review on the Review canvas, never writes GT directly, and through the
   verdict-guarded staging helper so a re-run never orphans recorded verdicts
4. Agent reads the staged result with its own image-capable read tool → visual QA pass

Visual QA is not optional: read what each tool actually leaves. `stage_proposals` stages
predictions and returns a box render (each accepted polygon's bounding box, not its mask outline)
to read before moving to the next image, catching a wrong class or a badly placed box.
`segment_prompt` writes nothing itself; its returned rings become annotations only once
`save_annotations` writes them, and reading that saved result is the QA pass for mask quality.
`capture_live_canvas` renders the human's live GUI canvas the same way (their own image,
viewport, and unsaved or in-progress shapes), so the agent can comment on work in progress
before they save.

Corrective loop (for missed objects):
1. `vision_tools.overlay_reference_grid(image_path)` (library call, or `tcip overlay-reference-grid`) → labeled reference grid ('A1' top-left) for spatial reference
2. Agent reads the grid overlay with its own image-capable read tool → identifies missed regions by grid cell
3. `segment_prompt(image_path, grid_cells=["B3", "D5"], tile_size=<echoed>)` → the engine segments
   at those locations
4. Save new annotations via `save_annotations`

Grid cell system:
- `vision_tools.overlay_reference_grid(image_path, tile_size=, overlap=)` renders square cells of `tile_size`
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
| `stage_proposals(assignments=...)` | Stage classified candidates as predictions | Classification |
| `vision_tools.overlay_reference_grid` (library call) | Spatial reference for corrections | Correction |
| `segment_prompt(grid_cells=...)` | Targeted segmentation | Correction |
| Agent's own image-capable read tool | Agent visual review | All phases |

## Review Protocol

1. Load ground truth with `annotation_tools.read_annotations` (library call, no MCP tool for it)
2. Load predictions (from inference or prior annotation)
3. `annotation_tools.score_predictions` (library call, or `tcip score-predictions`) pairs
   predictions to GT by IoU (default threshold: 0.5) and returns
   aggregate TP/FP/FN; `detail=True` adds a per-detection breakdown (each TP/FP/FN tagged with
   its class id, box/polygon, IoU, and confidence). On a classified bucket (predictions carrying
   the object class in `subject`) this scores the object's localization, never the classifier's
   own call; a bucket whose stamp predates the recorded scope refuses by name
4. Review in panel: accept correct predictions, correct errors, add missed objects. The recorded
   verdict action is one of `tcip_annotation.verdicts.VerdictAction`: accepted, rejected, edited,
   or swept (an explicit "checked this image, found nothing missed" attestation that mutates no
   ground truth)

### The review channel: propose on canvas, never write GT blind

The agent must never write ground truth the human hasn't seen. Stage proposals to the
*predictions* tree and drive the human to review them:

- `stage_proposals(image_path, *, assignments=None, boxes=None, polygons=None, model_name=None,
  overwrite=False)` writes model-/agent-proposed shapes to the predictions tree, not
  `annotations/`, so nothing here becomes ground truth before a human reviews it. Exactly one
  input regime per call: `assignments` reads back the candidates `propose_annotations` staged
  for this image, each a `{candidate_id, subject}` mapping, and lands them under
  `predictions/<engine>/<date>/<task>` with `created_by=<engine>`; `model_name` is refused
  alongside `assignments`, since the staged record already names the engine. `boxes`/`polygons`
  are explicit shapes an agent or another model already has in hand, with no cached record to
  read back; they require `model_name`, stamped as each object's `created_by`, and land under
  `predictions/<model_name>/<date>/<stem>.json`. Either way they render on the Review canvas as
  predictions for the human to accept/reject/edit; for the explicit regime, name the real
  producer in `model_name` (`sam`, `claude`, `groundingdino`, `model:<run>`), not a generic
  placeholder. A bucket (the prediction directory just written to, named by the engine or by
  `model_name`, not a score bin) that already carries review verdicts is immutable: a stage
  into it is redirected to a fresh `<engine>@r2` or `<model_name>@r2` bucket (the response's
  `bucket` field is the one actually written), so a re-run never overwrites reviewed
  predictions. Pass `overwrite=True` to force in-place (explicit regime only), which is still
  refused when verdicts exist.
- `focus_human_attention(tab='review', project_root, dataset_root, subject, date, model_name, image_index,
  detection_idx, filter_type, iou_threshold, conf_threshold)` drives the live Review tab straight to a model's
  predictions on a frame/detection, so the human sees exactly what you flagged (a false positive, a
  missed catkin) without hunting. The Review analog of `focus_human_attention(tab='annotate')`.
  Refuses before resolving anything when the GUI's open project (its `canvas_open_binding` record)
  does not name `project_root`: a mismatch, or no binding at all, refuses every time, naming what
  the GUI has open and how to converge on it. Once the binding agrees, a backend that is not
  running still answers `delivered: false` rather than raising.

Flow: run inference (or `stage_proposals`) → `focus_human_attention(tab='review')` the human to the weakest/flagged
frames → they accept on the canvas → only then does it become GT. See
`packages/tcip-mcp/src/tcip_mcp/knowledge/delivery.md` for what ships after sign-off.

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
  `tcip doctor <root>` runs separate checks bound to what each one reads:
  `check_negatives` reads the status store through the same seam `check_data_quality` does,
  and on a store that will not read, emits one warning and reports negatives unverified rather
  than walking any label file; `check_status_tokens` reads the raw store file directly and
  walks it regardless, flagging a status entry in a shape it cannot read and a stored
  `"complete"` whose label file holds no annotation of the confirmed subject. Never delete
  empty label files without asking.

## The coverage lattice and its grid zoom

The Annotate tab's coverage panel tracks two independent facts over a raster: which cells were
served to the browser at native resolution, and which cells have sat fully on screen at some
recorded scale. Neither is a claim about what the breeder actually looked at; whether a seen cell
counts as swept is judged against the subject's own working scale. That scale is the breeder's own
set grid zoom for the subject (`POST /api/coverage/grid_zoom`, stored per subject per dataset),
never derived from any annotation: the zoom a person draws at is not the zoom they inspect at.
There is no default zoom anywhere in this platform; with none set for a subject, `GET
/api/coverage/grid` answers `grid: null` with the reason, and the panel offers the control to set
one. A cell of that lattice is one screenful of native pixels at the set zoom
(`tcip_mcp.pipelines.reference_grid.derive_lattice_tile_size`), the same rule for a photograph and
an orthomosaic alike. An image already worked keeps whatever lattice its coverage was recorded on
until the breeder explicitly re-derives it, so a zoom change never silently invalidates earlier
sweeps. Region serving (the cell-aligned fetches a zoomed-in viewport streams in above the base
bitmap) runs on its own display-derived tiling, entirely independent of the coverage lattice or
its zoom.

## Active Learning

`prioritize_review_queue` ranks images by model uncertainty/diversity:
- High uncertainty = model unsure = most valuable to annotate
- Supports uncertainty, diversity, and combined scoring; can skip already-reviewed images
- `prioritize_review_queue` returns a prioritized list for the annotator
