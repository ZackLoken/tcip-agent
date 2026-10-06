---
name: annotation
description: "Annotation and review workflows for TCIP's native per-image label documents, and the import of an external dataset-level COCO into them. Covers engine-assisted auto-labeling (a method-neutral proposal seam over engines you name), the one editor where a person accepts, corrects or rejects proposals and marks an image complete, active learning scoring, and quality metrics. Load when labeling or reviewing image annotations, scoring unlabeled images for active learning, running engine-assisted auto-labeling, or preparing/QCing training data."
---

# Annotation Workflow

## Canonical format: per-image label document with provenance

Both GT and predictions are one per-image, COCO-shaped document (`tcip_annotation.json_io`), a
record in the database of the dataset root the image belongs to, keyed by the image's capture and
stem (a prediction by its bucket's name and stem), carrying `created_by` / `created_at` /
`accepted_by` / `accepted_at` provenance per object. Every tool addresses a document through its
image; a person reads the documents as files through `tcip dump-store`. `stage_proposals`
publishes this schema as a bucket of its own per image, its record naming the engine or agent that
proposed it (its `assignments` regime reads back the proposal record `propose_annotations` staged
in a prior run, not a label document). It is the one label document shape the platform
writes; object ground truth trains from it, beside the mask rasters and tables other tasks read.
Provenance is stamped by the save on the server from the person saving: a record unchanged since
it was stored keeps its own, any other is that person's at the save's time, and nothing a client
sends as provenance is read.

The document also carries each subject's completion marks under `complete`: the rect a person
attested complete (the whole image, or a region of an orthomosaic), who, when, the digest of that
subject's annotations at that moment, and whether proposals were hidden while they worked. A mark
holds only while its digest still names that subject's annotations, so an edit of a subject and
the loss of its marks are one write of one document; another subject's marks are untouched.
`LabelDocument.state(subject)` is the one reading of it: `complete` (marked, annotations held),
`negative` (marked, none held), `partial` (annotations, no mark) or `unannotated`.

## Reading labels, and importing an external COCO

Nothing here trains or calibrates on a dataset-level COCO file (an `images` or `categories`
key): it is never a per-image document, and the only way in is the import. An external COCO
export becomes per-image documents through `import_coco(document, dataset_root, date)`, over
images `ingest_images` already placed. `date` names the capture: the dated folder under
`images/`, or the flat `images/` root for a dataset with no dated folders. Every declared category
must be a subject the dataset's registry declares. An ordinary image is the capture's image with
exactly the document's `file_name`; a `.bandgroup` capture is tied by stem. An image with no
annotations writes nothing. Before anything is written the import reports every fault it finds
together and refuses: a malformed, repeated or unregistered category, a record the reader or the
writer refuses (a record's `image_id` and its content are checked independently, so one record
can report both), an image id that is not an integer or is listed twice, an image not in the
capture, a stated frame the image does not have, or an existing per-image document. Every
document and the import's audit event then commit together, or none does: a label placed
meanwhile refuses the whole import. A crowd region keeps its `iscrowd` flag, and a run-length mask
becomes the rings `mask_contours.mask_to_polygon_rings` extracts from it. The records keep the
provenance the COCO carried and gain none. The audit event records the COCO file's path, the
digest of the bytes read and the documents written; an import that writes no document, whether
refused or carrying no annotations, changes nothing and leaves no line.

A record carrying `iscrowd` is a region of unseparated objects, never one instance: the built-in
detection and instance heads train its region as background (they read no crowd flag, so a crowd
box handed to them would train as one object), a detection inside it is neither a true nor a
false positive at evaluation, and it counts as no object wherever a ground-truth count is formed
or an object is paired, the classifier calibration's pairing included. Every loader target keeps
it under `iscrowd`, so a `train(ctx)` of your own can act on it.

The per-image document is read by `json_io.read_label_document`, wrapped for the agent by
`annotation_tools.read_annotations`, a library call, not a tool of its own, and written by
`save_annotations`. A missing label document reads as no annotations; a present one the reader
cannot make sense of raises `json_io.UnreadableLabelDocumentError` naming the document or the malformed
record's index, rather than reading short: a stored value that does not decode, a non-dict
document, a completion mark missing any of its fields, an `annotations` value that is not a list, a record
that is not a dict,
a record with no string subject, and a record carrying a value that is not what the schema states,
whichever geometry the record resolves to (a `bbox` that is not four numbers or has no positive
extent, a `segmentation` that is not rings of three or more points, a `point` that is not two
numbers, a `score` that is not a number, an attribute value that is not a non-empty name, an
`iscrowd` that is not 0, 1, true or false). An absent or `null` value reads as absent, for every
optional field. A shape `save_annotations` or the Annotate tab saves is translated into this
record and decoded by the same decoder, so it is refused for the same values. The import's COCO
reader, `format_io.parse_coco_annotations`, names each record's subject from the document's own
`categories` and hands each record to that same decoder. An unreadable document is not the same
fact as no document.

A collaborator's delivery in a schema other than COCO is yours to convert: read a sample, write a
one-off converter script that emits a COCO document, and import it. COCO is the one built-in
import.

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
2. Review: a bucket's predictions shown as proposals on the Annotate canvas, each accepted,
   corrected or rejected, and the image marked complete
3. Active learning: score unlabeled images by model uncertainty to prioritize annotation effort

## Tools

| Tool | Purpose |
|------|---------|
| `save_annotations` | Write an image's per-image label document |
| `import_coco` | Convert an external dataset-level COCO into per-image label documents |
| `propose_annotations` | Run a named proposal engine over an image, whole-frame or over named grid cells |
| `push_panel_event` | Push an arbitrary event to a GUI panel over the tcip-web backend for this server's project, not restricted to images/annotations; delivered only while the GUI has that project open |
| `prioritize_review_queue` | Rank images by active-learning uncertainty/diversity for the next review batch, skipping those marked finished for a subject |

## Engine-assisted auto-labeling (the engine is a capability, not a fixed method)

Auto-labeling runs through a method-neutral proposal seam (`tcip_mcp.pipelines.proposal`): the
agent names an `engine` and the platform runs it. No engine ships built in: name one registered
(`register_proposal_engine`) or pass a dotted `module:factory` you wrote, a SAM-family segmenter, a
Grounding DINO / open-vocab detector, or a bespoke proposer, exactly the way `model_source` lets
you bring a model. An empty or unregistered name refuses, listing the registered ones. Engine-specific knobs travel in `engine_params`; the candidate schema is
neutral (`candidate_id` / `bbox` / `area` / `rings` / `score`), engine signals under `engine_meta`
(`rings` is `Polygon.rings`, one contour per connected region of the proposed mask, so an
occlusion-split object stays split from proposal through accept).
That bespoke proposer may be classical-analysis-based (an OpenCV / scikit-image pipeline the agent
writes), one option among engines, not a prescribed step; its proposals are soft and prove out only
by surviving review.

Trial and compare by review: pick the engine the data justifies. Don't assume one engine; wire
two or three, propose on the same images, and let the breeder's review decide: the useful engine is
the one whose high-confidence proposals *survive review* (high accept rate, few edits). That accept
rate, read from the verdict shards the editor writes, is the measured comparison. Do not promise
an engine that isn't wired.

### Vision-guided auto-labeling (the engine is the "hands", the agent's vision the "eyes")

The agent labels images by using a proposal engine for geometry and its multimodal vision for
classification and QA.

Full workflow:
1. `propose_annotations(image_path, engine=<name>)` → the engine proposes candidates, renders a
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
   tree for a person to accept or reject on the Annotate canvas; it never writes GT directly
4. Agent reads the staged result with its own image-capable read tool → visual QA pass

Visual QA is not optional: read what each tool actually leaves. `stage_proposals` stages
predictions and returns a box render (each accepted polygon's bounding box, not its mask outline)
to read before moving to the next image, catching a wrong class or a badly placed box.
`capture_live_canvas` renders the human's live GUI canvas the same way (their own image,
viewport, and unsaved or in-progress shapes), so the agent can comment on work in progress
before they save.

Corrective loop (for missed objects):
1. `vision_tools.overlay_reference_grid(image_path)` (library call, or `tcip overlay-reference-grid`) → labeled reference grid ('A1' top-left) for spatial reference
2. Agent reads the grid overlay with its own image-capable read tool → identifies missed regions by grid cell
3. `propose_annotations(image_path, engine=<name>, grid_cells=["B3", "D5"], tile_size=<echoed>)`
   → the engine proposes over those cells' bounding rect
4. Stage the right candidates with `stage_proposals(assignments=...)` for the person to accept

Grid cell system:
- `vision_tools.overlay_reference_grid(image_path, tile_size=, overlap=)` renders square cells of `tile_size`
  native pixels (omitted, it derives a legible default from the image dims) and echoes the full
  grid geometry back in every response: `tile_size`, `overlap`, `cols`, `rows`, `width`, `height`
- Agent references cells like "B3" or "F5" instead of pixel coordinates
- A cell name is meaningless without its grid, so `propose_annotations(grid_cells=...)` requires
  the explicit `tile_size` (plus `overlap` if nonzero) the overlay echoed; it refuses rather than
  assume a grid, and recomputes the identical cells through the shared reference-grid geometry
- `tcip_annotation.grid.grid_to_rect()` looks a cell name up in a supplied cell list and returns
  its native-pixel rect

| Tool | Role | Phase |
|------|------|-------|
| `propose_annotations` | Propose candidate masks with a named engine, whole-frame or `grid_cells`-scoped | Discovery and correction |
| `stage_proposals(assignments=...)` | Stage classified candidates as predictions | Classification |
| `vision_tools.overlay_reference_grid` (library call) | Spatial reference for corrections | Correction |
| Agent's own image-capable read tool | Agent visual review | All phases |

## Review Protocol

1. Load ground truth with `annotation_tools.read_annotations` (library call, no MCP tool for it)
2. Load predictions (from inference or prior annotation)
3. `annotation_tools.score_predictions` (library call, or `tcip score-predictions`) pairs
   predictions to GT through the platform's one matcher, by IoU (over regions when any
   annotation is a polygon; default threshold: 0.5), and returns aggregate TP/FP/FN and average
   precision; `detail=True` adds a per-detection breakdown (each TP/FP/FN tagged with the
   annotation it names and its indices). A prediction carries its object class in
   `subject`, so this scores the object's localization, never an attribute head's call
4. Review on the Annotate canvas, which shows a bucket's predictions as proposals beside the
   image's annotations: accept a proposal, correct a value or a geometry, add a missed object,
   reject a proposal, and mark the image (or, on an orthomosaic, the region in view) complete. A
   proposal that overlaps an annotation of its subject is shown paired with it, through the one
   matcher the assessment counts with, so accepting it confirms that annotation rather than
   adding a second. Every gesture saves through the one label save; an accept or reject also
   appends one entry, `{proposal, action, by, at}` with `action` one of
   `tcip_annotation.verdicts.VerdictAction` (accepted, rejected), to the image's verdict shard
   under that bucket

The proposals the editor serves come with the bucket's own validated operating point
(`delivery.admitted_conf`), or the reason none is validated; the editor's confidence floor starts
there, and the person moves it.

### The review channel: propose on canvas, never write GT blind

The agent must never write ground truth the human hasn't seen. Stage proposals as a prediction
bucket and drive the human to review them:

- `stage_proposals(image_path, *, assignments=None, boxes=None, polygons=None, model_name=None)`
  publishes model-/agent-proposed shapes as prediction documents, never the image's label
  document, so nothing here becomes ground truth before a human reviews it. Exactly one input regime per call:
  `assignments` reads back the candidates `propose_annotations` staged for this image, each a
  `{candidate_id, subject}` mapping, stamped `created_by=<engine>`; `model_name` is refused
  alongside `assignments`, since the staged record already names the engine. `boxes`/`polygons`
  are explicit shapes an agent or another model already has in hand, with no cached record to
  read back; they require `model_name`, stamped as each object's `created_by`. Either way the
  image's proposal is published once as its own bucket named `<producer>/[<date>/]<stem>`, the
  name the answer's `bucket` carries, its record naming the engine or `model_name` as what
  proposed it and no checkpoint or execution record, and renders on the Annotate canvas for the
  person to accept, correct or reject; for the explicit regime, name the real producer in
  `model_name` (`claude`, `groundingdino`, `model:<run>`), not a generic placeholder. A second
  stage of the same image under the same producer refuses, since a reviewer may already have
  judged it, so further shapes go under another `model_name`. No delivery ships a proposal.
- `focus_human_attention(dataset_root, subject, date, image_index, mode, bucket, proposal)` drives
  the live Annotate tab to a frame, showing the proposals of the bucket named `bucket` under
  `dataset_root` (a published bucket, a proposal's included) with `proposal` selected, so the
  person sees exactly what you flagged without hunting. The event names this
  server's project by its id, and the backend delivers it only while the GUI has that project
  open: otherwise the answer is `delivered: false` with the id of the project the GUI does have
  open, and a backend that is not running answers `delivered: false` too.

Flow: run inference (or `stage_proposals`) → `focus_human_attention(bucket=...)` the
person to the weakest/flagged frames → they accept on the canvas → only then does it become GT. See
`packages/tcip-mcp/src/tcip_mcp/knowledge/delivery.md` for what ships after sign-off.

## Quality Metrics

- Coverage: fraction of images with labels
- Negatives: a training negative is a label document holding no annotation of the subject plus a
  person's completion mark for that subject over the whole image (`state == "negative"`). The
  admission, the review queue and the editor's own listing all read it through
  `LabelDocument.state`. An empty document nobody marked is unannotated: you cannot manufacture
  negatives, since writing empty label documents does not create them; only the person's mark
  does. `tcip doctor <root>` reports an empty document no mark finishes. Never delete empty label
  documents without asking.

## Active Learning

`prioritize_review_queue` ranks images by model uncertainty/diversity:
- High uncertainty = model unsure = most valuable to annotate
- Supports uncertainty, diversity, and combined scoring; with a `subject` named it skips the images
  whose label document marks that subject finished
- `prioritize_review_queue` returns a prioritized list for the annotator
