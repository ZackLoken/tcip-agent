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
proposed it (its `assignments` regime reads back the record of the one `propose_annotations` run
its `staging_id` names, not a label document). It is the one label document shape the platform
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

## Reading labels, and importing external COCO or YOLO labels

Nothing here trains or calibrates on a dataset-level COCO file (an `images` or `categories`
key): it is never a per-image document, and the only way in is the import. An external COCO
export becomes per-image documents through `import_coco(document, dataset_root, date,
vertex_spacing, rationale)`, over images `ingest_images` already placed. `date` names the capture:
the dated folder under `images/`, or the flat `images/` root for a dataset with no dated folders.
Every declared category
must be a subject the dataset's registry declares. An ordinary image is the capture's image with
exactly the document's `file_name`; a `.bandgroup` capture is tied by stem. An image with no
annotations writes nothing. Before anything is written the import reports every fault it finds
together and refuses: a malformed, repeated or unregistered category, a record the reader or the
writer refuses (a record's `image_id` and its content are checked independently, so one record
can report both), an image id that is not an integer or is listed twice, an image not in the
capture, a stated frame the image does not have, or an existing per-image document. Every
document and the import's audit event then commit together, or none does: a label placed
meanwhile refuses the whole import. A crowd region keeps its `iscrowd` flag, and a run-length mask
becomes the rings `mask_contours.mask_to_polygon_rings` extracts from it, simplified at the
stated `vertex_spacing`, pixels of perimeter per vertex: each ring keeps at most the vertex count
its own perimeter implies at that spacing (the nearest integer, at least
`state.RING_MIN_POINTS`), as the subset of the contour's own vertices `cv2.approxPolyDP` keeps at
the least tolerance reaching it, so 0 keeps
every contour vertex. The records keep the provenance the COCO carried and gain none. The
audit event records the COCO file's path, the digest of the bytes read, the stated spacing with
its rationale and the documents written; an import that writes no document, whether
refused or carrying no annotations, changes nothing and leaves no line.

YOLO labels enter through `import_yolo(labels_dir, names, dataset_root, date)`: one
`<image stem>.txt` per image, whatever extension the capture holds, a `.bandgroup` by stem.
`names` lists subject names in class-id order; each used class must be registered. A non-blank
line is `class x_center y_center width height` for a box or `class x1 y1 x2 y2 ... xn yn`
for a polygon of at least three points, normalized to the image's display frame. A malformed
line, an uncovered class id, a coordinate outside [0, 1] beyond 1e-9 floating-point slack, or a
non-positive box extent refuses naming the file and line. A file naming no capture image
refuses too. The files become in-memory COCO data and go through the same conversion and
transaction, with the same document refusals. Images without files are untouched. Empty files
write nothing and their stems are listed under `empty` in the answer: they need a person's
completion mark to become negatives. The `yolo_labels_imported` event records the label folder,
class names, capture, empty and written stems, and `labels_digest`, a digest of the file names
and store versions from the reads that supplied the labels. No document written means no event.

A record carrying `iscrowd` is a region of unseparated objects, never one instance: the built-in
detection and instance heads train its region as background (they read no crowd flag, so a crowd
box handed to them would train as one object), a detection inside it is neither a true nor a
false positive at evaluation, and it counts as no object wherever a ground-truth count is formed
or an object is paired, the classifier calibration's pairing included. Every loader target keeps
it under `iscrowd`, so a `train(ctx)` of your own can act on it.

The per-image document is read by `json_io.read_label_document`, wrapped for the agent by
`read_annotations`, and written by `save_annotations`. A missing label document reads as no annotations; a present one the reader
cannot make sense of raises `json_io.UnreadableLabelDocumentError` naming the document or the
malformed record's index, rather than reading short: a stored value that does not decode, a
non-dict document, a completion mark missing any of its fields, an `annotations` value that is
not a list, a record that is not a dict, a record with no string subject, and a record carrying a
value that is not what the schema states,
whichever geometry the record resolves to (a `bbox` that is not four numbers or has no positive
extent, a `segmentation` that is not rings `state.is_ring` admits, a `point` that is not two
numbers, a `score` that is not a number, an attribute value that is not a non-empty name, an
`iscrowd` that is not 0, 1, true or false). An absent or `null` value reads as absent, for every
optional field. A shape `save_annotations` or the Annotate tab saves is translated into this
record and decoded by the same decoder, so it is refused for the same values. The import's COCO
reader, `format_io.parse_coco_annotations`, names each record's subject from the document's own
`categories` and hands each record to that same decoder. An unreadable document is not the same
fact as no document.

A collaborator's delivery in a schema other than COCO or YOLO is yours to convert: read a
sample, write a one-off converter script that emits a COCO document, where
`packages/tcip-mcp/src/tcip_mcp/knowledge/project-setup.md` places your own work for a project,
and import it.

## Coordinate frame: the reading and its count, a photograph's EXIF applied once

An image's frame depends on how it is read and at what band count: `SourceHeader`
(`raster_source.py`) answers it, `display_frame` for the frame the GUI canvas and the image
route serve, `frame_at(n)` and `open(n)` for a read at `n` bands (the model's count). An array
whose axes read differently at different counts (a `(3, 64, 4)` raster displays 64 wide and 3
tall, and opens at three bands 4 wide and 64 tall) has different frames for those reads, so
annotation coordinates
are authored and compared in the frame of the read that produced them. For a photograph the
frame is EXIF-upright: its size and its decoded pixels go through one shared EXIF
orientation-tag read. This matters most for
Orientation-6 phone/camera JPEGs whose stored frame is transposed (e.g. 5712×4284 ↔
4284×5712): denormalizing an upright-authored box against the raw sensor frame scatters
every box. Do not re-open images with a bare `PIL.Image.open` for anything coordinate-
bearing (denormalizing, cropping, drawing); read the frame and the pixels through
`SourceHeader` or the readers it opens.

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
| `import_yolo` | Convert YOLO box and polygon label files into per-image label documents |
| `propose_annotations` | Run a named proposal engine over an image, whole-frame or over named grid cells |
| `push_panel_event` | Push an arbitrary event to a GUI panel over the tcip-web backend for this server's project, not restricted to images/annotations; delivered only while the GUI has that project open |
| `prioritize_review_queue` | Rank images by active-learning uncertainty/diversity for the next review batch, skipping those marked finished for a subject |
| `triage_predictions` | Sort a registered checkpoint's own predictions into a needs-review queue by a stated confidence band and its rationale, beside the images whose predictions carry no confidence at all; writes nothing |

## Engine-assisted auto-labeling (the engine is a capability, not a fixed method)

Auto-labeling runs through a method-neutral proposal seam (`tcip_mcp.pipelines.proposal`): the
agent names an `engine` and the platform runs it. Two engines ship registered, each prompted
through `engine_params`:

- `sam`, a promptable segmenter of the SAM family: `points` (a list of `[x, y]` pixel points) and
  `boxes` (a list of `[x1, y1, x2, y2]` pixel boxes), each point and each box a prompt of its own
  and at least one stated; `multimask_output` (required: the model's several ranked masks per
  prompt, or its one best); `checkpoint` (default `facebook/sam-vit-large`). Every mask is one
  polygon candidate, `engine_meta` naming its prompt (indexed over `points` then `boxes`) and its
  rank.
- `grounding_dino`, a text-prompted detector of the Grounding DINO family: `phrase`,
  `box_threshold`, `text_threshold`, `postprocess` (the cross-tile merge, named as a tiled
  inference pass names it) and `cross_tile_nms` (the merge's match threshold), all required;
  `tile_size` and `overlap` (the lattice, stated, else derived); `checkpoint` (default
  `IDEA-Research/grounding-dino-base`). Every detection is one box candidate, `engine_meta`
  naming its text label.

The segmenter runs each prompt on a window of the frame cut at its processor's `longest_edge`
around the prompt (`engine_meta`'s `window`, a box wider than the edge taking its own extent), and
the processor scales the window's longer side to that edge: a window whose longer side is the edge
reaches the model at native resolution, a frame shorter than the edge is upsampled and a wider box
downsampled by its ratio to the edge. The detector is run by the tiled pass a checkpoint's inference
runs, on a lattice a whole-frame pass's boxes derive the way inference derives one from ground
truth: the edge five times the objects' p99 side (`TILE_EDGE_DERIVATION`), never under the
model's own input edge (recorded `derived` where that floor applies), and the overlap that holds
such an object whole in a tile (`TILE_OVERLAP_DERIVATION`). A lattice whose edge reaches the
frame's longer side is the frame itself. A frame larger than the model's input edge that the
whole-frame pass finds nothing on refuses, naming `tile_size` and `overlap` to state. The
pass's execution record, each value beside its source, travels with the staged proposal into
the bucket its accepted candidates publish to.

Each engine downloads its checkpoint's weights from the Hugging Face host on its first call.
Prompts are pixel coordinates in the upright frame the canvas shows. Bring any other engine,
an open-vocabulary detector or a bespoke proposer, by `register_proposal_engine` or a dotted
`module:factory` you wrote, exactly the way `model_source` lets you bring a model. An empty or
unregistered name refuses, listing the registered ones. The candidate schema is neutral
(`candidate_id` / `bbox` / `area` / `rings` / `score`), engine signals under `engine_meta`
(`rings` is `Polygon.rings`, one contour per connected region of the proposed mask, so an
occlusion-split object stays split from proposal through accept, or `null` for a box candidate,
which stages as its box). A bespoke proposer may be classical-analysis-based (an OpenCV pipeline
the agent writes); its proposals are soft and prove out only by surviving review.

Trial and compare by review: pick the engine the data justifies, and let the breeder's review
decide: the useful engine is the one whose high-confidence proposals *survive review* (high
accept rate, few edits). That accept rate, read from the verdict shards the editor writes, is the
measured comparison. Do not promise an engine that isn't wired.

### Vision-guided auto-labeling (the engine is the "hands", the agent's vision the "eyes")

The agent labels images by using a proposal engine for geometry, its multimodal vision where
candidates need sorting, and the person's canvas for review.

Full workflow, one image per call:

1. `propose_annotations(image_path, engine=<name>)` → the engine proposes candidates, renders a
   numbered overlay. `grid_cells=[...]` (with `tile_size`, `overlap` echoed by
   `overlay_reference_grid`) restricts the pass to the named cells' bounding rect instead of the
   whole frame, useful on a large or crowded image; the engine itself never sees a region, only a
   crop, and the returned candidates are already in full-frame coordinates. On a dataset image
   this stages the run as a record of its own (`staged: true`) under a fresh `staging_id` the
   answer carries, keyed by the dataset, capture date, stem and that id, alongside the content
   identity of the pixels the engine ran on; an earlier run's record stays as it was. On a path
   outside any dataset's `images/` tree the render and candidates still come back
   (`staged: false`, `staging_id: null`, naming why), but there is nothing for
   `stage_proposals`'s `assignments` regime to read back later
2. Assign each kept candidate its subject, since `stage_proposals` takes every one as
   `{candidate_id, subject}`. Where the run's subject is already known, every candidate is kept
   under it, with no overlay read. Read the numbered overlay with your own image-capable read tool
   only to assign candidates among several subjects or to drop ones that are plainly not the
   object
3. `stage_proposals(image_path, assignments={staging_id: <the run's>, candidates: [{candidate_id:
   0, subject: "leaf"}, ...]})`
   → reads that run's candidates, refusing if the image has changed since the run, then stages
   the kept ones as predictions (`created_by=<engine>`) in a bucket named for the run's id; it
   never writes GT directly
4. Review the bucket on the person's canvas with `focus_human_attention` and
   `capture_live_canvas` (the review channel below) → visual QA pass

Visual QA is not optional: read what the person's canvas shows. A rendered image
(`visualize`) is for what the canvas cannot draw.

Corrective loop (for missed objects):

1. `overlay_reference_grid(image_path, overlap=)` → labeled reference grid ('A1' top-left) for
   spatial reference
2. Agent reads the grid overlay with its own image-capable read tool → identifies missed regions
   by grid cell
3. `propose_annotations(image_path, engine=<name>, grid_cells=["B3", "D5"], tile_size=<echoed>,
   overlap=<echoed>)` → the engine proposes over those cells' bounding rect
4. Stage the right candidates with `stage_proposals(assignments={staging_id, candidates})`, the
   `staging_id` this run answered, for the person to accept

Grid cell system:

- `overlay_reference_grid(image_path, overlap=, tile_size=)` renders square cells of
  `tile_size` native pixels (omitted, it derives a legible default from the image dims) stepping
  by the stated `overlap`, and echoes the full grid geometry back in every response:
  `tile_size`, `overlap`, `cols`, `rows`, `width`, `height`
- Agent references cells like "B3" or "F5" instead of pixel coordinates
- A cell name is meaningless without its grid, so `propose_annotations(grid_cells=...)` requires
  the explicit `tile_size` and `overlap` the overlay echoed; it refuses rather than assume a
  grid, and recomputes the identical cells through the shared reference-grid geometry
- `tcip_annotation.grid.grid_to_rect()` looks a cell name up in a supplied cell list and returns
  its native-pixel rect

| Tool | Role | Phase |
|------|------|-------|
| `propose_annotations` | Propose candidate masks with a named engine, whole-frame or `grid_cells`-scoped | Discovery and correction |
| `stage_proposals(assignments=...)` | Stage classified candidates as predictions | Classification |
| `overlay_reference_grid` | Spatial reference for corrections | Correction |
| Agent's own image-capable read tool | Agent visual review | All phases |

## Review Protocol

1. Load ground truth with `read_annotations`
2. Load predictions (from inference or prior annotation)
3. `score_predictions` pairs
   predictions to GT through the platform's one matcher, by IoU (over regions when any
   annotation is a polygon) at the `iou_threshold` and `conf_threshold` you state, and returns
   aggregate TP/FP/FN and average
   precision; `detail=True` adds a per-detection breakdown (each TP/FP/FN tagged with the
   annotation it names and its indices). A prediction carries its object class in
   `subject`, so this scores the object's localization, never an attribute head's call
4. Review on the Annotate canvas, which shows a bucket's predictions as proposals beside the
   image's annotations: accept a proposal, correct a value or a geometry, add a missed object,
   reject a proposal, and mark the image (or, on an orthomosaic, the region in view) complete. A
   proposal that overlaps an annotation of its subject is shown paired with it, through the one
   matcher the assessment counts with, so accepting it signs that annotation off, stamping the
   person's `accepted_by` and `accepted_at` on it, rather than adding a second; rejecting it
   records the rejection and leaves the annotation as it was. Every gesture saves through the one
   label save; an accept or reject also appends one entry, `{proposal, action, by, at}` with
   `action` one of `tcip_annotation.verdicts.VerdictAction` (accepted, rejected), to the image's
   verdict shard under that bucket. An annotation no person has made or signed
   off (authorship `tool`, a producer that is no person, or `unattributed`, no producer
   recorded) awaits review whether or not a bucket is shown; the
   save's `confirm` gesture, by position among the annotations the save writes, signs it off the
   same way and writes no verdict entry

The proposals the editor serves come with the bucket's own validated operating point
(`delivery.admitted_conf`), or the reason none is validated; the editor's confidence floor starts
there, and the person moves it.

The editor names how the labels and the bucket agree about each object without saying which side
is right: matched (both hold it), proposal only (the bucket proposes it and the labels do not hold
it, which is a missed label as often as a false positive) and annotation only (the labels hold it
and the bucket does not propose it). The review decides; the names never assume the labels are
complete.

A person can flag an annotation, a proposal or the image as a whole with a comment asking for a
second look, and anyone can resolve a flag with a reply. Flags save through the one label save
into the image's own flags record (`tcip_annotation.flags`), are kept once resolved, and a flag on
an annotation the save removes is resolved as removed. A flag is a note between people: it never
changes a label, a verdict or a completion mark.

### The review channel: propose on canvas, never write GT blind

The agent must never write ground truth the human hasn't seen. Stage proposals as a prediction
bucket and drive the human to review them:

- `stage_proposals(image_path, *, assignments=None, boxes=None, polygons=None, model_name=None)`
  publishes model-/agent-proposed shapes as prediction documents, never the image's label
  document, so nothing here becomes ground truth before a human reviews it. Exactly one input
  regime per call:
  `assignments` (`{staging_id, candidates}`) reads back the candidates of the one
  `propose_annotations` run its `staging_id` names, each kept candidate a
  `{candidate_id, subject}` mapping, stamped `created_by=<engine>`; `model_name` is refused
  alongside `assignments`, since the staged record already names the engine. `boxes`/`polygons`
  are explicit shapes an agent or another model already has in hand, with no cached record to
  read back; they require `model_name`, the producer of the shapes (`claude`, `groundingdino`,
  `model:<run>`), as each object's `created_by` records it. Either way the image's proposal is
  published as a new bucket named `<producer>/<date>/<stem>/<id>`, `id` the run's `staging_id`
  (a fresh one for explicit shapes), the name the answer's `bucket` carries, its record naming
  the engine or `model_name` as what proposed it, carrying the execution record of the tiled pass
  the proposal ran through (`null` where none ran: a prompt window, explicit shapes) and naming
  no checkpoint or class scope, and renders on the Annotate canvas for the person to accept,
  correct or reject. A run's bucket is published once:
  a second stage of the same run refuses, and a new run on the image publishes a bucket of its
  own beside the first, so a published bucket is never overwritten and the verdicts a reviewer
  gave an earlier one stay with it. The audit line `propose_annotations` writes with a run's
  staged record and the one its publication writes each name the actor that door was given: the
  signed-in person for a prompt sent from the Annotate canvas (`identity.actor` of the request's
  `user`, as the label save records it), and `None` for a call the MCP server makes on an agent's
  behalf. No delivery ships a proposal.
- `focus_human_attention(dataset_root, subject, date, image_index, mode, bucket, proposal)` drives
  the live Annotate tab to a frame of `subject` on `date` under `dataset_root`; `bucket` and
  `proposal` are optional, and with them named it shows the proposals of that bucket (a published
  bucket, a proposal's included) with `proposal` selected, so the person sees exactly what you
  flagged without hunting. The answer's `delivered` says whether the backend accepted the event,
  never that a browser showed it: the backend accepts it only while it has this server's project
  open, otherwise answering `delivered: false` with the id of the project it does have open, and
  with no backend running the answer is `delivered: false` too.
- `capture_live_canvas(refresh, crop_to_viewport, max_edge)` renders the canvas state the GUI last
  pushed for this project (its image, viewport, and shapes, unsaved or in-progress ones included)
  and returns an `image_path` you read with your own image-capable read tool. With `refresh` it
  first pings the GUI to push fresh state and waits briefly for it. The answer's `refreshed`,
  `refresh_ping_delivered` and `state_age_seconds` describe the canvas state's metadata, which a
  heartbeat updates without shapes; whether the shapes were pushed and how old they are is
  `shapes_missing` and `shapes_age_seconds`. With no state pushed at all it refuses.

Flow, in the order of the bullets above: a bucket from `run_inference` or `stage_proposals` →
`focus_human_attention` → the person's review on the canvas. See
`packages/tcip-mcp/src/tcip_mcp/knowledge/delivery.md` for what ships after sign-off.

## Quality Metrics

- Coverage: fraction of images with labels
- Negatives: a training negative is a label document holding no annotation of the subject plus a
  person's completion mark for that subject over the whole image (`state == "negative"`). The
  admission, the review queue and the editor's own listing all read it through
  `LabelDocument.state`. An empty document nobody marked is unannotated: you cannot manufacture
  negatives, since writing empty label documents does not create them; only the person's mark
  does. `diagnose_project` names an empty document no mark finishes among its findings; the
  orientation a new conversation of a session launched for a project starts with holds the first
  of them and how many there are. Never delete empty label documents without asking.

## Active Learning

`prioritize_review_queue` ranks images by model uncertainty/diversity:

- High uncertainty = model unsure = most valuable to annotate
- The scorer is a stated block, `{"name": ..., **values}`: the built-in `uncertainty` and
  `diversity` take no values and `combined` requires `uncertainty_weight` and
  `diversity_weight`; the budget and the `rationale` for both are stated beside it. With a
  `subject` named it skips the images whose label document marks that subject finished
- `prioritize_review_queue` returns a prioritized list for the annotator, beside the scorer block,
  budget and rationale it ranked under
