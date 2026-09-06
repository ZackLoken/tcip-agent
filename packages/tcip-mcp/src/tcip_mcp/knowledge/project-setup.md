---
name: project-setup
description: "The front-door arc: turn a breeder's raw pile of photos plus a stated goal into a structured, trainable TCIP project. Covers project naming, ingest_images (capture-date bucketing), translating a goal into a trait/task/classes.json, SAM-assisted bootstrap annotation, splitting, bespoke model design, training, inference, and review handoff. Load this when someone arrives with unstructured images and a phenotyping goal rather than a prepared dataset."
---

# Project setup: from raw photos to a trainable project

The real user is a tree-crop breeder, not an ML engineer, and the first interaction
is a sentence, not a folder picker:

> "I have all these images of hazelnut bushes with catkins on the plant. Help me detect
> the individual catkins."

Your job is to turn that (a raw image folder plus a goal) into a structured project the
rest of the platform (and the GUI) can work with. Do not improvise per-project file ops;
follow this arc. Each step links out to the domain skill that owns its detail.

## 0. Orient

Start the session with `load_project_memory` (kind='reports' and kind='retrospectives'), then `inspect_project`
on any project you're handed. Surface friction with `report_friction` the moment you hit
it (ambiguous goal, unconfirmed format).

## 1. Name the project: `crop_subject_phenotype`

Projects live under the workspace (`TCIP_WORKSPACE`, default `~/tcip-projects/`), one
folder per project. The shape is declared once, in `workspace.format_project_name`/
`parse_project_name`: three lowercase segments joined by underscores, hyphens allowed
within a segment. Neither function checks a segment against a vocabulary (tentative);
`ingest_images`, `initialize_project` and
`tcip import-project` refuse a non-conforming name when the directory they create lands
under the workspace.

    black-locust_tree_trunk-diameter

- `crop`: one of the six controlled crops (verify in
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`).
- `subject`: the object the annotations isolate (see step 3 below).
- `phenotype`: what is being measured (see
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crop-science.md`), never a `crops.yml` trait name.

The site (field/orchard) is no longer part of the name: it is a required argument of
`initialize_project` and `ingest_images`, recorded once on the project's own record
(`tcip_mcp.project_record`) and shown in the picker. Ask the human for it rather than
guessing it from a path or filename; a project that already records a different site
refuses the call rather than silently overwriting it.

Scales across 6 crops × subjects × phenotypes, and sorts sensibly on disk.

## 2. Ingest: `ingest_images`

Structure the raw pile into the canonical layout. One auditable primitive:

```python
from tcip_mcp.tools.ingest_tools import ingest_images
ingest_images(source="<raw folder or glob>", name="black-locust_tree_trunk-diameter", site="<the breeder's site>")
```

- Copies by default (originals are left byte-identical); pass `copy=False` only when
  the human explicitly wants the source moved.
- Buckets by the capture date each file itself states → ISO `YYYY-MM-DD`: a photo's EXIF
  `DateTimeOriginal`, a raster's own date metadata (its `DateTime` tag, an EXIF IFD, or a
  stitching engine's capture-date item). A file that states none goes to `images/undated/`.
  Override with `date_from="none"` (all undated) or a literal ISO date
  (`date_from="2026-02-11"`) when you know the capture date the camera didn't record.
- Never overwrites, and refuses a stem collision before copying a byte: two source files
  landing on the same bucket+stem (case-folded), or a source colliding with an existing
  image or `.bandgroup` manifest, refuses the whole call, naming both sides; re-ingesting
  the exact same file already placed is the one exception, skipped and reported in
  `skipped_collisions`. Relay the manifest to the human: `{total, buckets, undated,
  skipped_collisions, unreadable_dates}`, especially a refusal, a large `undated` count
  (dates may need `date_from`), and `unreadable_dates`, which separates files whose
  container could not be read at all from files that simply state no date. Every admitted
  file is ingested either way; the date never gates ingestion.

`ingest_images` does not annotate, split, choose a task, or write `classes.json`; the
next steps do. After it, `inspect_project` reports the capture dates and image count.

`ingest_images` scaffolds `.tcip/` (`artifacts/`, `models/`) as a side effect of
structuring images. If you need that scaffold before there are images to ingest, call
`initialize_project(project_path, site=<site>)` directly; it creates the identical layout without
touching image files. Both share the same internal scaffolding, so calling one after the other
with the same site is idempotent, not additive; calling either with a different site than the
one already recorded refuses rather than silently changing it.

After ingest, `register_dataset(dataset_root, crop)` records the dataset's identity (a minted
`id` plus a whole-dataset content fingerprint) in `<dataset_root>/dataset.json` and the project's
`.tcip/datasets.json`, so a later delivered number can be traced back to the exact data behind
it. `crop` is required and is never inferred from the path or a slug.

## 3. Translate the goal into a trait, task, and `classes.json`

Turn the breeder's sentence into a trait and the classes they distinguish. The CV task is yours to
derive from the data, not from the phrasing (see
`packages/tcip-mcp/src/tcip_mcp/knowledge/pipeline-design.md`):

- Task: the task string is an input to `build_dataset`, which routes a known set; a bespoke
  `dataset_source` is the seam for a task it does not route. A breeder saying "detect the
  individual catkins" names the *object* and the *phenotype* (the catkin, and a count), not the
  CV task. Which task measures that is yours to derive from object scale, separability, and what
  the trait actually counts; their verb is vocabulary, not the answer.
- Subject: the object class the annotations isolate (e.g. `catkin`, `bush`), not a path
  segment. Labels are one file per image (`annotations/<date>/<stem>.json`; see
  `dataset_layout.py`), holding every subject's annotation records for that image; `subject` is a
  field inside each record, resolved through the dataset's `classes.json` registry. Multiple
  subjects coexist in the same file (a bush isolated alongside its catkins).
- Classes: register the subject/attribute vocabulary in `classes.json` via the audited
  `write_class_map(dataset_root, subjects)` tool (never hand-edit the file) for what the breeder
  actually distinguishes: it validates the nested subject/attribute shape and writes the file plus
  an audit record. Keep it minimal first (progressive disclosure); class semantics live in
  `classes.json`, never in filenames. Verify crop traits against
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/` before asserting them.

## 4. Bootstrap annotation (engine-assisted)

There must be something to train on. Two paths (see
`packages/tcip-mcp/src/tcip_mcp/knowledge/annotation.md`):

- Agent/MCP path: `propose_annotations` a starter batch with a chosen `engine` (`'sam'` is the
  built-in reference; the agent can bring another) → review the candidates visually
  (`vision_tools.visualize`, a library call, or `tcip visualize`, then your client's
  image-capable read tool on the returned `image_path`) → `stage_proposals`
  with `assignments=[...]` for the good ones. `grid_cells=[...]` restricts a pass to a region of a large or crowded image
  instead of the whole frame. Trial engines and keep the one whose high-conf
  proposals survive review. An empty label file is not a negative on its own; it trains as one
  only once the breeder marks that image Complete (`.tcip/state/image_status.json`), so an empty
  file you write reads as unannotated until then. Never delete or skip them.
- Human path: hand off to the GUI Annotate tab for the breeder to label a seed set.

Never train or evaluate on an unconfirmed format: `tcip_annotation.format_io.detect_format`
refuses rather than guesses, inherited by `load_annotations_any` and
`annotation_tools.read_annotations` (library call, no MCP tool for it).

## 5. Split: `draw_splits`

Create leakage-free train/val/calibration splits with `draw_splits` (group-aware, keeps sibling
tiles of one source image in the same split; there is no held-out test list, and no launch path
honours one). Non-destructive by default (a stats dict only; a manifest is written only when
`output_path` is given or `materialize=True`); pass `materialize=True` to also lay out a
`{train,val,calibration}/{images,labels}/` tree, with the
platform's own per-image JSON labels (not YOLO's `.txt` format; `tcip-annotation` supports
`{json, coco}` only). Writing a manifest requires `subject` and all three ratios stated non-zero:
its members are drawn through the same admission a training run uses, and a run names it with
`data.split.manifest_dir` to train against that exact partition instead of drawing its own; the
manifest's `calibration` side is never bound to a loader, so a run's own selection (`val`) side is
never what the checkpoint is later validated against (see the `evaluation` skill's
Calibration/Holdout Split section). A run that drew its own split can have that exact
partition frozen into a manifest afterwards with `freeze_split_manifest(experiment_id)`, so a
later run binds to it from the data picker; a frozen manifest records an empty `calibration`
side and an `origin` naming the run, and the calibration doors refuse it by their own floor.

## 6. Build a model, train, infer

- Write an `nn.Module` (from scratch or importing the plain blocks) + a `train(ctx)` loop,
  build via `model_source` → `build_model`, pre-flight with `model_contract`; see
  `packages/tcip-mcp/src/tcip_mcp/knowledge/pipeline-design.md`.
- `launch_training` (immutable experiment per run) → watch metrics.
- `run_inference` to produce predictions for the review loop.

## 7. Prioritize review + deliver

`prioritize_review_queue` to focus the breeder's attention on the model's weakest
predictions, then deliver per `packages/tcip-mcp/src/tcip_mcp/knowledge/delivery.md`. Set the
workspace's active project so the GUI opens what you built.

## Reading the live session: `view_gui_state`

The GUI (a separate process) and you share the workspace, not memory. `view_gui_state`
is the bridge: it reads the active-project marker (`<workspace>/.active`) plus that
project's `<project_root>/.tcip/state/gui.json` and returns what the human is looking at
right now: `active_project`, `project_root`, `subject`, `date`, `active_tab`, and
`current_image_index` / `current_image`. Call it when the human says "this image" or "the
one I'm on" without a path. The nav index is persisted debounced as they page through
frames, so it lags a beat; treat it as "roughly where they are," not a frame-exact cursor.

Adopt the project with `activate_project` before doing project work. Adoption writes the
active marker *and* repins the platform-state root to `<workspace>/<project>`, so from then on
the experiment store and the model registry live under that one project's `.tcip/` alongside its
data, and the platform's own audit log is now this project's own, one file at one key
(self-contained and portable; a dataset's own audit log stays beside that dataset, unaffected by
adoption; `tcip archive-project <project_path> --output-path <path>` bundles the
project into a ZIP at the destination you name, or `--output-dir <dir>` writes the identical
bundle as a directory tree; `tcip import-project <bundle_path> <destination>`
restores either container into a destination dir, round-tripping back to an
`inspect_project`-visible project). After
adoption, `inspect_project`, `rank_registered_models` (listing with `metric=""` or ranking with
one stated) and `register_model` all default (`project_path=""`) to that project, and a
training run auto-registers there, so the model you trained is the one you retrieve. Pass an
explicit `project_path` only to reach a *different* project's registry: that holds for
`inspect_project`, `rank_registered_models`, and `register_model`'s explicit
mode. `register_model`'s experiment mode binds only in the experiment's own root; a
`project_path` there must name that same root or the call refuses by name. The repin reaches
only the calling process, so a training run in flight keeps writing to the project it started
under even if you (or the human, in the GUI) adopt another one meanwhile. The web backend
converges on your adopt as soon as it delivers; an MCP server
you are already running converges only at its next start inside the platform's own agent
terminal, so `inspect_project`'s divergence report is the guard in between.

## Invariants (from CLAUDE.md)

- State changes go through `@audited` MCP tools; `ingest_images` is one.
- Experiments are immutable: a new run each time; never overwrite history.
- Confirm before destructive/outward actions (moving source images, overwriting weights,
  exporting deliverables). Copy-by-default keeps ingestion non-destructive.
