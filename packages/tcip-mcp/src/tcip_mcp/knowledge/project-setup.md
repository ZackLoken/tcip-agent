---
name: project-setup
description: "The front-door arc: turn a breeder's raw pile of photos plus a stated goal into a structured, trainable TCIP project. Covers creating the project (its id, display name and site), ingest_images (capture-date bucketing), translating a goal into a trait/task/subjects.json, SAM-assisted bootstrap annotation, splitting, bespoke model design, training, inference, and review handoff. Load this when someone arrives with unstructured images and a phenotyping goal rather than a prepared dataset."
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
it (ambiguous goal, unconfirmed format). `tcip doctor <project>` names a stray file
under `.tcip/state` no store claims; delete one through `delete_stray_state_file` with the
person's confirmation as `reason`.

## 1. Create the project

Projects live under the workspace `TCIP_WORKSPACE` names (every process refuses to start without
it), one folder per project. `initialize_project(project_path, display_name, site)` creates one: its
`.tcip/` scaffold and its record (`tcip_mcp.project_record`), which holds a freshly minted `id`,
the `display_name` the picker shows, and the `site`. The id is the project's identity; the
folder's name and the display name are neither of them identity, so a display name is the
breeder's own words, for example (hazelnut, one marked example) `Hazelnut catkin counts, north
block`.

The site (field/orchard) is the breeder's own words too, recorded once on the record and shown
in the picker. Ask the human for both rather than guessing them from a path or filename; a project
that already records a different site or display name refuses the call rather than silently
overwriting it (`tcip write-project-site` corrects a site typed wrong once).

This MCP server acts on the one project it was started for (`--project <path>`); every tool that
acts on a project takes that one, never a path you pass. A project you create here is worked on
by a server started for it: the GUI opens it from the picker, and the agent terminal it starts
runs a server for the project the GUI has open.

A display name that turns out wrong is corrected from the project picker's own Rename control. A
rename changes the display name alone: the folder, every run, mapping, registry and delivery stay
where they are, since each record names in-project paths relative to the project.

## 2. Ingest: `ingest_images`

Structure the raw pile into the canonical layout of the project this server was started for. One
auditable primitive:

```python
ingest_images(source="<raw folder or glob>")
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

`ingest_images` does not annotate, split, choose a task, or write `subjects.json`; the
next steps do. After it, `inspect_project` reports the capture dates and image count.

After ingest, `register_dataset(dataset_root, crop)` records the dataset's identity (a minted
`id` plus a whole-dataset content fingerprint) in `<dataset_root>/dataset.json` and the project's
`.tcip/datasets.json`, so a later delivered number can be traced back to the exact data behind
it. `crop` is required and is never inferred from the path or a slug.

## 3. Translate the goal into a trait, task, and `subjects.json`

Turn the breeder's sentence into a trait and the subjects they distinguish. The CV task is yours to
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
  field inside each record, resolved through the dataset's `subjects.json` registry. Multiple
  subjects coexist in the same file (a bush isolated alongside its catkins).
- Subjects: register the subject/attribute vocabulary in `subjects.json` via the audited
  `write_subject_registry(dataset_root, subjects)` tool (never hand-edit the file) for what the
  breeder actually distinguishes: it validates the nested subject/attribute shape and writes the
  file plus an audit record. Keep it minimal first (progressive disclosure); subject semantics
  live in `subjects.json`, never in filenames. Verify crop traits against
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/` before asserting them.

## 4. Bootstrap annotation (engine-assisted)

There must be something to train on. Three paths (see
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
- Existing labels in an external dataset-level COCO: `import_coco(document, dataset_root, date)`
  converts them into per-image documents over the images `ingest_images` placed under that date.
  Nothing trains on the COCO file itself.

Never train or evaluate on an unconfirmed format: the one per-image label reader refuses any
document of another shape (a dataset-level COCO, the old `objects` schema, an unrecognized one)
rather than guessing, and every training, calibration and review reader reads through it.

## 5. Select: `draw_splits`

Draw a leakage-free train/val/calibration selection with `draw_splits` (group-aware, keeps
sibling tiles of one source image in the same split; there is no held-out test list, and no
launch path honors one). It copies nothing: a selection lists, per sample, the image source, the
label document, a group key and a side, so a draw spanning capture dates trains in place. Nothing
is written without `output_path` (a stats dict only). Writing a selection requires all three
ratios stated non-zero, and `subject` whenever its ground truth is per-image label documents,
whose admission is subject-scoped; a selection over `<stem>.png` masks or a `.csv` table of one
row per image takes none, since each is admitted by existing. Its samples are
drawn through the same admission a training run
uses, and a run names it with `data.split.selection_dir` to train against that exact partition
instead of drawing its own; the `calibration` side is never bound to a loader, so a run's own
selection (`val`) side is never what the checkpoint is later validated against (see the
`evaluation` skill's Calibration/Holdout Split section). A run that drew its own split can have
that exact partition frozen into a selection afterwards with `freeze_selection(experiment_id)`,
so a later run binds to it from the data picker; a frozen selection records an empty
`calibration` side and an `origin` naming the run, and the calibration doors refuse it by their
own floor.

## 6. Build a model, train, infer

- Write an `nn.Module` (from scratch or importing the plain blocks) + a `train(ctx)` loop,
  build via `model_source` → `build_model`, pre-flight with `model_contract`; see
  `packages/tcip-mcp/src/tcip_mcp/knowledge/pipeline-design.md`.
- `launch_training` (immutable experiment per run) → watch metrics.
- `run_inference` to produce predictions for the review loop.

## 7. Prioritize review + deliver

`prioritize_review_queue` to focus the breeder's attention on the model's weakest
predictions, then deliver per `packages/tcip-mcp/src/tcip_mcp/knowledge/delivery.md`.

## Reading the live session: `view_gui_state`

The GUI (a separate process) and you share the project's files, not memory. `view_gui_state`
is the bridge: it reads this project's `.tcip/state/gui.json` and returns what the human is
looking at right now: `dataset_root`, `subject`, `date`, `active_tab`, and
`current_image_index` / `current_image`. Call it when the human says "this image" or "the
one I'm on" without a path. The nav index is persisted debounced as they page through
frames, so it lags a beat; treat it as "roughly where they are," not a frame-exact cursor.
It reads this project's own snapshot: while the GUI has another project open, that snapshot is
where the human last was in this project, not what they are looking at now.

Everything a project holds lives under its own `.tcip/` beside its data: the experiment store,
the model registry and its audit log (a dataset's own audit log stays beside that dataset).
`tcip archive-project <project> --output-path <path>` bundles the project into a ZIP at the
destination you name, or `--output-dir <dir>` writes the identical bundle as a directory tree;
`tcip import-project <bundle_path> <destination>` restores either container into a destination
dir, keeping the project's id, round-tripping back to an `inspect_project`-visible project. A
training run started for a project keeps writing to that project whatever the GUI opens
meanwhile.

## Invariants (from CLAUDE.md)

- State changes go through audited MCP tools, each leaving one audit line per act (the tool's own,
  or its library's event); `ingest_images` is one.
- Experiments are immutable: a new run each time; never overwrite history.
- Confirm before destructive/outward actions (moving source images, overwriting weights,
  exporting deliverables). Copy-by-default keeps ingestion non-destructive.
