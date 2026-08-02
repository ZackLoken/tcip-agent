---
name: project-setup
description: "The front-door arc: turn a breeder's raw pile of photos plus a stated goal into a structured, trainable TCIP project. Covers project naming, ingest_images (EXIF date bucketing), translating a goal into a trait/task/classes.json, SAM-assisted bootstrap annotation, splitting, bespoke model design, training, inference, and review handoff. Load this when someone arrives with unstructured images and a phenotyping goal rather than a prepared dataset."
---

# Project setup: from raw photos to a trainable project

The real user is a **tree-crop breeder, not an ML engineer**, and the first interaction
is a sentence, not a folder picker:

> "I have all these images of hazelnut bushes with catkins on the plant. Help me detect
> the individual catkins."

Your job is to turn that (a raw image folder plus a goal) into a structured project the
rest of the platform (and the GUI) can work with. Do not improvise per-project file ops;
follow this arc. Each step links out to the domain skill that owns its detail.

## 0. Orient

Start the session with `load_project_memory` (kind='reports' and kind='retrospectives'), then `inspect_project`
on any project you're handed. Surface friction with `claude_reports` the moment you hit
it (missing site, ambiguous goal, unconfirmed format).

## 1. Name the project: `{crop}_{trait}_{site}`

Projects live under the workspace (`TCIP_WORKSPACE`, default `~/tcip-projects/`), one
folder per project. The name is a controlled slug: **lowercase; hyphens within a segment,
underscores between segments**:

    {crop}_{trait}_{site}
    hazelnut_catkin-05-50-95-per-date_valley-farm

- `crop`: one of the six controlled crops (verify in `.github/skills/crops/`).
- `trait`: the phenotype being measured (see `.github/skills/crop-science`).
- `site`: the field/orchard. **If the goal doesn't name a site, ask the human**; don't
  invent one. A wrong site name fragments a breeder's data across seasons.

Scales across 6 crops × many traits × sites, and sorts sensibly on disk.

## 2. Ingest: `ingest_images`

Structure the raw pile into the canonical layout. One auditable primitive:

```
ingest_images(source="<raw folder or glob>", name="{crop}_{trait}_{site}")
```

- **Copies** by default (originals are left byte-identical); pass `copy=False` only when
  the human explicitly wants the source moved.
- Buckets by **EXIF `DateTimeOriginal` → ISO `YYYY-MM-DD`**; images with no EXIF date go
  to `images/undated/`. Override with `date_from="none"` (all undated) or a literal ISO
  date (`date_from="2026-02-11"`) when you know the capture date the camera didn't record.
- **Never overwrites**: a stem collision (two source files → same bucket+stem) is skipped
  and reported in `skipped_collisions`. Relay the manifest to the human:
  `{total, buckets, undated, skipped_collisions}`, especially any collisions or a large
  `undated` count (a sign the photos lack EXIF and dates may need `date_from`).

`ingest_images` does not annotate, split, choose a task, or write `classes.json`; the
next steps do. After it, `inspect_project` reports the capture dates and image count.

`ingest_images` scaffolds `.tcip/` (`config.toml`, `artifacts/`, `models/`) as a side effect of
structuring images. If you need that scaffold before there are images to ingest, call
`init_project(project_path)` directly; it creates the identical layout without touching image
files (both share the same internal scaffolding, so calling one after the other is idempotent,
not additive).

After ingest, `register_dataset(dataset_root, crop)` records the dataset's identity (a minted
`id` plus a whole-dataset content fingerprint) in `<dataset_root>/dataset.json` and the project's
`.tcip/datasets.json`, so a later delivered number can be traced back to the exact data behind
it. `crop` is required and is never inferred from the path or a slug.

## 3. Translate the goal into a trait, task, and `classes.json`

Turn the breeder's sentence into a trait and the classes they distinguish. The CV task is yours to
derive from the data, not from the phrasing (see `.github/skills/pipeline-design`):

- **Task**: the task string is an input to `build_dataset`, which routes a known set; a bespoke
  `dataset_source` is the seam for a task it does not route. A breeder saying "detect the
  individual catkins" names the *object* and the *phenotype* (the catkin, and a count), not the
  CV task. Which task measures that is yours to derive from object scale, separability, and what
  the trait actually counts; their verb is vocabulary, not the answer.
- **Subject**: the object class the annotations isolate (e.g. `catkin`, `bush`), not a path
  segment. Labels are one file per image (`annotations/<date>/<stem>.json`; see
  `dataset_layout.py`), holding every subject's annotation records for that image; `subject` is a
  field inside each record, resolved through the dataset's `classes.json` registry. Multiple
  subjects coexist in the same file (a bush isolated alongside its catkins).
- **Classes**: register the subject/attribute vocabulary in `classes.json` via the audited
  `write_class_map(dataset_root, subjects)` tool (never hand-edit the file) for what the breeder
  actually distinguishes: it validates the nested subject/attribute shape and writes the file plus
  an audit record. Keep it minimal first (progressive disclosure); class semantics live in
  `classes.json`, never in filenames. Verify crop traits against `.github/skills/crops/` before
  asserting them.

## 4. Bootstrap annotation (engine-assisted)

There must be something to train on. Two paths (see `.github/skills/annotation`):

- **Agent/MCP path**: `propose_annotations` a starter batch with a chosen `engine` (`'sam'` is the
  built-in reference; the agent can bring another) → review the candidates visually (`visualize`,
  then your client's image-capable read tool on the returned `image_path`) → `accept_proposals`
  the good ones. Trial engines and keep the one whose high-conf
  proposals survive review. An empty label file is not a negative on its own; it trains as one
  only once the breeder marks that image Complete (`.tcip/state/image_status.json`), so an empty
  file you write reads as unannotated until then. Never delete or skip them.
- **Human path**: hand off to the GUI Annotate tab for the breeder to label a seed set.

Never train or evaluate on an unconfirmed format: if `read_annotations` returns
the format cannot be determined, `read_annotations` returns an error rather than a guess.

## 5. Split: `make_splits`

Create leakage-free train/val/test splits with `make_splits` (group-aware, keeps sibling
tiles of one source image in the same split). Non-destructive by default (writes stem
manifests + stats); pass `materialize=True` to also lay out a
`{train,val,test}/{images,labels}/` tree, with the platform's own per-image JSON labels
(not YOLO's `.txt` format; `tcip-annotation` supports `{json, coco}` only).

## 6. Build a model, train, infer

- Write an `nn.Module` (from scratch or importing the plain blocks) + a `train(ctx)` loop,
  build via `model_source` → `build_model`, pre-flight with `model_contract`; see
  `.github/skills/pipeline-design`.
- `launch_training` (immutable experiment per run) → watch metrics.
- `run_inference` to produce predictions for the review loop.

## 7. Prioritize review + deliver

`prioritize_review_queue` to focus the breeder's attention on the model's weakest
predictions, then deliver per `.github/skills/delivery`. Set the workspace's active
project so the GUI opens what you built, closing the loop for the human.

## Reading the live session: `view_gui_state`

The GUI (a separate process) and you share the workspace, not memory. `view_gui_state`
is the bridge: it reads the active-project marker (`<workspace>/.active`) plus that
project's `<project_root>/.tcip/state/gui.json` and returns what the human is looking at
right now: `active_project`, `project_root`, `subject`, `date`, `active_tab`, and
`current_image_index` / `current_image`. Call it when the human says "this image" or "the
one I'm on" without a path. The nav index is persisted debounced as they page through
frames, so it lags a beat; treat it as "roughly where they are," not a frame-exact cursor.

**Adopt the project with `set_active_project` before doing project work.** Adoption writes the
active marker *and* repins the platform-state root to `<workspace>/<project>`, so from then on
the `@audited` log, the experiment store, and the model registry all live under that one
project's `.tcip/` alongside its data (self-contained and portable; `archive_project` bundles
everything; `import_project` restores that ZIP into a destination dir, round-tripping back to a
`inspect_project`-visible project). After adoption, `inspect_project`, `select_best_model`, `list_registered_models`,
and `register_model` all default (`project_path=""`) to that project, and a
training run auto-registers there, so the model you trained is the one you retrieve. Pass an
explicit `project_path` only to reach a *different* project's registry. The repin is a
deliberate action, so a training run in flight keeps writing to the project it started under
even if you (or the human, in the GUI) adopt another one meanwhile.

## Invariants (from CLAUDE.md)

- State changes go through `@audited` MCP tools; `ingest_images` is one.
- Experiments are immutable: a new run each time; never overwrite history.
- Confirm before destructive/outward actions (moving source images, overwriting weights,
  exporting deliverables). Copy-by-default keeps ingestion non-destructive.
