---
name: project-setup
description: "The front-door arc — turn a breeder's raw pile of photos plus a stated goal into a structured, trainable TCIP project. Covers project naming, ingest_images (EXIF date bucketing), translating a goal into a trait/task/classes.json, SAM-assisted bootstrap annotation, splitting, model recommendation, training, inference, and review handoff. Load this when someone arrives with unstructured images and a phenotyping goal rather than a prepared dataset."
---

# Project setup — from raw photos to a trainable project

The real user is a **tree-crop breeder, not an ML engineer**, and the first interaction
is a sentence, not a folder picker:

> "I have all these images of hazelnut bushes with catkins on the plant. Help me detect
> the individual catkins."

Your job is to turn that — a raw image folder plus a goal — into a structured project the
rest of the platform (and the GUI) can work with. Do not improvise per-project file ops;
follow this arc. Each step links out to the domain skill that owns its detail.

## 0. Orient

Start the session with `load_reports` + `load_retrospectives`, then `get_project_status`
on any project you're handed. Surface friction with `claude_reports` the moment you hit
it (missing site, ambiguous goal, unconfirmed format).

## 1. Name the project — `{crop}_{trait}_{site}`

Projects live under the workspace (`TCIP_WORKSPACE`, default `~/tcip-projects/`), one
folder per project. The name is a controlled slug: **lowercase; hyphens within a segment,
underscores between segments**:

    {crop}_{trait}_{site}
    hazelnut_catkin-05-50-95-per-date_valley-farm

- `crop` — one of the six controlled crops (verify in `.github/skills/crops/`).
- `trait` — the phenotype being measured (see `.github/skills/crop-science`).
- `site` — the field/orchard. **If the goal doesn't name a site, ask the human** — don't
  invent one. A wrong site name fragments a breeder's data across seasons.

Scales across 6 crops × many traits × sites, and sorts sensibly on disk.

## 2. Ingest — `ingest_images`

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
  `{total, buckets, undated, skipped_collisions}` — especially any collisions or a large
  `undated` count (a sign the photos lack EXIF and dates may need `date_from`).

`ingest_images` does **not** annotate, split, choose a task, or write `classes.json` — the
next steps do. After it, `get_project_status` reports the capture dates and image count.

## 3. Translate the goal into a trait, task, and `classes.json`

Turn the sentence into a pipeline shape (see `.github/skills/pipeline-design`):

- **Task**: "detect the individual catkins" → object **detection** (`detect`). Dense
  small objects → detection; whole-region measurement may be segmentation.
- **Trait / annotation campaign**: the `<trait>` path segment under `annotations/`
  (e.g. `catkin`). Multiple campaigns can coexist (`catkin`, `bush`).
- **Classes**: write `classes.json` for what the breeder actually distinguishes. Keep it
  minimal first (progressive disclosure) — class semantics live in `classes.json`, never
  in filenames. Verify crop traits against `.github/skills/crops/` before asserting them.

## 4. Bootstrap annotation (SAM-assisted)

There must be something to train on. Two paths (see `.github/skills/annotation`):

- **Agent/MCP path**: `sam_auto_label` a starter batch → review the candidates visually
  (`visualize` / `view_image`) → `accept_candidates` the good ones. Empty label files are
  **valid negatives** — never delete or skip them.
- **Human path**: hand off to the GUI Annotate tab for the breeder to label a seed set.

Never train or evaluate on an unconfirmed format: if `load_annotations` returns
`format_confident: false`, stop and confirm.

## 5. Split — `split_dataset` / `make_splits`

Create leakage-free train/val/test splits. Prefer `make_splits` (group-aware, keeps
sibling tiles of one source image in the same split) when tiling is involved.

## 6. Recommend a model, train, infer

- `recommend_model_spec` for a starting point (compose from `backbone → neck → heads →
  loss`; the registry is a library, not a constraint — see `.github/skills/training`).
- `launch_training` (immutable experiment per run) → watch metrics.
- `run_inference` to produce predictions for the review loop.

## 7. Prioritize review + deliver

`prioritize_review_queue` to focus the breeder's attention on the model's weakest
predictions, then deliver per `.github/skills/delivery`. Set the workspace's active
project so the GUI opens what you built, closing the loop for the human.

## Invariants (from CLAUDE.md)

- State changes go through `@audited` MCP tools; `ingest_images` is one.
- Experiments are immutable — a new run each time; never overwrite history.
- Confirm before destructive/outward actions (moving source images, overwriting weights,
  exporting deliverables). Copy-by-default keeps ingestion non-destructive.
