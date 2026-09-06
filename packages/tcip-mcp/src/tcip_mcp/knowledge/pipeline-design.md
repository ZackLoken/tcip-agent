---
name: pipeline-design
description: "The one build path, the bespoke seams, and what the platform can ingest today. You build every model one way: an agent-written nn.Module + train(ctx) loop, via model_source. No pipeline shape is supplied; the decomposition is yours to derive from the data. Load when deciding how to measure a new trait, designing an ML pipeline, or building a model architecture for a trait."
---

# Pipeline Design

## No pipeline shape is supplied

This skill gives you no pattern to match a trait against. How many stages a trait
needs, what each one does, and whether it is one model or several is your decomposition to derive
from the data in hand.

Derive it by measuring the dataset, not by classifying the trait:

- `tcip scan-dataset <folder_path>`: how many images, labels and predictions exist,
  and the detected label format; capture dates come from `ingest_images`, below.
- `pipelines.derivations.gt_aspect_ratios` over the GT `(w, h)`: the object elongation that
  actually occurs here, rather than an assumed shape.
- Object scale against your tile size: whether objects survive tiling, and whether a seam cuts
  them. `pipelines.derivations.derive_cross_tile_nms` returns `None` when the GT gives no basis
  for a threshold; that `None` is expected, not a failure.
- Capture-date bucketing from `ingest_images`: whether a time series exists at all, and at what
  cadence.

Those readings are facts about *this* dataset. A trait category is not.

## What the platform can ingest today

Interface constraints you cannot read off the toolkit. If a trait needs something in the second
list, say so plainly rather than approximating it.

Buildable now:

- 2D imagery from any capture modality: aerial, ground, rover-mounted, lab/benchtop.
- RGB and N-channel rasters (GeoTIFF, NPZ, grayscale). `num_channels` threads to the backbone's
  `in_chans`, and inference is channel-aware.
- The task strings `build_dataset` routes, or a bespoke `dataset_source` you write for a task it
  does not route. The seam is open; the loader set is not a taxonomy.

Not buildable now (no loader, no task type, no scaffolding carried):

- 3D point clouds (LiDAR / SfM). No point-cloud dataset or loader, and no task type. See
  CLAUDE.md's Scope section and README's Roadmap.
- Non-imagery spectral readings (a bare NIR / hyperspectral sample, not a raster). The dataset
  layer reads 2D imagery; there is no loader for a spectrum.
- A *learned* contextual-ranking task: a model that scores a plant relative to its plot or
  block neighbours. No task type or loader exists for it. Ranking plants by a measurement you
  already produced is ordinary postprocessing over the per-plant table, and is available now.

## Conditions in this domain's imagery

Properties of the subjects and the capture, observed on real breeding-block imagery. What any of
them costs you depends on what you are measuring and how; that part is yours to work out.

- Plants in a row overlap and merge at typical standoff; their boundaries are frequently not
  separable in the image at all.
- Lighting and weather vary between captures enough that a model can learn the covariate instead
  of the trait.
- Objects of interest are often a few pixels across, near the resolution floor, and tiling cuts
  them at seams.
- Labelled examples are scarce, and scarcest where labelling one costs a judgment call rather than
  a box.
- Capture cadence is irregular and dates go missing within a season.
- Wind moves the subject between captures of the same plant.

## You own the model and the training loop

You are the CV scientist:
for every trait you write a bespoke `nn.Module`, from scratch or by importing the plain
building blocks (FPN/PAN necks, the classification/ordinal/regression/semantic-seg heads, the
losses, backbone wrappers, and `build_detector` + the `_build_*` detector functions), and,
when the technique is novel, a custom training loop. There is no model spec, no composer, and
no component registry: nothing forces a model to a fixed shape or the default trainer.

The building blocks are plain importable symbols, e.g.:

```python
from tcip_mcp.pipelines.components.backbones import BackboneWrapper
from tcip_mcp.pipelines.components.necks import FPN, PAN
from tcip_mcp.pipelines.components.heads import ClassificationHead, SemanticSegHead
from tcip_mcp.pipelines.components.losses import build_loss, compute_class_weights
from tcip_mcp.pipelines.components.detectors import build_detector, BackboneNeckAdapter
```

Compose them inside your own `nn.Module`, or ignore them and write the network from scratch.
No architecture is imposed. The `toolkit-inventory` skill is the name-and-location map for the
whole set: the `build_detector` / `build_loss` / task string names, the heads/necks/backbones,
the derivations, the `ctx` craft library, and the proposal-engine and scorer registries.

Tailor the architecture to the data in hand (CLAUDE.md: derive, don't pin):

- Anchors from the GT box shapes, not a fixed `(0.5, 1, 2)`. Feed the dataset's GT `(w, h)`
  through `pipelines.derivations.gt_aspect_ratios` and set anchor *sizes* from the GT object-size
  distribution, so anchors cover the objects that actually occur (e.g. elongated organs a default
  ratio can't match).
- Strides / feature levels to the object scale: add a finer pyramid level for tiny objects,
  drop levels you don't need.
- Normalization to the batch size: with the tiny batches large detectors force, BatchNorm
  statistics are unreliable; prefer `GroupNorm` (or another batch-independent norm).
- Activations / layers where the data warrants it: this is engineering judgment, not a menu.

Three seams support bespoke work; the platform guarantees integrity around it:

- `pipelines.data.datasets.build_dataset(task, dataset_source, **kwargs)` builds from a
  `dataset_source` when the task string isn't one of the known loaders: an *importable* builder you
  wrote (`{"builder": "my_module:build_ds", "builder_kwargs": {...}, "source_files": [...],
  "task": "..."}`, mirroring `model_source`). It receives the run's data context (`images_dir` /
  `labels_dir` / `stems` / `transforms` / `task`) merged with `builder_kwargs` (which win on
  conflict) and must return a torch `Dataset`. Registry-free, imported like any module, never
  `exec`'d.
- `pipelines.model_build.build_model(config)` builds from a `model_source`: an *importable*
  builder you wrote (`{"builder": "my_module:build_net", "builder_kwargs": {...},
  "source_files": [...], "task": "detection", "in_chans": 3}`). It is imported, never `exec`'d.
  `pipelines.model_contract`
  states the *only* model-side contract, the measurement boundary: your model must train (finite
  gradient loss) and emit inference output the library scorers consume. `launch_training` runs this
  contract for you: `preflight_config(smoke=True)` builds the model and smokes it at the *resolved*
  in_chans/num_classes/img_size before the training subprocess spawns, so a broken builder fails the
  launch, not a wasted run. `ctx.check_contract` / `ctx.overfit_check` are the same proofs on
  demand; `launch_training(overfit_check=True)` runs `ctx.overfit_check`'s own diagnostic at
  launch, on the contract's batch, and records the result on the run's `model_contract`, never
  gating (a valid model can fail twenty steps on noise).
- `training_source` points the envelope at your custom `train(ctx)`. The `TrainContext` (`ctx`,
  `pipelines.training.envelope`) hands you the craft library: prebuilt leakage-free loaders,
  `ctx.build_optimizer` / `ctx.build_scheduler` / `ctx.evaluate` / `ctx.set_seed`, the
  progressive-unfreeze primitive `ctx.apply_stage_freeze`, `ctx.tiled_dataset`, `ctx.calibrate`,
  and the correctness checks `ctx.check_contract` / `ctx.overfit_check`, plus the envelope-owned
  sinks `ctx.log_metrics`, `ctx.save_checkpoint`, `ctx.record_artifact`, `ctx.should_cancel`. Route
  your loop's metrics and checkpoints through those sinks and the run stays audited, immutably
  versioned, and provenance-snapshotted no matter what your loop does. `ctx.record_artifact` is a
  free-form sink for any other name; the name `"model_weights"` is reserved for the run's
  deliverable and is routed to `ctx.set_final_weights` instead, with a warning, rather than
  recorded or raised. `ctx.default_train()` is one convenience, not a requirement: call it,
  extend it, or replace it entirely.

  `state` reserves two top-level keys: `schema_version` (the platform's own checkpoint-version
  field) and `config` (always this run's own launch config, the record every publishing door
  reads a run's `(subject, attribute, id_map)` scope from). A `state` carrying either refuses;
  name a bespoke loop's own field something else.

  Registration needs one more fact your loop states explicitly. A checkpoint saved via
  `ctx.save_checkpoint(state, "model_best")` or `"model_final"` is found automatically after your
  loop returns; any other tag (or the default, untagged `ctx.save_checkpoint(state)`) is not
  registered as the run's deliverable unless you call `ctx.set_final_weights(path)` yourself. A
  "completed" run with no discoverable weights and no `set_final_weights` call is marked `failed`
  rather than registering a nonexistent path; audit/provenance are unconditional, registration
  is not. Under `run_hyperparameter_search`, a bespoke loop whose own metrics don't share the stock trainer's key
  names (`selection`/`val_objective`/`val_loss`, the only ones the automatic per-epoch pruning
  signal recognizes) can call `ctx.report_objective(value)` directly to report trial progress for
  pruning, a no-op outside HPO, safe to call unconditionally.

When the plain blocks and your own primitives both plateau on a trait, the next move is to research the
literature for a technique that fits; see the `cv-research` skill for the research→implement→validate
loop (and the rule that a new method must beat the baseline on the *measured phenotype* before you
trust it).

Dimensional traits: mask geometry is a supported measurement. `pipelines.measurement.mask_geometry`
(also `ctx.mask_geometry`) computes area, perimeter, centroid and the extents along the mask's own PCA
principal/secondary axes on a *validated* mask, in pixels and in the caller's stated unit when given a
scale. Geometry on a validated mask is a valid measurement, subject to the same
validate-before-you-trust rule as any other. An axis extent is a straight chord of the mask's
footprint, not an anatomical span: it answers the trait's dimension only when the structure is
straight and its visual long axis is the statistically dominant one. When the definition calls for a
span a chord cannot represent (an arc length, a skeleton path, a landmark-to-landmark distance),
compose that computation on the same validated mask instead of relabeling an extent as it.

A canopy segment (`deliver_orthomosaic_plant_counts`'s `canopy_subject` argument) is whatever the
breeder accepted, reviewed into the raster's own label document: a hand trace, a SAM proposal a
reviewer accepted, or a bespoke instance-segmentation model's own output once a reviewer has
accepted it, all admitted the same way. The door prescribes no model architecture for how the
boundary was produced; what it requires is that a person positively stands behind it.

## Multi-phase pipelines

When a trait's decomposition needs more than one training phase, write a one-off logged script
that chains the canonical primitives: one build path, every step audited.
Each training phase calls `launch_training` (full audited envelope, leakage-free split,
tiling persistence) against a `model_source` builder; run each stage's model with
`run_inference`; then aggregate with the importable postprocessing libs
(`aggregate_per_plant` / `export_aggregated_csv`, or `deliver_phenology_milestones` for milestone dates):

```python
# <trait>_pipeline.py: chain the primitives; each launch_training goes
# through the audited envelope, so provenance and immutability hold across the whole run.
stage_a = launch_training(config={"model_source": {...}, "data": {...}})
stage_b = launch_training(config={"model_source": {...}, "data": {...}})
run_inference(checkpoint_path=stage_b_best, images_dir=images_dir, output_dir="stage_b_preds")

# aggregate_per_plant never guesses plant identity from a filename; supply a real plant_id_fn.
# build_plant_mapping (a GNSS + capture-sequence resolver) is the real mechanism.
from pathlib import Path
from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping

build = load_mapping(project_root, mapping_name)  # from a prior build_plant_mapping call
by_stem = {row["stem"]: row for rows in build.rows().values() for row in rows}

def plant_id_fn(image_path: str) -> str | None:
    row = by_stem.get(Path(image_path).stem)
    return row["plot_name"] if row else None

image_results = [
    {**r, "image": r["image"],
     "plant_id_source": (row := by_stem.get(Path(r["image"]).stem)) and row["source"],
     "plant_id_distance_m": row and row["distance_m"],
     "plant_attribution": row and row["plant_attribution"]}
    for r in read_stage_b_preds_as_image_results("stage_b_preds")  # your own per-image count reader
]

# The final CSV is a phenotype delivery door: without pred_dirs it floors to unvalidated
# regardless of any asserted string, and this call passes no acknowledgement, so it still refuses.
summaries = aggregate_per_plant(image_results, plant_id_fn=plant_id_fn)
export_aggregated_csv(summaries, "phenotype_csv", delivered_phenotype="<phenotype>", pred_dirs=["stage_b_preds"])
```

How many stages there are, and what each one does, is your decomposition to derive; the chaining
mechanics are identical for one stage or four, and there is no fixed phase vocabulary. See the
`training` and `delivery` skills for the primitive signatures.

## Design principles

- Start with the simplest thing that could measure the trait, and add complexity only when the
  data or the metrics justify it (CLAUDE.md's progressive-disclosure rail).
- Write a retrospective (`write_retrospective`) when you finish. Record what you measured
  about *this* dataset and what it implied: object scale, capture cadence, class imbalance, where
  the operating point resolved and why. Not a reusable pipeline shape: the next dataset re-derives
  its own decomposition.
