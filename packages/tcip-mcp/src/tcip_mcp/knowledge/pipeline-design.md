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
  them. `pipelines.derivations.derive_cross_tile_nms` reads the GT's neighbor-IoU tail for an
  `nms` merge, and returns `None` when the GT gives no basis for a threshold. An assessment or a
  full-frame evaluation derives an unstated threshold from its reference this way and refuses
  when nothing derives; any other tiled pass, and every IoS merge (`nmm`/`greedynmm`), states
  `cross_tile_nms`. An `iou_match` trait's match threshold derives from the GT's box size at the jitter
  and margin the trait itself authors.
- Capture-date bucketing from `ingest_images`: whether a time series exists at all, and at what
  cadence.

Those readings are facts about *this* dataset. A trait category is not.

## What the platform can ingest today

Interface constraints you cannot read off the toolkit. If a trait needs something in the second
list, say so plainly rather than approximating it.

Buildable now:

- 2D imagery from any capture modality: aerial, ground, rover-mounted, lab/benchtop.
- RGB and N-channel rasters (GeoTIFF, NPZ, grayscale). A run reads its sources at one width,
  `data.num_channels` when it states one and the sources' own count otherwise, recorded on the run
  and read back by every later reader, so inference is channel-aware.
- The task strings `build_dataset` routes, or a bespoke `dataset_source` you write for a task it
  does not route. The seam is open; the loader set is not a taxonomy.

Not buildable now (no loader, no task type, no scaffolding carried):

- 3D point clouds (LiDAR / SfM). No point-cloud dataset or loader, and no task type.
- Non-imagery spectral readings (a bare NIR / hyperspectral sample, not a raster). The dataset
  layer reads 2D imagery; there is no loader for a spectrum.
- A *learned* contextual-ranking task: a model that scores a plant relative to its plot or
  block neighbors. No task type or loader exists for it. Ranking plants by a measurement you
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
- Labeled examples are scarce, and scarcest where labeling one costs a judgment call rather than
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

Tailor the architecture to the data in hand, derived rather than pinned:

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

- `pipelines.data.datasets.build_from_dataset_source((dataset_source, layout), task=,
  samples=, scope=, transforms=)` builds from a `dataset_source`: an *importable* builder you
  wrote (`{"builder": "my_module:build_ds", "builder_kwargs": {...}, "source_files": [...]}`,
  mirroring `model_source`), imported from the run's layout. A run naming one in its
  `data.dataset_source` builds its loaders, and `ctx.build_dataset`'s, through it;
  `build_dataset` builds the platform's own loaders only. It receives the samples the
  platform's own producer named for the side being built and the class space they were
  admitted under (`samples` / `scope` / `transforms` / `task`), each sample carrying the logical
  image its pixels are read from as `image` (a `BandGroupRef` for a grouped capture), sizing the
  dataset it builds itself, plus your own `builder_kwargs`, and must return a torch `Dataset`.
  Never a directory, a document path or a format flag: the platform names the samples and your
  builder builds over them, so a strip split over your own samples composes the tiled wrapper
  inside `train(ctx)` at the lattice it states (`ctx.tiled_dataset(base, tile_size=,
  overlap=)`). The platform resolves no `data.tiling` for a bespoke run and refuses a config
  stating one beside its `dataset_source`. `ctx.build_dataset` takes `samples` and `transforms`
  and nothing else, and builds over the samples you hand it, whole: through the run's builder,
  or through the platform's factory at the run's recorded `sizes` and `tiling`, under the run's
  `scope` either way. On a within-image split run (`ctx.spatial`) it builds the run's one
  sample's train view, every tile inside the recorded train region, and refuses any other
  sample list naming that sample, so no held-out pixel trains. `builder_kwargs` configure
  your builder and may not restate
  `samples`, `scope`, `task` or `transforms`; a builder that did would train on membership or a
  class space the run's own record does not describe, so the seam refuses it by name. `scope` is
  a `ClassScope`: its `subject` and `attributes`, every attribute the registry declares for that
  subject (name, type, values in declared order, a value's id its position there), both `None`
  where the ground truth carries its own classes (a mask raster, a table row): derive the class
  space from the ground truth you were handed. A task with no
  built-in loader is not a task with no producer: the platform admits by the shape of the ground
  truth (the images' own label documents, or the masks or table `data.labels_dir` names), whatever
  the task, so your builder receives the same samples a built-in loader would. Registry-free, imported like any module, never `exec`'d.
- `pipelines.model_build.build_from_model_source(spec.model_source, layout, dims)` builds from a
  validated config's `model_source`: an *importable* builder you wrote (`{"builder":
  "my_module:build_net", "builder_kwargs": {...}, "source_files": [...], "task": "detection"}`,
  `builder` and `task` required), imported from `layout`, the run's declared files laid out under
  one root (`model_build.SourceLayout`; `ctx.build_model` hands it the run's own). `source_files`
  names the files your run imports, the builder's own module among them and a
  `training_source` loop's module too; a builder whose module none of them is refuses, and so
  does a module inside a package whose `__init__.py` is not declared beside it. Preflight
  imports from the same layout the run will hold. A declared module's own imports resolve when
  it is imported: once another run's sources are imported in the same process, an import a
  function makes at call time binds that run's copy of the name, or refuses. Each run copies the
  declared files into its own
  directory, and the run and every checkpoint it writes import those copies. A checkpoint carries
  the digest of the copies' contents its run recorded, so it loads only in a project holding
  that run: its own, or another one the whole run was carried into (`archive_project`, then
  `import_project`); copied alone beside another run of the same file names, it refuses. It is
  imported, never `exec`'d.
  The platform hands your builder the run's width as `in_chans` (`data.num_channels`, the band count its
  sources carry) and its head sizes: over label documents, `num_classes`, the subjects the scope
  isolates, and `attributes`, the scope's attribute records when it declares any, one
  per-instance head per record sized by its values (`build_detector` takes them and adds those
  heads over its boxes); over ground truth carrying its own classes, `num_classes` or `num_ranks`
  as that ground truth derives them; a regression run carries no count. Your builder accepts
  those keywords, and `builder_kwargs` restating one refuses by name. A bespoke `dataset_source` whose loader composes its own bands
  states `data.num_channels` for the width it hands the model, and one over ground truth that
  carries its own classes states `data.num_classes` or `data.num_ranks`.
  `pipelines.model_contract`
  states the *only* model-side contract, the measurement boundary: your model must train (finite
  gradient loss) and emit inference output the library scorers consume. `launch_training` runs this
  contract for you: `preflight_config(smoke=True)` builds the model and smokes it at the *resolved*
  dims and frame (the tile edge, else the frame every untiled source shares), or on one real batch
  of the run's own dataset when it resolved no frame, every attribute head included, before the
  training subprocess spawns, so a broken builder fails the launch, not a wasted run. `ctx.check_contract` / `ctx.overfit_check` are the same proofs on
  demand; `launch_training(overfit_check=True)` runs `ctx.overfit_check`'s own diagnostic at
  launch, on the contract's batch, and records the result on the run's `model_contract`, never
  gating (a valid model can fail twenty steps on noise).
- `training_source` points the envelope at your custom `train(ctx)`. The `TrainContext` (`ctx`,
  `pipelines.training.envelope`) hands you the craft library: prebuilt leakage-free loaders,
  `ctx.build_optimizer` / `ctx.build_scheduler` / `ctx.evaluate` / `ctx.set_seed`, the
  progressive-unfreeze primitive `ctx.apply_stage_freeze`, `ctx.tiled_dataset`, and the
  correctness checks `ctx.check_contract` / `ctx.overfit_check`, plus the envelope-owned
  sinks `ctx.log_metrics`, `ctx.log_batch`, `ctx.save_checkpoint`, `ctx.record_artifact`,
  `ctx.should_cancel`. Route
  your loop's metrics and checkpoints through those sinks and the run stays audited, immutably
  versioned, and provenance-snapshotted no matter what your loop does. Each sink writes into the
  run's own directory, each checkpoint tag and artifact name once, and refuses once the run has
  ended. `ctx.record_artifact` copies any other file in under a name of its own.
  `ctx.default_train()` is one convenience, not a requirement: call it, extend it, or replace it
  entirely.

  `state` reserves the keys the platform stamps every checkpoint with: `config` (this run's own
  launch config, the record every publishing door reads a run's `data.scope` from) and
  `source_snapshot` (the digest of the source snapshot this run took). A `state` carrying either
  refuses; name a bespoke loop's own field something else.

  Registration needs one more fact your loop states explicitly. A checkpoint your loop saved via
  `ctx.save_checkpoint(state, "model_best")` or `"model_final"` is the deliverable once your loop
  returns; any other tag (or the default, untagged `ctx.save_checkpoint(state)`) is the
  deliverable only once you name that tag with `ctx.set_final_weights(tag)`, which refuses a tag
  your loop saved nothing under. The deliverable is read back through the same verified reader
  every loading door uses; a "completed" run with no deliverable, or one that reader refuses, is
  marked `failed` naming why. Audit and provenance are unconditional, registration is not. A
  metrics row carrying a `selection` value is the run's progress on its objective, and under
  `run_hyperparameter_search` it reports trial progress for pruning and the trial's result; a
  bespoke loop whose rows carry none records one with `ctx.report_objective(value)`, which writes
  that row.

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

When a trait's decomposition needs more than one training phase, write a one-off script that
chains the canonical primitives, in the project's own directory, never in this repository: one
build path, every step audited.
Each training phase calls `launch_training` (full audited envelope, leakage-free split,
tiling persistence) against a `model_source` builder; run each stage's model with
`run_inference`; then aggregate with the importable postprocessing libs
(`aggregate_per_plant` / `export_aggregated_csv`, or `deliver_phenology_milestones` for milestone dates):

```python
# <trait>_pipeline.py: chain the primitives; each launch_training goes
# through the audited envelope, so provenance and immutability hold across the whole run.
stage_a = launch_training(config={"model_source": {...}, "data": {...}})
stage_b = launch_training(config={"model_source": {...}, "data": {...}})
run_inference(checkpoint_path=stage_b_best, images_dir=images_dir, bucket="stage-b/<date>")

# aggregate_per_plant never guesses plant identity from a filename; supply a real plant_id_fn.
# build_plant_mapping (a GNSS + capture-sequence resolver) is the real mechanism.
from pathlib import Path
from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping

build = load_mapping(project_root, mapping_name)  # from a prior build_plant_mapping call
by_stem = {row["stem"]: row for rows in build.rows().values() for row in rows}

def plant_id_fn(image_path: str) -> str | None:
    row = by_stem.get(Path(image_path).stem)
    return row["plot_name"] if row else None

# Each row carries its assignment's own source, distance_m and plant_attribution, read by those names.
image_results = [
    {**r, **{k: row[k] for k in ("source", "distance_m", "plant_attribution")}}
    for r in read_stage_b_preds_as_image_results("stage-b/<date>")  # your own per-image count reader
    if (row := by_stem.get(Path(r["image"]).stem))
]

# The final CSV is a phenotype delivery door: it clears the one gate over the published buckets
# the counts came from, and an unvalidated bucket refuses without a breeder's recorded acknowledgment.
summaries = aggregate_per_plant(image_results, plant_id_fn=plant_id_fn)
deliver_per_plant_csv(summaries, "phenotype.csv", delivered_phenotype="<phenotype>",
                      delivery_kind="per_plant_count_aggregate", plants=[...],
                      dataset_root=dataset_root, buckets=["stage-b/<date>"])
```

How many stages there are, and what each one does, is your decomposition to derive; the chaining
mechanics are identical for one stage or four, and there is no fixed phase vocabulary. See the
`training` and `delivery` skills for the primitive signatures.

## Design principles

- Start with the simplest thing that could measure the trait, and add complexity only when the
  data or the metrics justify it.
- Write a retrospective (`write_retrospective`) when you finish. Record what you measured
  about *this* dataset and what it implied: object scale, capture cadence, class imbalance, where
  the operating point resolved and why. Not a reusable pipeline shape: the next dataset re-derives
  its own decomposition.
