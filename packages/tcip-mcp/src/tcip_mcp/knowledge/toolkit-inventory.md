---
name: toolkit-inventory
description: "The composition vocabulary for the whole library: the string names build_detector / build_loss accept, the heads/necks/backbones you import, the derivations that size a model to the data, the ctx craft library a train(ctx) loop composes, the two bespoke seams (model_source / training_source / dataset_source), and the auto-label proposal-engine and active-learning scorer registries you can register your own into. Load when composing an nn.Module or train(ctx) loop, reaching for a building block and unsure of its name, wiring a bespoke dataset/proposer/scorer, or asking 'what pieces already exist so I don't rebuild one'."
---

# Toolkit inventory: the pieces to compose

The platform hands you plain importable PyTorch primitives, a few name→builder factories, and
open seams for the parts you write yourself. The tables below inventory what exists; they are a
map, not a fixed set. Adding a primitive (a new head, a new proposal engine, a new
acquisition scorer) is expected; it extends the map, it does not close it. For the build path, the
bespoke seams, and what the platform can ingest today see the `pipeline-design` skill; it supplies
no pipeline shape; this skill is the name-and-location reference its import block points at.

`build_detector` and `build_loss` already self-document at runtime: an unknown name raises
a `ValueError` naming the registered names. The heads, necks, backbones, and ctx methods have no
such factory, so this skill is where their names live.

## Model building blocks (import and compose inside your own `nn.Module`)

| Piece | Where | Names / notes |
|-------|-------|---------------|
| `build_detector(name, adapter, num_classes, attributes=(), **kw)` | `pipelines.components.detectors` | names: `faster_rcnn`, `fcos`, `retinanet`, `mask_rcnn`. `attributes`, the scope's attribute records the platform hands a builder, adds one per-instance head per attribute over the features pooled at each box (`AttributeDetector`: cross-entropy for a categorical attribute, the ordinal head's CORN loss for an ordinal one, a row unassessed for an attribute masked from that head's loss); every output then carries `attributes`, one id per attribute per box. Every build requires `anchor_base_size`, `min_size`/`max_size`, `num_levels`, `featmap_names` and `image_mean`/`image_std`, plus `aspect_ratios` for every detector except `fcos`, which is anchor-free and has none, so an `fcos` builder does not forward the `aspect_ratios` it is handed. The geometry is the one the run's resolution records (`derive_detector_geometry` at the run's frame, or for a run resolving no frame the one `model_source.detector_geometry` states) and the statistics the `normalization` record's, both handed to your builder (below). The torchvision detector class's own constructor kwargs (`box_nms_thresh`, `rpn_nms_thresh`, …) are forwarded too, but an operating-point knob, a detection cap (`box_detections_per_img`, `detections_per_img`) or a score threshold (`box_score_thresh`, `score_thresh`), which refuses: each forward is capped at its frames' own cap and runs at its pass's conf, or at no floor during validation. Wrap a backbone+neck in `BackboneNeckAdapter` first. Unknown name → `ValueError`; a kwarg this detector doesn't take → `TypeError` naming the detectors that do. A build with no `image_mean`/`image_std` is refused by the builder's signature, and one with statistics of another length than `in_chans` (probed from the adapter's first conv when unstated) → `ValueError`. |
| `set_detector_operating_point(model, score_thresh=, detections_per_img=)` / `detector_operating_point_holder(model)` | `pipelines.operating_point` | Where a detector's in-model operating-point knobs live: the module itself, its `.detector.roi_heads` (two-stage), or its `.detector` (one-stage), the first that exposes either attribute name (`score_thresh`, `detections_per_img`). The setter sets both knobs, each required, before a forward, and refuses naming a knob the model does not expose; the model's own in-model NMS stays as its builder constructed it. A bespoke, non-torchvision module exposes both on itself and keeps, in its eval forward, only the detections scoring at least `score_thresh`, the `detections_per_img` highest. The holder refuses, naming both, when more than one location exposes a knob. |
| `build_loss(name, ...)` | `pipelines.components.losses` | names: `cross_entropy`, `weighted_ce`, `focal`, `smooth_l1`, `huber`, `bce`, `dice`, `giou`, `corn`, `coral`. Compose terms with `+` (e.g. `"bce+dice"` → `CombinedLoss`). A weightable loss (`cross_entropy`/`weighted_ce`/`focal`) auto-injects inverse-frequency weights when given `class_distribution`. `compute_class_weights` is the standalone weigher. |
| Heads (no factory, import the class) | `pipelines.components.heads` | `ClassificationHead` (task `classification`), `OrdinalHead` (task `ordinal`, the CORN loss), `RegressionHead` (task `regression`), `SemanticSegHead` (task `semantic_seg`, blends CE+Dice internally). Each implements `forward` / `compute_loss` / `decode`. |
| Necks (no factory, import the class) | `pipelines.components.necks` | `FPN`, `PAN` (both take `add_p2` for a finer level), `IdentityNeck` (pass-through), `GlobalAvgPoolNeck` (→ flat `[B, C]` vector for classification/ordinal/regression heads). |
| Backbones | `pipelines.components.backbones` | `BackboneWrapper` is the whole surface: wrap a module that already emits a list/tuple/dict of feature maps (timm `features_only=True`, torchvision `create_feature_extractor`, or your own staged module, finest stage first) to get `s0..sN` naming, a declared `out_channels`, and per-stage `freeze_to`. It does not turn a classifier into a feature extractor. There is no build helper: `out_indices` decides whether you get a pyramid at all, so it is yours to choose. See the module docstring for the two-line wrap. |

`task` strings (the `model_source["task"]` value; also the `build_dataset` task keys and the
head `task_type`s): `detection`, `instance_seg`, `semantic_seg`, `classification`, `ordinal`,
`regression`. They route measurement/eval and dataset loading; a genuinely new task uses the
`dataset_source` seam below.

## Deriving a model to the data in hand (derive, don't pin)

| Derivation | Where | Returns |
|------------|-------|---------|
| `probe_channels(image_path)` | `pipelines.derivations` | band count of *this* raster (RGB→3, N-band GeoTIFF→N), the probe a run's `data.num_channels` is resolved by. |
| `num_classes_from_distribution(dist)` | `pipelines.derivations` | `max class id + 1` from *this* label set. |
| `gt_aspect_ratios(boxes)` | `pipelines.derivations` | anchor aspect ratios spanning *this* dataset's GT box shapes (elongated organs a fixed ratio set can't match), the p10 and p90 of h/w with 1.0, or `None` when the GT gives no basis. |
| `derive_detector_geometry(frame, regions)` | `pipelines.derivations` | a torchvision detector's `schemas.DetectorGeometry` for inputs of `frame` (`(width, height)`: the run's tile edge when it tiles, else the frame its sources share) over `Region`s: `min_size`/`max_size` the frame's own edges, so a frame is seen at its native scale and never rescaled; the finest level's anchor size the p10 of the objects' characteristic sizes floored to a pixel, each coarser level doubling it; the ratios `gt_aspect_ratios` derives; its `source` the derivation's label. A detection run's resolution derives it over the regions it trains on and records it as `TrainGeometry.detector`, which the platform hands every detection builder; a run resolving no one frame (mixed sizes, a `dataset_source`) states the four with its reason as `model_source.detector_geometry` (`schemas.StatedDetectorGeometry`) or refuses naming both. Regions holding no object leave a record its type refuses. |
| `derive_cross_tile_nms(regions)` | `pipelines.derivations` | cross-tile merge threshold for an IoU merge, the p99 of the GT neighbor-IoU tail plus 0.05, or `None` when underivable (a pass then refuses unless `cross_tile_nms` is stated); refuses when the margin carries the neighbors' own overlap tail to 1 or past it; an IoS merge states its threshold. |
| `derive_mask_binarize(pairs)` / `mask_rings_iou(pairs, threshold)` | `pipelines.derivations` | the probability an instance segmentation head's masks are cut at, over pairs of a detection's uncut soft mask and the rings of the object it matches: each pair's own pixel-IoU peak a candidate, the one whose cuts reach the highest mean rings IoU over every pair the threshold. An untiled assessment stating `masks` true derives it over its calibration side at the fitted conf unless it states a `mask_cut`; one stating `masks` false collects no masks and derives none. Refuses when no matched mask holds a positive value. |
| `derive_vertex_spacing(pairs, threshold)` | `pipelines.derivations` | the pixels of perimeter per vertex an instance segmentation head's mask rings are simplified at, over the same pairs cut at `threshold`: each pair's largest contour perimeter over its object's largest ring's vertex count, the median over the pairs; `mask_to_polygon_rings` then keeps each ring at most the count its own perimeter implies at that spacing. An untiled assessment derives it after `mask_binarize`, the two filling its `mask_cut`, unless it states one. Refuses when no pair's mask cuts to a region beside an object ring. |
| `derive_localization_tolerance_frac(gt_boxes_per_image)` | `pipelines.derivations` | center-match tolerance, as a fraction of `gt_class_avg_size`, half the p10 of the GT's own nearest-neighbor center spacing, or `None` when no image has two-or-more of the class; refuses when that p10 is zero (a coincident pair below that share leaves it positive). An assessment's count criterion derives this once over the calibration side and reuses it for the holdout tolerance too; `TraitEntry.localization_tolerance_frac` is only the fallback when it can't. |
| `derive_sliver_frac(char_sizes)` | `pipelines.derivations` | tile-seam sliver cutoff, as a fraction of the class's characteristic size, the p10 of the GT characteristic sizes over their mean, or `None` below five sizes (too few boxes to measure a spread from). `TiledDetectionDataset` derives this itself when the caller doesn't pass an explicit `sliver_frac`, and refuses naming `tiling.sliver_frac` when too few boxes exist to derive it from. |
| `derive_tile_geometry(regions, tile_size=, overlap=)` | `pipelines.derivations` | a tile lattice's `(edge, overlap)` over `Region`s (each its objects and its pixel area, `datasets.object_region`): each stated value kept, else, with the extent the p99 of the objects' larger box side rounded up to a pixel, the edge five times the extent and the overlap the ratio under which `slicing.slice_lattice` lays neighboring tiles overlapping by at least the extent, so such an object lies whole in some tile. A run's resolution writes the lattice into its `data.tiling` once, which its tiler and spatial split read; a tile-level diagnostic resolves its own the same way, and a tiled pass that neither states nor records one derives it from its reference; refuses with no object to derive from, or an extent the edge cannot hold. |
| `derive_object_density(regions)` / `detection_cap(density, pixels)` | `pipelines.derivations` | objects per pixel at the 0.99 quantile of the densities of the regions holding at least one object, refusing when none does; a frame's detection cap is `ceil(density x its pixels)`. Training records it over the regions it serves (frames, or tiles for a tiled run); an assessment's reference derives its own. |
| `band_normalization_stats(image_paths, num_channels, max_images=, seed=)` | `pipelines.derivations` | a `schemas.Normalization`: per-band `image_mean` and `image_std` over the tensors the loader actually yields, from at most `max_images` rasters drawn at random from `seed`, its `source` the sampling record (`schemas.ImageStatsSampling`) naming what was read, or `None` when no raster could be read; `band_normalization_stats_sampled` is its windowed sibling. The statistics a model with no pretrained weights normalizes by. |
| `pretrained_normalization(weights)` | `pipelines.derivations` | a `schemas.Normalization` holding the statistics the `schemas.PretrainedWeights` were trained under, as their library publishes them (timm's `pretrained_cfg` mean and std, a torchvision weights enum's preset transform's), sourced by the weights record itself; a torchvision detection preset carries none and refuses. The statistics a model loading pretrained weights normalizes by. A model reading more bands than its weights carry statistics for states every band's mean and std instead, its source a `schemas.StatedBeyondWeights` holding the weights and the reason. |

The statistics go on the model source as their derivation returns them, never hand-assembled;
the platform hands their values to your builder beside the geometry the run's resolution
records:

```python
from tcip_mcp.pipelines.derivations import band_normalization_stats

stats = band_normalization_stats(
    train_paths, in_chans, max_images=..., seed=...
)
model_source = {
    "builder": "my_module:build_net",
    "task": "detection",
    "source_files": ["code/my_module.py"],
    "normalization": stats.model_dump(mode="json"),
}
# my_module.build_net(..., min_size, max_size, anchor_base_size, aspect_ratios,
#                     image_mean, image_std, in_chans, ...) passes them to build_detector
```

Every data-sounding `derived_from` label a value carries must map to a real implementation in
`DERIVATION_IMPLEMENTATIONS` (`pipelines.derivations`); `test_provenance_honesty` enforces it, so
a derivation label is never stamped without a computation behind it.

## The three bespoke seams (parallel; each imported, never `exec`'d)

| Seam | Where | What you supply |
|------|-------|-----------------|
| `model_source` | `pipelines.model_build.build_from_model_source` | `{"builder": "my_module:build_net", "builder_kwargs": {...}, "source_files": [...], "task": "detection", "normalization": {...}}` (`builder`, `task` and `normalization` required; `detector_geometry` stated only for a detection run resolving no one frame, and refused beside a frame the run resolves): your `nn.Module`. `source_files` names the files the run imports, the builder's module among them; the run and its checkpoints import the run's own copies of them (`pipeline-design` skill). The platform hands the builder `in_chans` (`data.num_channels`) and its head sizes: over label documents `num_classes`, the subjects the scope isolates, and `attributes`, the scope's attribute records when it declares any; over ground truth carrying its own classes the count it derives (`num_classes`, or `num_ranks` for ordinal ground truth); `image_mean`/`image_std` from `normalization` (a `schemas.Normalization`: the data's band statistics with no weights, the weights' own with them, a stated record only for bands the weights carry none for) and, when its source names them, `pretrained_weights` (the `schemas.PretrainedWeights` your builder loads); and to a detection builder `min_size`, `max_size`, `anchor_base_size` and `aspect_ratios` from the geometry the run's resolution recorded (`TrainGeometry.detector`, on the run record and every checkpoint); `builder_kwargs` restating one refuses by name. A `normalization` sourced by band statistics names only the run's own sources. `preflight_config(smoke=True)` (run by `launch_training`) builds + smokes it at the resolved dims before the training subprocess spawns. |
| `training_source` | `pipelines.training.envelope` | a dotted `train(ctx)`: your loop, handed the `TrainContext` craft library below; its module is one of `model_source.source_files`. |
| `dataset_source` | `pipelines.data.datasets.build_from_dataset_source` | `{"builder": "my_module:build_ds", "builder_kwargs": {...}, "source_files": [...]}`: a bespoke `BaseDataset` (a built-in loader or your own subclass; anything else refuses naming the builder), for a task the known loaders don't cover or for one they do. Mirrors `model_source`; the known loaders stay the default. Your builder receives `samples` (the sample list the platform's own producer named for the side being built), `scope` (the `ClassScope` they were admitted under, its subject and every attribute the registry declares for it, each with its values in declared order, both `None` where the ground-truth shape carries its own classes), `task` (the run's own, `model_source.task`) and `transforms`, and never a directory, a document path or a format flag: the platform names the samples and you build over them. `builder_kwargs` configure your builder and may not restate those four (refused where the config is validated), and `ctx.build_dataset(samples=, transforms=)` builds through the builder the run's own `data.dataset_source` names and takes nothing else. A task with no built-in loader is not a task with no producer: the platform admits by the shape of the ground truth (the images' own label documents, or the masks or table `data.labels_dir` names), whatever the task, so your builder receives the same samples a built-in loader would. |

The `model_contract` (`check_model_contract` / `overfit_check`) states the *only* model-side
contract: the measurement boundary. The model must train (finite-gradient loss) and emit inference
output the library scorers consume. Both synthesize a batch for the task strings `build_dataset`
routes; for any other task they take `sample_batch=`, an `(images, targets)` pair from your own
dataset. Given neither, `check_model_contract` returns a `not_smokeable` reason and `overfit_check`
returns `passed: False` with the reason in `issue`. `preflight_config(smoke=True)` builds that
batch from the run's `data` config and feeds it to `check_model_contract`; `overfit_check` runs on
the same batch only when `overfit=True` is also passed, since it is a voluntary diagnostic, not
part of the always-on smoke build. `smoke["batch_source"]` records which reference proved the
contract. `launch_training(overfit_check=True)` runs the diagnostic at launch, on the contract's
own batch, before the subprocess spawns, and records the result on the run's `model_contract`.

## The `ctx` craft library (`TrainContext`, `pipelines.training.envelope`)

A hand-rolled `train(ctx)` composes these instead of reimplementing them. `ctx.default_train()` is
one convenience, not a requirement.

| Group | Methods |
|-------|---------|
| model / correctness | `ctx.build_model`, `ctx.check_contract`, `ctx.overfit_check` (voluntary diagnostic, non-gating; `launch_training(overfit_check=True)` runs the same diagnostic at launch, before your `train(ctx)` ever starts), `ctx.default_train` |
| data | `ctx.train_loader`/`ctx.val_loader` (built from the partition the launch resolved), `ctx.build_dataset` (on a within-image split run, the one sample's train view, its tiles inside the recorded train region; `ctx.spatial` is that split), `ctx.tiled_dataset`, `ctx.task_collate`, `ctx.build_sampler`, `ctx.build_augmentation` |
| optimize / schedule / freeze | `ctx.build_optimizer` (the run's own stated `optimizer` block over the model's `get_param_groups`, else its `backbone` at `backbone_lr` and the rest at `head_lr`), `ctx.build_scheduler` (the run's own stated `scheduler` block, every setting handed to the torch scheduler, a plateau one watching in the selection metric's direction; step it on the run's selection value), `ctx.apply_stage_freeze` (progressive-unfreeze through the model's own `freeze_backbone` + monotonic guard: the primitive the default trainer uses; a freeze the model cannot apply refuses by name), `ctx.capture_training_state` / `ctx.restore_training_state` (a model's weights, buffers and optimizer state taken at one moment and put back into a rebuilt optimizer, matched by parameter name: what the default trainer hands each stage from its predecessor's best epoch), `ctx.compute_lr_scale`, `ctx.set_seed` (the run's stated `seed` and `deterministic`), `ctx.evaluate`, `ctx.compute_class_weights` |
| measurement | `ctx.mask_geometry`, `ctx.instance_geometries` (the operating point a delivery runs at is an assessment's, `assess_checkpoint`, never a training run's) |
| audited sinks | `ctx.log_metrics`, `ctx.log_batch` (one per-batch row at a step within an epoch, the Training tab's in-progress line, never an epoch row; the loop chooses its cadence), `ctx.save_checkpoint` (stamps the run's `config`, its `model_source` and task included, which the one predictor builds the model from, its source snapshot and the geometry its launch resolved, each refused in the saved state; a checkpoint missing any of them refuses where it is read; a payload with no `model_state_dict` is saved but refused when the run's checkpoint is registered; a `metrics` key in the saved state registers as `metrics_source="training_source"`, unverified, ranked by `rank_registered_models` only with `include_unverified=True`), `ctx.record_artifact` (copies a file in under a name of its own), `ctx.should_cancel`, `ctx.set_final_weights` (declares which saved tag is the deliverable), `ctx.report_objective` (records a `selection` row under the run's `evaluation.selection_metric`, the value an HPO trial reports) |

Route metrics and checkpoints through the sinks and the run stays audited, immutably versioned, and
provenance-snapshotted no matter what the loop does; each tag and artifact name is written once, and
every sink refuses once the run has ended. Registration additionally needs the checkpoint to be
saved by the loop: under `"model_best"`/`"model_final"`, or any tag named with
`ctx.set_final_weights`.

## Hyperparameter search: Ray Tune (`pipelines.training.hpo`, `run_hyperparameter_search`)

HPO is a capability, not a fixed algorithm; `available_search_algs()` / `available_schedulers()`
list the search algorithms and schedulers this machine has installed, and the
`run_hyperparameter_search` tool's own docstring states the sweep's on-disk layout and its
refusal shape.
An uninstalled pick (an install that skipped the hpo extra) errors clearly, never silently
swapped for another algorithm: `build_search_alg` raises `ValueError` naming the missing
backend and what's actually available. One seam the docstring doesn't carry:
`tune_search(objective_fn=..., param_space=..., …)`, every keyword by name
(`hpo.TuneSearchKeywords`), is the
bring-your-own-objective seam under `run_hyperparameter_search`. Bring your own `objective_fn(config, report)`
(call `report(value)` each step) for a search that isn't a training sweep; `sweep_dir` is
required, trial results land in that directory, never Ray's home-directory default. A training
sweep's trials are run directories like any run, listed, monitored, canceled and streamed by
their own ids, and the GUI's Training tab relaunches a recorded sweep through the same door a
run relaunches through.

## Concurrent runs: `tcip inspect-compute-resources`

Every `launch_training`/`run_hyperparameter_search` call already trains in its own OS process (crash/OOM isolation
between concurrent runs), but nothing caps how many you launch at once or how much of the host each
one claims; that's a judgment call, not a platform-enforced number (a pinned memory/CPU ceiling
would be right on one host and wrong on the next). Before launching another concurrent candidate,
`tcip inspect-compute-resources` gives you the actual facts to reason from: free VRAM
per GPU, host CPU/RAM headroom (`None` if `psutil` isn't installed; everything else still works),
and how many runs this process's own registry currently reports running.

## Auto-labeling: the proposal-engine registry (`pipelines.proposal`)

An auto-label engine turns an image into candidate shapes for a human to review. The seam is as
open as `model_source`: implement the `Proposer` protocol (`propose(image)` for whole-image
candidates) and register it with `register_proposal_engine(name, engine)` so `engine=<name>`
resolves to it, or bring one by dotted `module:factory` with no registration at all, so you can
wire, trial, and compare techniques and deduce which serves a task best by how well each engine's
high-conf proposals survive breeder review. No engine ships registered: `propose_annotations`
names its engine, and an empty or unknown name refuses, listing every name registered in this
process.

Candidates use a neutral schema (`candidate_id` / `bbox` / `area` / `rings` / `score` / `engine`
/ `engine_meta`) so the shared review/staging path stays method-agnostic. `rings` is `Polygon.rings`
(one contour per connected region), extracted by `tcip_annotation.mask_contours.mask_to_polygon_rings`
at the threshold and vertex spacing its caller states, the same extractor a predictor's masks go through, so proposal-derived GT and model predictions describe an
occlusion-split object identically.

## Active learning: the scorer registry (`pipelines.active_learning.scorer`)

An acquisition function is a capability, not a fixed menu; scorers resolve through a dict registry
you can extend rather than a welded `if/elif`. Implement `BaseScorer.score(image_paths,
predictor) -> list[tuple[str, float]]` ((path, score) pairs sorted descending, highest = most
valuable), reading the model, the device and the width to load an image at off the predictor you
are handed, and register your own (margin, least-confidence, …) with `register_scorer(name,
factory)`, the factory taking `(task, **settings)`. `resolve_scorer(block, task)` builds the
scorer a block `{"name": ..., **values}` (`pipelines.active_learning.ScorerSpec`) names, a
built-in, a registered one, or a dotted `module:factory` you wrote, from the block's own values;
an unresolvable name raises `ValueError` naming the built-ins, and the factory's own signature
refuses (`TypeError`) a value it requires that the block leaves out and one it does not take,
rather than silently substituting one.

## Classical image analysis (OpenCV)

The classical CV toolkit, `cv2` (OpenCV, the one image-processing library), is importable and yours
to compose, another set of primitives on the map, not a menu with a prescribed use. Trained ML models
are the deliverable; classical analysis is a *situational* assist. In some situations it can cheaply
produce a method's required inputs (a rough mask/box to prompt a proposer, a channel/threshold
derivation) or bootstrap soft labels for training. It has real failure modes: it doesn't generalize
across lighting/background/scale the way a trained model does, and can be confidently wrong, so the
agent judges when it fits. Any labels it produces are soft: they go through the review/validation
loop (staged as predictions, breeder-reviewed) before they are trusted as GT, never written into
a label document blind. It can also back a bring-your-own proposer (see the annotation skill).

## When the map runs out

If no existing piece fits, write the primitive: a new head, a bespoke `train(ctx)`, a second
proposal engine, a novel acquisition scorer, and register or import it. When the plain blocks and
your own primitives both plateau on a trait, research the literature (see `cv-research`) and prove
the new method beats the baseline on the measured phenotype before trusting it. Capture what you
built in a `write_retrospective`.
