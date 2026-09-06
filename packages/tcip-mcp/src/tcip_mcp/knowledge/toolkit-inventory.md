---
name: toolkit-inventory
description: "The composition vocabulary for the whole library: the string names build_detector / build_loss / build_model accept, the heads/necks/backbones you import, the derivations that size a model to the data, the ctx craft library a train(ctx) loop composes, the two bespoke seams (model_source / training_source / dataset_source), and the auto-label proposal-engine and active-learning scorer registries you can register your own into. Load when composing an nn.Module or train(ctx) loop, reaching for a building block and unsure of its name, wiring a bespoke dataset/proposer/scorer, or asking 'what pieces already exist so I don't rebuild one'."
---

# Toolkit inventory: the pieces to compose

The platform hands you plain importable PyTorch primitives, a few name→builder factories, and
open seams for the parts you write yourself. The tables below inventory what exists; they are a
map, not a fixed set. Adding a primitive (a new head, a new proposal engine, a new
acquisition scorer) is expected; it extends the map, it does not close it. For the build path, the
bespoke seams, and what the platform can ingest today see the `pipeline-design` skill; it supplies
no pipeline shape; this skill is the name-and-location reference its import block points at.

`build_detector` and `build_loss` already self-document at runtime: an unknown name raises
`KeyError: Unknown … Available: [...]`. The heads, necks, backbones, and ctx methods have no
such factory, so this skill is where their names live.

## Model building blocks (import and compose inside your own `nn.Module`)

| Piece | Where | Names / notes |
|-------|-------|---------------|
| `build_detector(name, adapter, num_classes, **kw)` | `pipelines.components.detectors` | names: `faster_rcnn`, `fcos`, `retinanet`, `mask_rcnn`. kwargs are per-trait, not pinned: `anchor_base_size`, `min_size`/`max_size`, `num_levels`, `featmap_names`, `image_mean`/`image_std`, plus `aspect_ratios` for every detector except `fcos`, which is anchor-free and has none. The torchvision detector class's own constructor kwargs (`box_score_thresh`, `rpn_nms_thresh`, `detections_per_img`, …) are forwarded too. Wrap a backbone+neck in `BackboneNeckAdapter` first. Unknown name → `KeyError`; a kwarg this detector doesn't take → `TypeError` naming the detectors that do. A build at `in_chans != 3` (probed from the adapter's first conv) → `ValueError` unless per-band `image_mean`/`image_std` are supplied. |
| `set_detector_operating_point(model, score_thresh=, nms_thresh=, detections_per_img=)` / `detector_operating_point_holder(model)` | `pipelines.operating_point` | Where a detector's in-model operating-point knobs live: the module itself, its `.detector.roi_heads` (two-stage), or its `.detector` (one-stage), the first that exposes any of the three attribute names (`score_thresh`, `nms_thresh`, `detections_per_img`). A bespoke, non-torchvision module reaches a validated operating point by exposing one of them on itself; the setter returns `(applied, attribute_path)`, what it applied and the holder it found. The holder refuses, naming both, when more than one location exposes a knob. |
| `build_loss(name, ...)` | `pipelines.components.losses` | names: `cross_entropy`, `weighted_ce`, `focal`, `smooth_l1`, `huber`, `bce`, `dice`, `giou`, `corn`, `coral`. Compose terms with `+` (e.g. `"bce+dice"` → `CombinedLoss`). A weightable loss (`cross_entropy`/`weighted_ce`/`focal`) auto-injects inverse-frequency weights when given `class_distribution`. `compute_class_weights` is the standalone weigher. |
| Heads (no factory, import the class) | `pipelines.components.heads` | `ClassificationHead` (task `classification`, default loss `cross_entropy`), `OrdinalHead` (task `ordinal`, `corn`), `RegressionHead` (task `regression`, `smooth_l1`), `SemanticSegHead` (task `semantic_seg`, blends CE+Dice internally). Each implements `forward` / `compute_loss` / `decode`. |
| Necks (no factory, import the class) | `pipelines.components.necks` | `FPN`, `PAN` (both take `add_p2` for a finer level), `IdentityNeck` (pass-through), `GlobalAvgPoolNeck` (→ flat `[B, C]` vector for classification/ordinal/regression heads). |
| Backbones | `pipelines.components.backbones` | `BackboneWrapper` is the whole surface: wrap a module that already emits a list/tuple/dict of feature maps (timm `features_only=True`, torchvision `create_feature_extractor`, or your own staged module, finest stage first) to get `s0..sN` naming, a declared `out_channels`, and per-stage `freeze_to`. It does not turn a classifier into a feature extractor. There is no build helper: `out_indices` decides whether you get a pyramid at all, so it is yours to choose. See the module docstring for the two-line wrap. |

`task` strings (the `model_source["task"]` value; also the `build_dataset` task keys and the
head `task_type`s): `detection`, `instance_seg`, `semantic_seg`, `classification`, `ordinal`,
`regression`. They route measurement/eval and dataset loading; a genuinely new task uses the
`dataset_source` seam below.

## Deriving a model to the data in hand (derive, don't pin)

| Derivation | Where | Returns |
|------------|-------|---------|
| `probe_channels(image_path)` | `pipelines.derivations` | band count of *this* raster (RGB→3, N-band GeoTIFF→N); feed `in_chans`. |
| `num_classes_from_distribution(dist)` | `pipelines.derivations` | `max class id + 1` from *this* label set. |
| `gt_aspect_ratios(boxes)` | `pipelines.derivations` | anchor aspect ratios spanning *this* dataset's GT box shapes (elongated organs a default `(0.5,1,2)` can't match), or `None` when the GT gives no basis. |
| `derive_cross_tile_nms(gt_boxes_per_image)` | `pipelines.derivations` | cross-tile NMS IoU from the GT neighbor-overlap tail, or `None` when underivable. |
| `derive_localization_tolerance_frac(gt_boxes_per_image)` | `pipelines.derivations` | center-match tolerance, as a fraction of `gt_class_avg_size`, from the GT's own nearest-neighbor spacing, or `None` when no image has two-or-more of the class. `resolve_operating_point` derives this once per calibration reference and reuses it for the holdout tolerance too; `TraitSpec.localization_tolerance_frac` is only the fallback when it can't. |
| `derive_sliver_frac(char_sizes)` | `pipelines.derivations` | tile-seam sliver cutoff, as a fraction of the class's characteristic size, from the GT's own size spread, or `None` below `min_samples` (default 5; too few boxes to measure a spread from). `TiledDetectionDataset` derives this itself when the caller doesn't pass an explicit `sliver_frac`. |
| `band_normalization_stats(image_paths, num_channels)` | `pipelines.derivations` | per-band `(mean, std, paths_read)` over the tensors the loader actually yields, or `None` when no raster could be read. Required for a detector at `in_chans != 3`: torchvision's 3-element ImageNet stats silently broadcast a 1-channel image to 3 and raise at any other count, so `build_detector` refuses rather than picking numbers. Pass `mean`/`std` via `builder_kwargs`, and render this result (or `band_normalization_stats_sampled`'s) through `image_stats_provenance(result)` into `model_source.image_stats_sampling` (see below), never hand-assembled: the values are not in the checkpoint, so a builder that re-derives them at load time normalizes differently at inference, and `preflight_config` refuses `image_mean`/`image_std` with no provenance record beside them. |

`gt_aspect_ratios` is derive → pass, never auto-injected into `build_detector` (that would
re-pin the method):

```python
from tcip_mcp.pipelines.derivations import gt_aspect_ratios
from tcip_mcp.pipelines.components.detectors import build_detector
ratios = gt_aspect_ratios([(b.w, b.h) for b in gt_boxes])   # this dataset's shapes
if ratios is None:
    ratios = [0.5, 1.0, 2.0]                                 # underivable, yours to stamp
names = list(adapter(torch.zeros(1, in_chans, h, w)).keys())  # this adapter's actual keys
model = build_detector("faster_rcnn", adapter, num_classes=n,
                       featmap_names=names, num_levels=len(names),
                       aspect_ratios=tuple(ratios))
```

Every data-sounding `derived_from` label a value carries must map to a real implementation in
`DERIVATION_IMPLEMENTATIONS` (`pipelines.derivations`); `test_provenance_honesty` enforces it, so
a derivation label can never again be stamped without a computation behind it.

## The three bespoke seams (parallel; each imported, never `exec`'d)

| Seam | Where | What you supply |
|------|-------|-----------------|
| `model_source` | `pipelines.model_build.build_model` | `{"builder": "my_module:build_net", "builder_kwargs": {...}, "source_files": [...], "task": "detection", "in_chans": 3, "image_stats_sampling": {...}}`: your `nn.Module`. `image_stats_sampling` is required alongside `builder_kwargs.image_mean`/`image_std`: `derivations.image_stats_provenance(result)` renders either normalization derivation's own result into the typed `schemas.ImageStatsSampling` shape (`windows`: `[[path, rect_or_null], ...]`, `seed`, `pixel_fraction`, `window_size`, `max_windows_per_image`), so the provenance travels beside the kwargs rather than inside them, never hand-assembled. `preflight_config(smoke=True)` (run by `launch_training`) builds + smokes it at the resolved dims before the training subprocess spawns. |
| `training_source` | `pipelines.training.envelope` | a dotted `train(ctx)`: your loop, handed the `TrainContext` craft library below. |
| `dataset_source` | `pipelines.data.datasets.build_from_dataset_source` | `{"builder": "my_module:build_ds", "builder_kwargs": {...}, "source_files": [...], "task": "..."}`: a bespoke `Dataset` for a task the known loaders don't cover. Mirrors `model_source`; the known loaders stay the default. |

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
| data | `ctx.build_dataset`, `ctx.tiled_dataset`, `ctx.task_collate`, `ctx.build_sampler`, `ctx.build_augmentation`, `ctx.auto_train_val` |
| optimize / schedule / freeze | `ctx.build_optimizer`, `ctx.build_scheduler`, `ctx.apply_stage_freeze` (progressive-unfreeze + monotonic guard: the primitive the default trainer uses), `ctx.compute_lr_scale`, `ctx.set_seed`, `ctx.evaluate`, `ctx.compute_class_weights` |
| measurement | `ctx.calibrate` (resolve a trait's operating point from record gate evidence: the derived, held-out-validated point), `ctx.mask_geometry`, `ctx.instance_geometries` |
| audited sinks | `ctx.log_metrics`, `ctx.save_checkpoint` (stamps kind + `model_source` + `experiment_id`; refuses a payload with no `model_state_dict`, since that stamped kind is what a predictor sniffs to load the weights; a `metrics` key in the saved state registers as `metrics_source="training_source"`, unverified, ranked by `rank_registered_models` only with `include_unverified=True`), `ctx.record_artifact` (a free-form sink for any name except the reserved `"model_weights"`, routed to `ctx.set_final_weights` instead, with a warning), `ctx.should_cancel`, `ctx.set_final_weights` (declares the deliverable checkpoint), `ctx.report_objective` (reports HPO trial progress for pruning, no-op outside HPO) |

Route metrics and checkpoints through the sinks and the run stays audited, immutably versioned, and
provenance-snapshotted no matter what the loop does. Registration additionally needs the checkpoint
to be findable: save under `"model_best"`/`"model_final"`, or call `ctx.set_final_weights` yourself.

## Hyperparameter search: Ray Tune (`pipelines.training.hpo`, `run_hyperparameter_search`)

HPO is a capability, not a fixed algorithm; see the `run_hyperparameter_search` tool's own docstring for the
search-algorithm/scheduler menu (call `available_search_algs()` / `available_schedulers()` for
what this machine actually has installed), the sweep's on-disk layout, and its refusal shape.
An uninstalled pick (an install that skipped the hpo extra) errors clearly, never silently
swapped for another algorithm: `build_search_alg` raises `ValueError` naming the missing
backend and what's actually available. One seam the docstring doesn't carry:
`tune_search(objective_fn, param_space, …)` is the
bring-your-own-objective seam under `run_hyperparameter_search`. Bring your own `objective_fn(config, report)`
(call `report(value)` each step) for a search that isn't a training sweep; `storage_path` is
required, trial results land where you say, never Ray's home-directory default.

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
candidates, `segment(image, points/box)` for one prompted mask) and register it with
`register_proposal_engine(name, engine)` so `engine=<name>` resolves to it, or bring one by dotted
`module:factory` with no registration at all, so you can wire, trial, and compare techniques and
deduce which serves a task best by how well each engine's high-conf proposals survive breeder
review. `available_engines()` is the discovery call: it lists every name
`register_proposal_engine` has registered (SAM included unconditionally; SAM's own import is
lazy, attempted only when the engine actually runs), so presence there means registration, not
a per-machine importability probe.

Candidates use a neutral schema (`candidate_id` / `bbox` / `area` / `rings` / `score` / `engine`
/ `engine_meta`) so the shared review/staging path stays method-agnostic. `rings` is `Polygon.rings`
(one contour per connected region), extracted by `tcip_annotation.mask_contours.mask_to_polygon_rings`
, the same extractor prediction export uses, so proposal-derived GT and model predictions describe an
occlusion-split object identically.

## Active learning: the scorer registry (`pipelines.active_learning.scorer`)

An acquisition function is a capability, not a fixed menu; scorers resolve through a dict registry
you can extend rather than a welded `if/elif`. Implement `BaseScorer.score(image_paths, model,
device) -> list[tuple[str, float]]` ((path, score) pairs sorted descending, highest = most
valuable) and register your own (margin, least-confidence, …) with `register_scorer(name,
factory)`. `resolve_scorer(method, task)` resolves a built-in name, a registered one, or a
dotted `module:factory` you wrote; an unresolvable name raises `ValueError` naming the
built-ins itself, rather than silently substituting one.

`require_composed_detector` (`active_learning.helpers`) is the guard the logit-reading
scorers use: it returns an error rather than reading logits off a non-`nn.Module` model, and
`feedback_tools.triage_predictions` (run through `tcip triage-predictions`) is the
kind-agnostic fallback.

## Classical image analysis (OpenCV, scikit-image)

The classical CV toolkit, `cv2` (OpenCV) and `skimage` (scikit-image), is importable and yours to
compose, another set of primitives on the map, not a menu with a prescribed use. Trained ML models
are the deliverable; classical analysis is a *situational* assist. In some situations it can cheaply
produce a method's required inputs (a rough mask/box to prompt a proposer, a channel/threshold
derivation) or bootstrap soft labels for training. It has real failure modes: it doesn't generalize
across lighting/background/scale the way a trained model does, and can be confidently wrong, so the
agent judges when it fits. Any labels it produces are soft: they go through the review/validation
loop (staged as predictions, breeder-reviewed) before they are trusted as GT, never written to
`annotations/` blind. It can also back a bring-your-own proposer (see the annotation skill).

## When the map runs out

If no existing piece fits, write the primitive: a new head, a bespoke `train(ctx)`, a second
proposal engine, a novel acquisition scorer, and register or import it. When the plain blocks and
your own primitives both plateau on a trait, research the literature (see `cv-research`) and prove
the new method beats the baseline on the measured phenotype before trusting it. Capture what you
built in a `write_retrospective`.
