---
name: training
description: "Training configuration, progressive unfreezing, early stopping, HPO, and experiment tracking for ML model training. Load when configuring or launching a training run, setting up hyperparameter optimization, or tracking and comparing training experiments."
---

# Training Configuration

## Progressive Unfreezing

Multi-stage training for transfer learning:

```yaml
stages:
  - freeze_to: -1    # Freeze entire backbone, train head only
    epochs: 5
  - freeze_to: 2     # Unfreeze last 2 backbone layers
    epochs: 10
  - freeze_to: 0     # Full fine-tuning
    epochs: 10
```

Illustrative shape: the stage count, freeze depths, and epoch counts above are one example, not
a template; derive them per dataset (backbone size, object difficulty, data volume) rather than
pinning this shape.

Each stage has its own epoch count and freeze depth. Learning rate is not per-stage: the
top-level `optimizer` block's `backbone_lr`/`head_lr` apply uniformly across every stage (a
per-stage `lr` key is refused by name). The
optimizer is rebuilt between stages.

## Early Stopping

```yaml
early_stopping:
  enabled: true
  patience: 7        # Epochs without improvement before stopping
  min_delta: 0.0001  # Minimum change to count as improvement
```

Illustrative shape: the `patience` and `min_delta` values above are one example; derive or tune
them per dataset's own convergence noise, never pinned.

Early stopping and `model_best.pt` share the same selection criterion; there is no separate
`metric`/`mode` key on `early_stopping`. Both are driven by `evaluation.selection_metric`
(defaults to the composite objective for detection/instance_seg, `val_loss` otherwise), and both
compare in whichever direction `evaluation.HIGHER_IS_BETTER_BY_METRIC` declares for that metric,
not always "lower wins": selecting on `f1` keeps the highest-F1 checkpoint, selecting on `loss`
keeps the lowest-loss one. A `selection_metric` with no declared direction is refused. For a
count trait with a center-match criterion, an explicit `selection_metric` must be one of the
trait's own governing metrics (`objective`/`f1`/`precision`/`recall`/`loss`); the map50-family
comparability metrics are rejected:

```yaml
evaluation:
  trait: catkin
  selection_metric: f1   # optional, omit to use the task's default (objective/loss)
```

## Config Structure

`model_source` points at the importable builder for your agent-written `nn.Module` (add
`training_source` for a custom `train(ctx)` loop); see how you build the model and import
the plain blocks in `pipeline-design/SKILL.md`; don't re-derive it here.

The example below is representative, not exhaustive; the config is an open dict, and
`generic_trainer.train()`'s own docstring is the canonical, always-current list of every key it
reads (device/seed/deterministic/mixed_precision/stages/optimizer/scheduler/lr_scaling/
stage_warmup_epochs/enforce_monotonic_unfreeze/gradient_accumulation_steps/
checkpoint_every_n_epochs/early_stopping). Read that docstring rather than assuming this
example is complete. Every one of those keys, `evaluation` included, sits at the top level of
the config beside `model_source` and `data`. There is no `training` section: a config that
nests keys under one is refused by `preflight_config` by name, since nothing would read them.

```python
config = {
    "model_source": {  # importable nn.Module builder, see pipeline-design skill
        "builder": "my_module:build_net",
        "builder_kwargs": {"in_chans": 3, "num_classes": 3},
        "task": "detection",
    },
    # "training_source": "my_module:train",  # optional custom train(ctx) loop, a bare
    #     dotted string ("module:function"), not a dict, see pipeline-design skill
    "data": {
        "images_dir": "data/images",
        # where this run's ground truth lives, whatever shape it is: a directory of per-image
        # label documents, a directory of <stem>.png masks, or a .csv table of one row per image
        "labels_dir": "data/labels/detect",
        "task": "detection"
    },
    "batch_size": 4,
    "stages": [...],
    "mixed_precision": True,
    "device": "cuda",
    "seed": 42,             # optional, reproducible init/shuffle when set
    "deterministic": False,  # optional, cuDNN deterministic algorithms (slower)
    "augmentation": {
        "horizontal_flip": 0.5,
        "random_crop": {"min_scale": 0.8}
    }
}
```

## Samplers

The top-level `sampler` config key picks the train loader's sampling strategy by name
(`build_sampler` in `pipelines/data/samplers.py`). Registered names: `random` (the default:
plain DataLoader shuffle), `class_balanced`, `oversample`, `weighted_random` (imbalance
handling, weights auto-computed from the dataset's class distribution), and `tile_locality`.

`tile_locality` matters for windowed tiled training on full-width strip-layout rasters:
there a fully shuffled tile order forces the same strips to be decoded over and over, since
every tile in a row shares its row's strips and the block cache evicts them between visits.
It keeps each reading process inside contiguous bands of tile rows (band height derived at
construction from the per-reader GDAL cache share and the source's row byte cost) while
still shuffling sources, bands, and tiles within a band each epoch. Under multi-worker
loading it deals bands onto per-worker lanes and interleaves them in batches, matching the
DataLoader's round-robin batch dispatch, so every worker keeps its own banded read stream.
It consumes the loader context (`num_workers`, and `batch_size` when workers > 1) and
requires a tiled dataset over at least one windowed source; it refuses anything else,
naming why. Whole-frame training and whole-decode sources gain nothing from it; keep
`random` there.

## Tools

| Tool | Purpose |
|------|---------|
| `launch_training` | Start async training run (smokes the builder first, auto-launches TensorBoard); runs `training_tools.preflight_config` (`smoke=True` also builds + contract-smokes the model), a library call, not a tool of its own |
| `monitor_training(experiment_id=...)` | Check run progress, metrics, and TensorBoard URL |
| `monitor_training(sweep_id=...)` | Check one sweep's manifest and per-trial state from disk, exactly one of `experiment_id`/`sweep_id` |
| `list_experiments(launched_only=True)` | List all runs in session |
| `cancel_training` | Request graceful cancellation of a running run; stops at the next batch/epoch boundary, still saves `model_final.pt` |
| `cancel_hyperparameter_search` | Request cooperative cancellation of a running sweep: the running trial stops at its next batch boundary and reports the losing side, new trials report without training, the manifest records `canceled`; Ray's hard stop is only the fallback after the heartbeat window |
| `run_hyperparameter_search` | HPO on Ray Tune, you pick the search algorithm + trial scheduler |
| `tcip render-failure-cases` (logged command) | Surface + render images ranked by count-mismatch (not IoU-matched, see evaluation skill) |
| `create_experiment` | Track training run with full lineage |

## TensorBoard

- `launch_training` automatically starts a TensorBoard process and returns the URL
- Scalars logged: `train/loss`, `train/lr`, `val/*` per epoch
- `monitor_training` includes `tensorboard_url` if TB is still running
- Training panel has an iframe that loads the TensorBoard URL

## HPO

`run_hyperparameter_search` runs a Ray Tune sweep that trains each trial for real (minimizing the composite
selection objective). The search *algorithm* and trial *scheduler* are yours to choose per
task/data; match them to the space and budget; the defaults are a starting point, not a rule:

```python
run_hyperparameter_search(base_config=config, n_trials=20, search_alg="optuna", scheduler="asha",
        output_dir="runs/hpo_1", search_seed=17)
```
- `search_seed` (required) seeds the search algorithm itself, native or backend, and is recorded
  on the sweep's manifest so a relaunch replays it; it is distinct from `data.split.seed`. The
  `17` above is an arbitrary example value.
- `search_alg`: `random`/`grid` (native), plus `optuna`, `bayesopt`, `hyperopt`, all
  installed by default. An uninstalled or unoffered pick errors clearly (never silently swapped),
  and so does `split_draws` above one with a backend pick.
  Call `hpo.available_search_algs()` for the live list on this box.
- `scheduler`: `asha`, `hyperband`, `pbt`, `median`, or `none` to run every trial to
  completion. `grace_period`/`reduction_factor` tune the halving schedulers.
- `warm_start=True` seeds the search with a known-good baseline; `max_concurrent` bounds
  parallel trials (default 1, safe for single-GPU training).
- Ray persists trials under `output_dir` (also the TensorBoard logdir); auto-launches
  TensorBoard. Returns `best_params`, `best_value`, `all_trials`, and `tensorboard` URL.
- `monitor_training(sweep_id=...)` answers the same "how is this sweep doing" question the web
  Tuning tab reads, for a host with no browser open: the sweep's manifest plus every trial's own
  params and whether it has logged metrics, from disk alone (no live jobstore, so a sweep just
  launched over HTTP but not yet manifested reads as not found).
- Above one draw (`split_draws`), on a launch that is not a relaunch, `trial_budget` states the
  most trials the sweep may launch, checked at the door against Ray's own variant count over the
  built search space; a stated `trial_budget` is checked wherever it is stated, at one draw and on
  a relaunch alike, and the Tuning relaunch route passes the record's own `trial_budget` through,
  so a relaunch whose recount no longer fits the recorded budget is refused rather than replayed.

## Dataset Selections

Use `draw_splits` to draw a train/val/calibration selection: `draw_splits` has no `test_ratio`
parameter at all, no launch path honors a held-out test list (a separate, within-image
mechanism, `reserve_calibration_fraction` on the spatial_strip route, not this one). The samples
are drawn through the same admission a training run uses, and which admission that is depends on
where the dataset's ground truth lives. Writing a selection over the per-image label tree
(`output_path` given, no `ground_truth`) requires `subject`, since that admission is
subject-scoped, and takes `attribute` when the run is attribute-scoped. `ground_truth` names a
place explicitly instead of walking that tree, and the producer reads what is there: a directory
of label documents (subject-scoped the same way), a directory of `<stem>.png` masks, or a `.csv`
table of one row per image. A mask and a row are admitted by existing beside their image, and the
class space a run binding such a selection trains in is derived from that ground truth by the
loader that reads it.

`calibration_ratio` is a third side, held out from both training and checkpoint selection: it
draws no loader, so `evaluate_model` and delivery calibration read it as their reference universe
instead of the run's own `val`, keeping the checkpoint's own selection side out of the number
that later validates it. That universe is the selection's own held-out samples whatever shape
their ground truth is; each calibration door refuses only what its own measurement cannot do (a
count door needs detections to compare, a scalar door needs table rows), and no door refuses a
shape on the universe's behalf.
- A selection lists, per sample, the image source, the label document, a group key and a side.
  Every capture date the dataset holds enters one selection, so a trait needing examples from two
  dates trains in place: no derived folder, no copied imagery, and two dates holding a same-named
  image are two samples rather than one
- A stats-only call (no `output_path`) defaults to `train_ratio=0.8`, `val_ratio=0.2`,
  `calibration_ratio=0.0`; leakage-free (sibling tiles of one source image stay in the same
  split). Writing a selection has no default for any of the three and refuses a zero one, naming
  it: state all three ratios explicitly
- The draw refuses, before any write, when the tree holds fewer foreground groups of `subject`
  than the requested sides need at minimum, counted for the draw's own subject regardless of
  `stratify_foreground`
- `stratify_foreground=True` (default) balances splits by each source's foreground annotation
  count, not per-class distribution; the minimum-foreground floor above sees real foreground
  either way
- Reproducible with random seed

A run names the selection it should train against with `data.split.selection_dir` (the
`selection_dir` `draw_splits` returned). Any task can bind one whose samples carry the ground
truth that task reads: a per-image label document for detection and instance_seg, a `<stem>.png`
mask for semantic_seg, one table row for classification, ordinal and regression. A selection
naming another shape refuses by name when the run's loader is built over its samples: which
ground truth a loader reads is that loader's own fact. A selection of label
documents states its own subject, attribute and class-id map, and the run reads them from it
rather than restating them; a mask or table selection states none, and the class space its run
trains in is derived from the ground truth the run was handed, once for the run, so both its
loaders are built in one vocabulary.
A caller wanting a fixed validation side draws a selection for it rather than naming a second
directory pair: a selection carries validation membership, and a directory beside it would be a
second membership source.
`selection_dir` conflicts with a
drawn split's own parameters (`group_by`, `group_key_map`, `val_ratio`, `seed`,
`stratify_foreground`, `test_ratio`, `reserve_calibration_fraction`). The loaders read the
selection's `train` and `val` samples as recorded, admitting nothing afresh; its `calibration`
samples build neither loader. The run's `split.json` then records the bound membership plus a
`selection_binding` block (the selection's directory, its digest and whether the run redrew
inside it, never a second copy of the sample list).

`data.split.redraw_within_selection: true` beside `selection_dir` and `seed` admits `seed` (the
one conflict key it lifts) and redraws train and val fresh inside the selection's own
train-plus-val samples, at that seed, instead of binding the recorded partition; `calibration`
stays untouched and is never redrawn. A starved side (too few foreground groups among those
samples to give both train and val one) refuses by name rather than retrying or degrading.
`run_hyperparameter_search` with `split_draws` above 1 on a selection-bound `base_config` sets
this flag on its own copy, so a sweep's seed grid redraws inside the selection instead of every
trial training on its one recorded partition; `freeze_selection` still refuses a bound run,
redrawn or not, naming the reproduction for a redrawn one (bind a later run to the same selection
with the same seed and the flag, with the labels this run's own `split.json` recorded unchanged,
since the redraw reads per-stem annotation counts at run time) rather than a fresh freeze.

Feeding review-corrected labels back into training? `materialize_review_dataset` (see the
`annotation` skill) builds the curated dataset from review verdicts before you split/train.
Curation is your job: before training on review verdicts, materialize a curated set via
`materialize_review_dataset` if none exists yet.
