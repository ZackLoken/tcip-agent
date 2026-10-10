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
  - freeze_to: 2     # Unfreeze last 2 backbone layers
  - freeze_to: 0     # Full fine-tuning
```

Illustrative shape: the stage count and freeze depths above are one example, not a template;
derive them per dataset (backbone size, object difficulty, data volume) rather than pinning this
shape.

A stage carries no epoch count (a stage stating `epochs` is refused by name): it trains until
the stop rule below ends it, or until a horizon-bound schedule reaches its horizon. Each stage
has its own freeze depth and, optionally, its own
`gradient_accumulation_steps`. Learning rate is not per-stage: the top-level `optimizer` block's
`backbone_lr`/`head_lr` are stated for the first stage's target effective batch (a per-stage `lr`
key is refused by name). The physical batch is the loader's `batch_size` and stays fixed for the
run, so a later stage grows its effective batch through accumulation alone; an epoch's last
accumulation window may hold fewer batches, and its loss is averaged over the batches it holds.
Each epoch row records the stage's `target_eff_batch` (loader batch size times
accumulation) and the `optimizer_steps` the epoch actually took. An `lr_scaling: {scale_power,
max_lr}` block scales each stage's learning rates by its target effective batch over the first
stage's, raised to `scale_power`, which has no default.

The optimizer is rebuilt between stages, and each stage after the first starts from its
predecessor's best epoch whole: the weights, buffers and optimizer state of that one epoch, not
the last epoch's weights, with every parameter the predecessor's optimizer held carried over by
name (a stage's optimizer that drops one is refused). A
stage that ends with no selectable epoch (its selection value never ranked, for example a
non-finite validation metric every epoch) fails the run naming the stage rather than continuing
from its last weights. `model_best.pt` is the run's best epoch across every stage;
`model_final.pt` is the last epoch's weights.

## Early Stopping

```yaml
early_stopping:
  patience: ...   # epochs without an improvement of min_delta before a stage ends
  min_delta: ...  # the smallest change in the selection value that counts as improvement
```

The block is required and ships no default: both values are yours to set from the project in
hand, and to state with your reason. Set `min_delta` from the selection metric's own
epoch-to-epoch noise on this data (a change smaller than that noise is not an improvement), and
`patience` from the project's earlier runs where they exist (how many flat epochs preceded a
later gain); with no earlier run, choose one and say why. The rule reads the run's selection
value, which is the training loss for a run with no validation loader selecting on `loss`.

The stop rule ends a stage and the run. A stage whose selection value has not improved by
`min_delta` for `patience` epochs ends, and the next stage starts from that stage's best epoch.
The run ends completed with its last stage, or with the first stage whose best does not improve
by `min_delta` on the best of the stage before it; the stages after that one never run, and the
run's last epoch row names the stage that ended it.

A `cosine` or `onecycle` schedule is built over the `horizon_epochs` its block states, and a
stage under it ends once the schedule has run that many epochs past the stage's warmup, unless
its plateau ended it first. A `plateau` or `step` schedule has no horizon, and a stage under it
ends on its plateau alone. `stage_warmup_epochs` warms each stage after the first up from its
predecessor's rates with the schedule held, whatever the schedule.

The stop rule and `model_best.pt` share the same selection criterion; there is no separate
`metric`/`mode` key on `early_stopping`. Both are driven by `evaluation.selection_metric`
(defaults to the composite objective for detection/instance_seg, `loss` otherwise), and both
compare in whichever direction `evaluation.HIGHER_IS_BETTER_BY_METRIC` declares for that metric,
not always "lower wins": selecting on `f1` keeps the highest-F1 checkpoint, selecting on `loss`
keeps the lowest-loss one. A `selection_metric` with no declared direction is refused. For a
count trait with a center-match criterion, an explicit `selection_metric` must be one of the
trait's own governing metrics (`objective`/`f1`/`precision`/`recall`/`loss`); the map50-family
comparability metrics are rejected:

```yaml
evaluation:
  trait: catkin
  conf_threshold: ...    # a detector's: the confidence its validation counts boxes at
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
checkpoint_every_n_epochs/log_every_n_batches/early_stopping). Read that docstring rather than assuming this
example is complete. The tuned values ship no default: a launch (`preflight_config`,
`launch_training`, a sweep's every checked point and every trial) requires `batch_size`, a detector's
`evaluation.conf_threshold` under any trainer, and the `DefaultTrainerRegime` blocks (what the
default trainer reads when no `training_source` names a loop of your own), each stated by the
config or swept by a search, and refuses a config leaving one unstated, naming it. The schema
itself requires only `model_source` and `data`, so a checkpoint's config validates as what
inference reads. Every one of those keys, `evaluation` included, sits at the top level of
the config beside `model_source` and `data`. There is no `training` section: a config that
nests keys under one is refused by `preflight_config` by name, since nothing would read them.

```python
config = {
    "model_source": {  # importable nn.Module builder, see pipeline-design skill
        "builder": "my_module:build_net",
        "builder_kwargs": {"pretrained": False},  # the builder's own options; the platform hands
        "task": "detection",                      # it in_chans and the class count itself
        "source_files": ["code/my_module.py"],    # the files the run imports, the builder's
    },                                            # module (and a loop's) among them
    # "training_source": "my_module:train",  # optional custom train(ctx) loop, a bare
    #     dotted string ("module:function"), not a dict, see pipeline-design skill
    "data": {
        "images_dir": "data/images",  # each image's own label document is its ground truth
        # the subject it is admitted for; admission adds every attribute the registry declares
        "scope": {"subject": "fruit"},
        # the run's own train/val draw: its seed and its val share are both stated
        "split": {"seed": 7, "val_ratio": 0.2},  # 0.2 an example value
    },
    # each ... below is a value you tune for this dataset
    "batch_size": ...,
    "stages": [{"freeze_to": ...}],
    "optimizer": {"name": "adamw", "backbone_lr": ..., "head_lr": ..., "weight_decay": ...},
    "scheduler": {"type": "cosine", "eta_min": ..., "horizon_epochs": ...},
    "early_stopping": {"patience": ..., "min_delta": ...},
    "checkpoint_every_n_epochs": ...,
    "mixed_precision": True,
    "device": "cuda",
    "seed": 42,             # optional, reproducible init/shuffle when set
    "deterministic": False,  # optional, cuDNN deterministic algorithms (slower)
    # optional; each transform states every value its constructor takes
    "augmentation": {
        "horizontal_flip": {"p": ...},
        "random_crop": {"size": [..., ...], "min_scale": ..., "max_scale": ...},
    },
}
```

A mask or table run also names its ground truth, `data.labels_dir`: a directory of `<stem>.png`
masks or a `.csv` table of one row per image. A detection or instance run names none, since each
image's own label document is its ground truth, and neither does a run bound to a selection, whose
samples each name their own.

## Samplers

The top-level `sampler` config key is a block naming the train loader's sampler beside that
sampler's own values (`build_sampler` in `pipelines/data/samplers.py`); with no block the loader
draws every sample once an epoch in shuffled order. Registered names: `class_balanced`,
`oversample` (which states its `min_count`, the count every class is duplicated up to:
`{"name": "oversample", "min_count": ...}`), `weighted_random` (imbalance handling, weights
auto-computed from the dataset's class distribution), and `tile_locality`. A block missing a
value its sampler requires, or stating one it does not take, is refused at preflight by name.

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
naming why. Whole-frame training and whole-decode sources gain nothing from it; state no
sampler there.

## Tools

| Tool | Purpose |
|------|---------|
| `launch_training` | Start async training run (smokes the builder first, auto-launches TensorBoard); runs `training_tools.preflight_config` (`smoke=True` also builds + contract-smokes the model), a library call, not a tool of its own |
| `monitor_training` | Check a run's, a trial's or a sweep's progress by its id, with its TensorBoard URL |
| `list_experiments` | List every run of the project, and every sweep with its trial runs under it |
| `cancel_training` | Request graceful cancellation of a running run, trial or sweep by its id; a run stops at the next batch/epoch boundary and still saves `model_final.pt`; a sweep's running trials stop the same way and new trials start nothing, Ray's hard stop being only the fallback after the heartbeat window |
| `run_hyperparameter_search` | HPO on Ray Tune, you pick the search algorithm + trial scheduler |
| `tcip render-failure-cases` (command) | Surface + render images ranked by count-mismatch (not IoU-matched, see evaluation skill) |

A run launched on its own is a run directory, `.tcip/experiments/<experiment_id>/`, beside the
project's sweeps; a sweep's trial is one beneath its sweep (see HPO). Runs and sweeps share that one
directory, so a new one takes a name neither holds. A run directory is written as the run goes and never rewritten:
`run.json`, written by the launcher before the run starts (the config as
launched, its seed, environment, dataset identity, the run it was relaunched from,
and what the launch resolved: the data section, the partition it trains on and the objective it
selects by), `metrics.jsonl`, a heartbeat, and `final_status.json` once it ends (its state, its
`status_error`,
and the path and sha256 of the checkpoint its completion registers). Every checkpoint is written
once under its own name. A relaunch is a new directory naming its parent.

## TensorBoard

- `launch_training` automatically starts a TensorBoard process and returns the URL
- Scalars logged: every numeric key of each epoch row (`train_loss`, `lr`, `val_loss` and the
  other `val_` metrics, `selection`, ...) at its epoch, tagged by the key, through the run's one
  writer (`ctx.log_metrics`, which the default trainer uses too); and `batch_loss` ten
  times an epoch, at batch ends spaced evenly over that epoch's own batch count, and after
  every batch of an epoch with fewer than ten (owner ruling), or every
  `log_every_n_batches` training batches counted across the run when the config states that override
- The same per-batch emission (`ctx.log_batch`) writes a per-batch row to the run's
  `metrics.jsonl`, which the Training tab shows as the in-progress epoch's line
- `monitor_training` includes `tensorboard_url` if TB is still running; a sweep's board is its
  own directory, every trial's events beneath it
- The Training tab shows a run's epoch rows as a table beside the TensorBoard it embeds

## HPO

`run_hyperparameter_search` runs a Ray Tune sweep that trains each trial for real, toward the one
objective the sweep resolves from its base config, in that objective's own direction. The search
*algorithm*, the trial *scheduler* with its settings and the trial count are yours to choose per
task/data and to state; match them to the space and budget, since none has a default:

```python
run_hyperparameter_search(
    base_config=config, n_trials=..., search_alg="optuna", search_seed=17,
    scheduler={"name": "asha", "time_attr": "training_iteration", "max_t": ...,
               "grace_period": ..., "reduction_factor": ..., "brackets": ...,
               "stop_last_trials": ...},
    param_space={"optimizer.head_lr": {"type": "loguniform", "low": ..., "high": ...},
                 "batch_size": {"type": "categorical", "choices": [...]}})
```
- `param_space` (required) names each swept value by its config key or a dotted path into one
  (`optimizer.head_lr`), each range or choice set from the data and the model in hand. The base
  config may leave a swept value unstated; every other value a training config requires it
  states. Before the sweep starts its config is checked once at the corner of every axis's first
  value and once at each other choice and range end with the other axes held there; other
  combinations and a range's interior are checked as each trial applies its point: a trial whose
  config the schema refuses opens nothing and errors naming why, and one the launch door's other
  checks refuse (an unstated value the trainer reads, a source that will not import, a data
  location that does not exist) ends failed naming why.
- `n_trials` (required) is the number of sampled points, the trials launched being Ray's count
  over them (`hpo.planned_trial_count`).
- `search_seed` (required) seeds the search algorithm itself, native or backend, and is recorded
  in the sweep's input so a relaunch replays it; it is distinct from `data.split.seed`. The
  `17` above is an arbitrary example value.
- `search_alg`: `random`/`grid` (native), plus `optuna`, `bayesopt`, `hyperopt`, all
  installed by default. `grid` enumerates every discrete axis, and an `int` axis wider than
  `hpo.GRID_AXIS_LIMIT` values refuses by name; every other search samples an `int` axis from
  its bounds. An uninstalled or unoffered pick errors clearly (never silently swapped),
  and so does `split_draws` above one with a backend pick.
  Call `hpo.available_search_algs()` for the live list on this box.
- `scheduler`: a block naming `asha`, `hyperband`, `pbt`, `median`, or `fifo` to run every
  trial to completion (`{"name": "fifo"}`), beside every setting that scheduler's Ray Tune class
  takes, the objective and `pbt`'s mutations (the search space) excepted; a setting left
  unstated or one the class does not take is refused by name (`hpo.scheduler_settings`). Each
  trial reports once per epoch row carrying its `selection` value, so `time_attr:
  "training_iteration"` counts those epochs and `max_t` is the most epochs a trial runs under a
  halving scheduler; every scheduler's `time_attr` is stated like its other settings.
- `baseline_params` seeds the search with a point you state; `max_concurrent` bounds
  parallel trials (default 1, safe for single-GPU training).
- A sweep is its own directory, `.tcip/experiments/<sweep_id>/` beside the runs: its `sweep.json` input (the objective
  it resolved once included), a heartbeat, its final status once it ends, and one run directory
  per trial, `<sweep_id>_<ray trial id>`, a run like any other with the sampled point as its
  `trial_params` and the sweep's objective as its own. Ray's own store sits in the same directory
  (also the TensorBoard logdir); auto-launches TensorBoard. Returns the ended sweep as
  `monitor_training` reads it, `{"run", "sweep", "tensorboard_url"}`: the sweep's trial rows and
  their outcome (`best_params`/`best_value`) under `sweep`.
- `monitor_training` on a sweep's id answers the same "how is this sweep doing" question the web
  Training tab shows, for a host with no browser open: the sweep's input, every trial's own row,
  and what they amount to, from its directory alone.
- The Training tab relaunches a recorded sweep from its own input through the same door a run
  relaunches through, its body in a worker process of its own.
- Above one draw (`split_draws`), on a launch that is not a relaunch, `trial_budget` states the
  most trials the sweep may launch, checked at the door against Ray's own variant count over the
  built search space; a stated `trial_budget` is checked wherever it is stated, at one draw and on
  a relaunch alike, and a relaunch passes the record's own `trial_budget` through, so a relaunch
  whose recount no longer fits the recorded budget is refused rather than replayed.

## Dataset Selections

Use `draw_splits` to draw a selection's train, val, calibration and holdout sides. Each side but
`train` is stated as a share and `train` takes the remainder; an unstated share takes
`splits.DEFAULT_SHARES` (owner ruling, which the group draw rounds to whole groups), and the
`seed` has no default. A side the draw would leave empty refuses naming
it. A run drawing its own split takes no default share: it states `val_ratio`, and a run drawing
a within-image split (the spatial_strip route) reserves a holdout or calibration region only
where `holdout_ratio` or `calibration_ratio` states one. The samples
are drawn through the same admission a training run uses, and which admission that is depends on
what the dataset's ground truth is. A draw over the dataset's label documents (no
`ground_truth`) requires `subject`, since that admission is subject-scoped; the selection's scope
then carries every attribute the dataset's registry declares for it. `ground_truth` names ground
truth that is not label documents, and the producer reads what is there: a directory of
`<stem>.png` masks, or a `.csv` table of one row per image. A mask and a row are admitted by
existing beside their image, and the
class space a run binding such a selection trains in is derived from that ground truth by the
loader that reads it.

`calibration_ratio` and `holdout_ratio` draw the reference, held out from both training and
checkpoint selection: neither draws a loader, so an assessment (`assess_checkpoint`, see the
`evaluation` skill) fits its operating point on the `calibration` side and checks it on the
`holdout` side instead of the run's own `val`, keeping the checkpoint's own selection side out of
the number that later validates it. That reference is the selection's own held-out samples
whatever shape their ground truth is; an assessment refuses only what its delivery kind cannot
measure (a count needs detections to compare, a scalar needs table rows).
- A selection lists, per sample, the image source, the label document, a group key and a side.
  Every capture date the dataset holds enters one selection, so a trait needing examples from two
  dates trains in place: no derived folder, no copied imagery, and two dates holding a same-named
  image are two samples rather than one
- A stats-only call (no `output_path`) is the same draw, written nowhere and leaving no audit
  line; leakage-free (sibling tiles of one source image stay in the same split). A side whose
  share is zero is not drawn, written or not; a share outside `[0, 1)`, or shares leaving `train`
  nothing, refuse
- The draw refuses, before any write, when the tree holds fewer foreground groups of `subject`
  than one per requested side, counted for the draw's own subject regardless of
  `stratify_foreground`; a run's own train/val draw and a redraw inside a selection refuse by the
  same floor rather than training without validation
- `stratify_foreground=True` (default) balances splits by each source's foreground annotation
  count, not per-class distribution; the minimum-foreground floor above sees real foreground
  either way
- Reproducible from the stated seed

A run names the selection it should train against with `data.split.selection_dir` (the
`selection_dir` `draw_splits` returned). Any task can bind one whose samples carry the ground
truth that task reads: a per-image label document for detection and instance_seg, a `<stem>.png`
mask for semantic_seg, one table row for classification, ordinal and regression. A selection
naming another shape refuses by name when the run's loader is built over its samples: which
ground truth a loader reads is that loader's own fact. A selection of label
documents states its own subject and that subject's attributes, each with its values in
declared order, and the run reads them from it rather than restating them; a mask or table selection states none, and the class space its run
trains in is derived from the ground truth the run was handed, once for the run, so both its
loaders are built in one vocabulary.
A caller wanting a fixed validation side draws a selection for it rather than naming a second
directory pair: a selection carries validation membership, and a directory beside it would be a
second membership source.
`selection_dir` conflicts with a
drawn split's own parameters (`group_by`, `group_key_map`, `val_ratio`, `seed`,
`stratify_foreground`, `holdout_ratio`, `calibration_ratio`). The loaders read the
selection's `train` and `val` samples as recorded, admitting nothing afresh; its `calibration`
samples build neither loader. The run's `run.json` then records the bound membership in its
partition, with the selection it bound (the selection's directory, its digest and whether the run
redrew inside it).

`data.split.redraw_within_selection: true` beside `selection_dir` and `seed` admits `seed` (the
one conflict key it lifts) and redraws train and val fresh inside the selection's own
train-plus-val samples, at that seed, instead of binding the recorded partition; `calibration`
stays untouched and is never redrawn. A starved side (too few foreground groups among those
samples to give both train and val one) refuses by name rather than retrying or degrading.
`run_hyperparameter_search` with `split_draws` above 1 on a selection-bound `base_config` sets
this flag on its own copy, so a sweep's seed grid redraws inside the selection instead of every
trial training on its one recorded partition; `freeze_selection` still refuses a bound run,
redrawn or not, naming the reproduction for a redrawn one (bind a later run to the same selection
with the same seed and the flag, with the label documents this run's own `run.json` recorded
unchanged, since the redraw reads per-stem annotation counts at run time) rather than a fresh
freeze.

Feeding review-corrected labels back into training? A person's accepts and corrections in the
Annotate tab are written into the label documents themselves (see the `annotation` skill), so the
dataset already holds them: draw a fresh selection over it and train.
