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

Each stage has its own epoch count and freeze depth. Learning rate is not per-stage: the
top-level `optimizer` block's `backbone_lr`/`head_lr` apply uniformly across every stage (a
per-stage `lr` key is accepted but ignored; `preflight_config` warns if you set one). The
optimizer is rebuilt between stages.

## Early Stopping

```yaml
early_stopping:
  enabled: true
  patience: 7        # Epochs without improvement before stopping
  min_delta: 0.0001  # Minimum change to count as improvement
```

Early stopping and `model_best.pt` share the same selection criterion; there is no separate
`metric`/`mode` key on `early_stopping`. Both are driven by `evaluation.selection_metric`
(defaults to the composite objective for detection/instance_seg, `val_loss` otherwise). For a
count trait with a center-match criterion, an explicit `selection_metric` must be one of the
trait's own governing metrics (`objective`/`f1`/`precision`/`recall`/`loss`); the map50-family
comparability metrics are rejected, since selecting checkpoints by a metric the trait doesn't
trust is a defensibility regression:

```yaml
evaluation:
  trait: catkin
  selection_metric: f1   # optional, omit to use the task's default (objective/loss)
```

## Config Structure

`model_source` points at the importable builder for your agent-written `nn.Module` (add
`training_source` for a custom `train(ctx)` loop); see how you build the model and import
the plain blocks in `pipeline-design/SKILL.md`; don't re-derive it here.

The example below is representative, not exhaustive; `training` is an open dict, and
`generic_trainer.train()`'s own docstring is the canonical, always-current list of every key it
reads (device/seed/deterministic/mixed_precision/stages/optimizer/scheduler/lr_scaling/
stage_warmup_epochs/enforce_monotonic_unfreeze/gradient_accumulation_steps/
checkpoint_every_n_epochs/early_stopping/evaluation). Read that docstring rather than assuming this
example is complete.

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
        "labels_dir": "data/labels/detect",
        "task": "detection"
    },
    "training": {
        "batch_size": 4,
        "stages": [...],
        "mixed_precision": True,
        "device": "cuda",
        "seed": 42,             # optional, reproducible init/shuffle when set
        "deterministic": False  # optional, cuDNN deterministic algorithms (slower)
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "random_crop": {"min_scale": 0.8}
    }
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `preflight_config` | Validate config (`smoke=True` also builds + contract-smokes the model) |
| `launch_training` | Start async training run (smokes the builder first, auto-launches TensorBoard) |
| `check_training_status` | Check run progress, metrics, and TensorBoard URL |
| `list_training_runs` | List all runs in session |
| `cancel_training` | Request graceful cancellation of a running run; stops at the next batch/epoch boundary, still saves `model_final.pt` |
| `run_hpo` | HPO on Ray Tune, you pick the search algorithm + trial scheduler |
| `render_failure_cases` | Surface + render images ranked by count-mismatch (not IoU-matched, see evaluation skill) |
| `create_experiment` | Track training run with full lineage |

## TensorBoard

- `launch_training` automatically starts a TensorBoard process and returns the URL
- Scalars logged: `train/loss`, `train/lr`, `val/*` per epoch
- `check_training_status` includes `tensorboard_url` if TB is still running
- Training panel has an iframe that loads the TensorBoard URL

## HPO

`run_hpo` runs a Ray Tune sweep that trains each trial for real (minimizing the composite
selection objective). The search *algorithm* and trial *scheduler* are yours to choose per
task/data; match them to the space and budget; the defaults are a starting point, not a rule:

```python
run_hpo(base_config=config, n_trials=20, search_alg="optuna", scheduler="asha",
        output_dir="runs/hpo_1")
```
- `search_alg`: `random`/`grid` (native) or a backend when installed: `optuna`, `bayesopt`,
  `hyperopt`, `nevergrad`, `ax`, `hebo`, `zoopt`, `bohb`. An uninstalled pick errors clearly
  (never silently swapped). Call `hpo.available_search_algs()` for the live list on this box.
- `scheduler`: `asha`, `hyperband`, `bohb` (pair with the `bohb` searcher), `pbt`, `median`,
  or `none` to run every trial to completion. `grace_period`/`reduction_factor` tune the
  halving schedulers.
- `warm_start=True` seeds the search with a known-good baseline; `max_concurrent` bounds
  parallel trials (default 1, safe for single-GPU training).
- Ray persists trials under `output_dir` (also the TensorBoard logdir); auto-launches
  TensorBoard. Returns `best_params`, `best_value`, `all_trials`, and `tensorboard` URL.

## Dataset Splits

Use `make_splits` to create train/val/test splits:
- Default: 70/20/10, leakage-free (sibling tiles of one source image stay in the same split)
- `stratify_foreground=True` (default) balances splits by each source's foreground annotation
  count, not per-class distribution
- `materialize=True` also lays out a `{train,val,test}/{images,labels}/` tree; the labels
  inside are the platform's own per-image JSON, not YOLO's `.txt` format
- Reproducible with random seed

Feeding review-corrected labels back into training? `materialize_review_dataset` (see the
`annotation` skill) builds the curated dataset from review verdicts before you split/train.
Curation is your job: before training on review verdicts, materialize a curated set via
`materialize_review_dataset` if none exists yet.
