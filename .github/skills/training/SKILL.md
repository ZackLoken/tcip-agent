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
    lr: 1e-3
  - freeze_to: 2     # Unfreeze last 2 backbone layers
    epochs: 10
    lr: 1e-4
  - freeze_to: 0     # Full fine-tuning
    epochs: 10
    lr: 1e-5
```

Each stage has its own learning rate, epoch count, and freeze depth. The optimizer is rebuilt between stages.

## Early Stopping

```yaml
early_stopping:
  enabled: true
  patience: 7        # Epochs without improvement before stopping
  metric: val_loss   # Metric to monitor
  mode: min          # min for loss, max for mAP
```

## Config Structure

`model_source` points at the importable builder for your agent-written `nn.Module` (add
`training_source` for a custom `train(ctx)` loop) — see how you build the model and import
the plain blocks in `pipeline-design/SKILL.md`; don't re-derive it here.

```python
config = {
    "model_source": {  # importable nn.Module builder — see pipeline-design skill
        "builder": "my_module:build_net",
        "builder_kwargs": {"in_chans": 3, "num_classes": 3},
        "task": "detection",
    },
    # "training_source": {"train": "my_module:train"},  # optional custom loop
    "data": {
        "images_dir": "data/images",
        "labels_dir": "data/labels/detect",
        "task": "detection"
    },
    "training": {
        "batch_size": 4,
        "stages": [...],
        "mixed_precision": True,
        "device": "cuda"
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
| `run_hpo` | HPO on Ray Tune — you pick the search algorithm + trial scheduler |
| `render_failure_cases` | Surface + render images with worst prediction quality |
| `create_experiment` | Track training run with full lineage |

## TensorBoard

- `launch_training` automatically starts a TensorBoard process and returns the URL
- Scalars logged: `train/loss`, `train/lr`, `val/*` per epoch
- `check_training_status` includes `tensorboard_url` if TB is still running
- Training panel has an iframe that loads the TensorBoard URL

## HPO

`run_hpo` runs a Ray Tune sweep that trains each trial for real (minimizing the composite
selection objective). The search *algorithm* and trial *scheduler* are yours to choose per
task/data — match them to the space and budget; the defaults are a starting point, not a rule:

```python
run_hpo(base_config=config, n_trials=20, search_alg="optuna", scheduler="asha",
        output_dir="runs/hpo_1")
```
- `search_alg`: `random`/`grid` (native) or a backend when installed — `optuna`, `bayesopt`,
  `hyperopt`, `nevergrad`, `ax`, `hebo`, `zoopt`, `bohb`. An uninstalled pick errors clearly
  (never silently swapped). Call `hpo.available_search_algs()` for the live list on this box.
- `scheduler`: `asha`, `hyperband`, `bohb` (pair with the `bohb` searcher), `pbt`, `median`,
  or `none` to run every trial to completion. `grace_period`/`reduction_factor` tune the
  halving schedulers.
- `warm_start=True` seeds the search with a known-good baseline; `max_concurrent` bounds
  parallel trials (default 1 — safe for single-GPU training).
- Ray persists trials under `output_dir` (also the TensorBoard logdir); auto-launches
  TensorBoard. Returns `best_params`, `best_value`, `all_trials`, and `tensorboard` URL.

## Dataset Splits

Use `make_splits` to create train/val/test splits:
- Default: 70/20/10, leakage-free (sibling tiles of one source image stay in the same split)
- `stratify_foreground=True` (default) balances splits by each source's foreground annotation
  count — not per-class distribution
- `materialize=True` also lays out a YOLO `{train,val,test}/{images,labels}/` tree
- Reproducible with random seed

Feeding review-corrected labels back into training? `materialize_review_dataset` (see the
`annotation` skill) builds the curated dataset from review verdicts before you split/train.
Curation is your job: before training on review verdicts, materialize a curated set via
`materialize_review_dataset` if none exists yet.
