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
| `preflight_config` | Validate config before training |
| `launch_training` | Start async training run (auto-launches TensorBoard) |
| `check_training_status` | Check run progress, metrics, and TensorBoard URL |
| `list_training_runs` | List all runs in session |
| `run_hpo` | HPO via random search or Optuna with TensorBoard logging |
| `render_failure_cases` | Surface + render images with worst prediction quality |
| `create_experiment` | Track training run with full lineage |

## TensorBoard

- `launch_training` automatically starts a TensorBoard process and returns the URL
- Scalars logged: `train/loss`, `train/lr`, `val/*` per epoch
- `check_training_status` includes `tensorboard_url` if TB is still running
- Training panel has an iframe that loads the TensorBoard URL

## HPO

`run_hpo` runs an Optuna TPE/ASHA search that trains each trial:

```python
run_hpo(base_config=config, n_trials=20, output_dir="runs/hpo_1")
```
- TPE sampler with ASHA/MedianPruner for early trial termination
- Per-trial TensorBoard logs in `output_dir/hpo_tensorboard/trial_{n}/`
- HParams plugin logs for side-by-side param comparison
- Auto-launches TensorBoard pointing at HPO log directory
- Returns `best_params`, `best_value`, `all_trials`, and `tensorboard` URL

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
