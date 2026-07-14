---
name: evaluation
description: "Model evaluation methods, metrics interpretation, failure triage, worst-case analysis, and experiment comparison for ML models. Load when interpreting evaluation metrics, triaging or diagnosing model failures, inspecting worst predictions, or comparing experiments or checkpoints."
---

# Model Evaluation

## Metrics by Task Type

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Detection | mAP@50 | mAP@50:95, precision, recall, per-class AP |
| Instance Segmentation | mask mAP@50 | box mAP, mask quality (IoU distribution) |
| Classification | Accuracy | F1 (macro/weighted), confusion matrix, per-class precision/recall |
| Regression | RMSE | R², MAE, residual distribution |
| Ordinal | Quadratic weighted κ | Adjacent accuracy, confusion matrix |
| Change Detection | F1-score | Temporal consistency, event date accuracy |

## Tools

| Tool | Purpose |
|------|---------|
| `evaluate_detections` | Compute precision/recall/AP from matched boxes |
| `evaluate_dataset` | Full dataset evaluation with per-class breakdown |
| `get_worst_predictions` | Find N images with highest error |
| `compare_experiments` | Side-by-side metrics across experiments |
| `get_experiment_lineage` | Trace data → model → predictions chain |

## Failure Triage

When metrics are poor, investigate systematically:

1. **Data issues**: `validate_data_quality` — check for missing labels, format errors, class imbalance
2. **Worst cases**: `get_worst_predictions` — visually inspect the worst N images
3. **Class breakdown**: `evaluate_dataset` — identify which classes are failing
4. **Training dynamics**: Check metrics.jsonl — is loss still decreasing? Overfitting?
5. **Architecture**: Is the model appropriate for the task and data scale?

## Comparison Protocol

When comparing models:
1. Same dataset split (use `split_dataset` with fixed seed)
2. Same evaluation set
3. Compare on primary metric for the task
4. Check per-class performance — overall mAP can hide class-specific failures
5. Use `compare_experiments` for side-by-side analysis
