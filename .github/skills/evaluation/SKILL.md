---
name: evaluation
description: "Model evaluation methods, metrics interpretation, failure triage, worst-case analysis, and experiment comparison for ML models. Load when interpreting evaluation metrics, triaging or diagnosing model failures, inspecting worst predictions, or comparing experiments or checkpoints."
---

# Model Evaluation

## Metrics by Task Type

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Detection | mAP@50 | mAP@50:95, precision, recall |
| Instance Segmentation | mask mAP@50 | box mAP, mask quality (IoU distribution) |
| Classification | Accuracy | F1 (macro/weighted), confusion matrix, per-class precision/recall |
| Regression | RMSE | R², MAE, residual distribution |
| Ordinal | Quadratic weighted κ | Adjacent accuracy, confusion matrix |

Detection/instance-seg metrics (`coco_detection_metrics`) are aggregate only — no per-class AP
today. Per-class precision/recall/F1 is real for classification/ordinal (`evaluate_model`).
Change detection is not a built task type — see README's Roadmap.

## Tools

| Tool | Purpose |
|------|---------|
| `score_predictions` | Score on-disk predictions vs GT — an image file returns per-box matches (`detail=True` adds a per-detection breakdown); a dataset dir returns aggregate metrics + per-image TP/FP/FN |
| `render_failure_cases` | Surface + render the N images with highest triage error |
| `compare_experiments` | Side-by-side metrics across experiments |
| `get_experiment` (`view='lineage'`) | Trace data → model → predictions chain |

## Failure Triage

When metrics are poor, investigate systematically:

1. **Data issues**: `validate_data_quality` — check for missing labels, format errors, class imbalance
2. **Worst cases**: `render_failure_cases` — surface and visually inspect the worst N images
3. **Per-image breakdown**: `score_predictions` on a dataset dir — find images with the
   highest FP/FN counts (no built-in per-class breakdown for detection; use
   `score_predictions(<image>, detail=True)` per image and aggregate by `class_id` if
   class-level numbers are needed)
4. **Training dynamics**: Check metrics.jsonl — is loss still decreasing? Overfitting?
5. **Architecture**: Is the model appropriate for the task and data scale?

## Comparison Protocol

When comparing models:
1. Same dataset split (use `make_splits` with fixed seed)
2. Same evaluation set
3. Compare on primary metric for the task
4. For classification/ordinal, check per-class performance — overall accuracy can hide
   class-specific failures; for detection, check per-image FP/FN patterns instead (see
   Failure Triage above — there's no per-class AP today)
5. Use `compare_experiments` for side-by-side analysis
