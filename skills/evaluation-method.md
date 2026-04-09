---
name: evaluation-method
description: "Metric selection, result interpretation, failure triage, ablation study design, and statistical significance for model evaluation in breeding programs."
triggers:
  - evaluate
  - metric
  - mAP
  - F1
  - accuracy
  - confusion matrix
  - ablation
  - failure
  - retrain
  - compare
modes: [ResultsAnalyzer, TrainingOrchestrator]
priority: high
max_chars: 4000
---

# Evaluation Method

## Purpose

Select metrics, interpret results, diagnose failures, and design ablation studies for model evaluation. Guide evidence-based decisions on retraining vs HPO vs data collection.

## Metric Selection by Task

| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| Detection | mAP@0.5 | mAP@0.5:0.95, recall |
| Instance seg | mask mAP@0.5 | box mAP, boundary IoU |
| Classification | macro F1 | per-class F1, confusion matrix |
| Ordinal classification | QWK (quadratic weighted kappa) | MAE, Spearman ρ |
| Regression | R² | MAE, RMSE, Spearman ρ |
| Semantic seg | mIoU | per-class IoU, boundary F1 |

## Per-Class Analysis

Always break down metrics per class. Flag classes with:
- F1 < 0.5 (poor performance)
- Support < 20 (insufficient test data)
- >10% gap between precision and recall (bias)

## Failure Case Triage

Examine the worst 20 predictions. Classify each failure:

| Failure Type | Fix |
|-------------|-----|
| Annotation error | Correct labels, retrain |
| Ambiguous ground truth | Clarify guidelines, re-annotate |
| Distribution shift | Collect more diverse data |
| Small object missed | Add tiling, lower anchor sizes |
| Confusion between classes | More training data for confused classes |
| Overfitting (train >> val) | Regularization, more data, augmentation |

## Decision: Retrain vs HPO vs More Data

```
Val metric below target?
├── Training loss also high → model too small or LR issue → HPO
├── Training loss low, val high → overfitting → more data / regularization
├── Both reasonable, just below target → HPO (fine-tune hyperparams)
└── Large class imbalance → more data for underrepresented classes
```

## Ablation Study Design

Test one variable at a time. Standard ablations:
1. **Backbone**: ResNet-50 vs MobileNetV2 vs ResNet-101
2. **Augmentation**: with vs without horizontal flip, resize, color jitter
3. **Training stages**: 2-stage vs 3-stage vs 4-stage unfreezing
4. **LR**: 3 values spanning an order of magnitude

Report: table of (configuration, metric, Δ from baseline). Include training time.

## Statistical Significance for Breeding

Breeders rank genotypes — relative performance matters more than absolute.
- Spearman rank correlation between predicted and actual rankings
- If ρ > 0.85: predictions are useful for selection decisions
- If ρ < 0.70: model needs improvement before deployment

## Key Constraints

- HITL checkpoint #4: present results with metrics, failure cases, and recommendation
- Never deploy a model without per-class analysis
- Always report confidence intervals when dataset is small (<200 test images)
- Document negative results (what didn't work and why)
