---
name: annotation-workflow
description: "Iterative annotation refinement, model-assisted labeling, active learning, and annotation quality validation for plant phenotyping pipelines."
triggers:
  - annotation
  - labeling
  - label
  - SAM
  - pre-label
  - active learning
  - review
  - CVAT
modes: [PipelineDesigner, CodeGenerator]
priority: high
max_chars: 4000
---

# Annotation Workflow

## Purpose

Guide iterative annotation refinement, model-assisted labeling, active learning, and annotation quality validation.

## Iterative Refinement Loop

```
Seed annotations (manual, 50-100 images)
        ↓
Train initial model (detection/segmentation)
        ↓
Model predicts on unlabeled images
        ↓
Human reviews predictions (accept/reject/correct)  ← HITL #2
        ↓
Add reviewed images to training set
        ↓
Retrain with larger dataset
        ↓
Repeat until quality target met  ← HITL #4
```

## Model-Assisted Pre-Labeling

1. Train detector on seed set (even 50 images helps)
2. Run inference on unlabeled images
3. Filter predictions by confidence:
   - conf > 0.8: auto-accept (still human-verifiable)
   - 0.3 < conf < 0.8: present for review
   - conf < 0.3: discard
4. Human corrects errors, adds missed objects
5. Retrain — each round improves predictions

## Active Learning: Image Prioritization

Prioritize images for labeling by uncertainty:
1. **Least confident**: images where avg prediction confidence is lowest
2. **Most diverse**: images most different from current training set (embedding distance)
3. **Highest error**: images where model disagrees with itself (multi-model)

Budget: label 20-30% of total dataset, prioritized by above criteria.

## SAM-Assisted Polygon Generation

When SAM helps: clear object boundaries, large objects, instance segmentation tasks.
When SAM doesn't help: tiny objects (<20px), overlapping canopies, texture-based traits.

Pipeline: point/box prompt → SAM mask → simplify polygon → human verify.

## Annotation Quality Validation

- Cross-annotator agreement (if multiple annotators): Cohen's κ > 0.8
- Self-consistency: re-annotate 5% of images, compare IoU
- Class balance check: flag if any class has <10% of annotations
- Boundary quality: check polygon vertex density (too few = coarse, too many = noisy)

## Minimum Viable Datasets

| Task | Min Images | Min Objects/Image |
|------|-----------|-------------------|
| Detection | 100 | 5+ |
| Instance seg | 200 | 3+ |
| Classification | 50/class | — |
| Regression | 200 | — |

## Key Constraints

- Never auto-accept predictions without human review at HITL checkpoint #2
- Always validate annotation quality before training (HITL #4 for results review)
- Keep label format standard: YOLO detect/segment text files
