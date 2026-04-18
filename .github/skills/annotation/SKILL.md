---
name: annotation
description: "Annotation and review workflows for YOLO, COCO, PASCAL VOC, and LabelMe format image data. Covers SAM-assisted labeling, review cycles with IoU matching, active learning scoring, and quality metrics."
---

# Annotation Workflow

## Supported Formats

| Format | Files | Coordinates | Auto-detected by |
|--------|-------|------------|------------------|
| **YOLO** | One `.txt` per image | Normalized [0,1] | `.txt` extension |
| **COCO** | Single `.json` for dataset | Pixel coordinates | `.json` + content keys |
| **PASCAL VOC** | One `.xml` per image | Pixel coordinates | `.xml` extension |
| **LabelMe** | One `.json` per image | Pixel coordinates | `.json` + `"shapes"` key |

Use `detect_format()` from `tcip_annotation.format_io` to auto-detect.
Use `load_annotations` / `save_annotations` from `format_io` for format-agnostic I/O.

## Stages

1. **Initial labeling** — Manual or SAM-assisted bounding box and polygon annotation
2. **Review** — IoU matching between predictions and ground truth to accept/correct/reject
3. **Active learning** — Score unlabeled images by model uncertainty to prioritize annotation effort
4. **Quality audit** — Per-class AP, coverage analysis, inter-annotator agreement

## Tools

| Tool | Purpose |
|------|---------|
| `load_annotations` | Load labels for a set of images (auto-detects format) |
| `save_annotations` | Write annotations to any supported format |
| `sam_predict` | SAM-assisted polygon generation from point/box prompts |
| `run_matching` | Match predictions to ground truth by IoU |
| `evaluate_detections` | Compute precision/recall/AP from matched pairs |
| `push_panel_data` | Send images + annotations to the annotation or review panel |
| `score_unlabeled` | Rank unlabeled images by model uncertainty |
| `get_review_queue` | Get prioritized list of images needing review |

## SAM-Assisted Labeling

1. User clicks a point or draws a rough box on the annotation canvas
2. `sam_predict` generates a precise polygon mask
3. Polygon is saved in the project's configured annotation format
4. Supports point prompts (positive/negative) and box prompts

## Review Protocol

1. Load ground truth with `load_annotations`
2. Load predictions (from inference or prior annotation)
3. `run_matching` pairs predictions to GT by IoU (default threshold: 0.5)
4. `evaluate_detections` computes TP/FP/FN per class
5. Review in panel: accept correct predictions, correct errors, add missed objects

## Quality Metrics

- **Per-class AP** at IoU 0.5 and 0.5:0.95
- **Coverage**: fraction of images with labels
- **Empty files**: valid negative samples — never delete without asking
- **Cohen's κ**: inter-annotator agreement (if multiple annotators)

## Active Learning

`score_unlabeled` ranks images by model uncertainty:
- High uncertainty = model unsure = most valuable to annotate
- Supports entropy, margin, and committee disagreement scoring
- `get_review_queue` returns a prioritized list for the annotator
