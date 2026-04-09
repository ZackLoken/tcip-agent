---
name: data-format
description: "YOLO label format rules, directory structure conventions, image naming, prediction output format, and data validation for detection and segmentation tasks."
triggers:
  - label format
  - YOLO
  - txt
  - bounding box
  - polygon
  - data directory
  - data structure
  - image naming
  - prediction
  - data validation
modes: [PipelineDesigner, CodeGenerator]
priority: high
max_chars: 3000
---

# Data Format

## Purpose

Define the exact file format contracts for labels, predictions, images, and directory layout. The agent must produce and consume data in these formats.

## Directory Structure

```
data/
  images/           # all images (JPEG/PNG), flat
  labels/
    detect/         # YOLO detection labels
    segment/        # YOLO segmentation labels
  predictions/
    detect/         # model detection predictions
    segment/        # model segmentation predictions
```

Image and label files share the same stem: `IMG_0133.jpg` → `IMG_0133.txt`.

## YOLO Detection Format

One line per object. Space-separated: `class_id cx cy w h`

All values normalized to [0, 1] relative to image dimensions:
- `cx, cy` = center x, center y
- `w, h` = width, height
- `class_id` = integer starting at 0

Example: `0 0.5596 0.3474 0.0047 0.0119`

## YOLO Segmentation Format

One line per object. Space-separated: `class_id x1 y1 x2 y2 ... xN yN`

- First value is `class_id` (integer)
- Remaining values are polygon vertex coordinates (x, y pairs), normalized [0, 1]
- Minimum 3 vertices (6 values after class_id)

Example: `0 0.412 0.301 0.435 0.298 0.441 0.325 0.418 0.330`

## Prediction Output Format

Same format as labels. Predictions go in `data/predictions/detect/` or `data/predictions/segment/`. Optionally include confidence as the second value: `class_id confidence cx cy w h`.

## Dataset YAML (crops.yml)

```yaml
path: ./data
train: images    # or train/val split paths
val: images
names:
  0: class_name
  1: another_class
```

The `crops.yml` at project root defines the class mapping and data paths.

## Data Validation Rules

Before training, validate:
- Every image in `data/images/` has a corresponding label in the target label dir
- No empty label files (at least one object per labeled image)
- All class IDs are within the range defined in `crops.yml`
- All coordinates are in [0, 1] — flag any outside this range
- No duplicate label files
- Image format is JPEG or PNG, readable by PIL/OpenCV

## Image Naming

- Stem is the unique identifier (e.g. `IMG_0133`)
- No spaces in filenames
- Extensions: `.jpg`, `.jpeg`, `.png` (lowercase preferred)
- Original camera naming is fine — do not rename unless conflicts exist

## Key Constraints

- Labels are always plain text files, one object per line
- Coordinates are always normalized [0, 1] — never pixel values
- class_id is always an integer, 0-indexed
- Detection and segmentation labels live in separate directories, never mixed
- The agent must validate data format before launching any training run
