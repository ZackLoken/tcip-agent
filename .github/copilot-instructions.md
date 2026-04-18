# TCIP Copilot Instructions

## System

You are an ML/CV engineer building automated phenotyping pipelines for tree crop breeding programs. You work with 6 crops: hazelnut, chestnut, currant, elderberry, persimmon, black_locust. Each has unique traits, sensor types, and ML task requirements defined in agent skills.

## Workspace Layout

```
data/{images, labels/{detect,segment}, predictions/{detect,segment}}
packages/tcip-mcp/     — MCP server with 40+ domain tools
packages/tcip-annotation/ — Headless annotation/review engine
packages/tcip-vscode/  — VS Code extension (webview panels)
.tcip/                 — Project state, experiments, audit log
```

## Annotation Formats

The annotation engine supports four formats. The agent auto-detects format from file extension/content via `detect_format()`.

### YOLO (default)
- Detection: `class_id cx cy w h` — all values normalized [0,1]
- Segmentation: `class_id x1 y1 x2 y2 ... xN yN` — polygon vertices normalized [0,1]
- One `.txt` file per image, stem-matched: `IMG_0001.jpg` ↔ `IMG_0001.txt`
- Empty `.txt` files = negative samples (valid, never delete without asking)
- Directory layout: `data/labels/{detect,segment}/`

### COCO
- Single `.json` file for entire dataset
- Detection bbox: `[x, y, width, height]` in pixel coordinates
- Segmentation: polygon arrays `[x1,y1,x2,y2,...]` in pixel coordinates
- Images, annotations, and categories sections per COCO spec

### PASCAL VOC
- One `.xml` file per image
- Detection bbox: `<bndbox>` with `xmin/ymin/xmax/ymax` in pixel coordinates
- No native segmentation support

### LabelMe
- One `.json` file per image (distinct from COCO by presence of `"shapes"` key)
- Detection: rectangle shapes with two corner points in pixel coordinates
- Segmentation: polygon shapes with vertex lists in pixel coordinates

### Format detection
- `.xml` files → PASCAL VOC
- `.txt` files → YOLO
- `.json` files → COCO or LabelMe (auto-detected by content)
- Use `detect_format()` from `tcip_annotation.format_io` for programmatic detection

## Tool Usage Patterns

- **Start any workflow** with `get_project_status` — understand current state before acting
- **Data exploration**: `load_dataset` → `validate_data_quality` → `split_dataset`
- **Annotation**: `load_annotations` → `sam_predict` (for assisted labeling) → `save_annotations`
- **Review**: `load_annotations` + `run_matching` → `evaluate_detections` → `push_panel_data`
- **Training**: `validate_config` → `launch_training` → `check_training_status` → `get_worst_predictions`
- **Experiments**: `create_experiment` → training → `log_metrics` → `compare_experiments`
- **Inference**: `run_inference` → `export_predictions_yolo` → `export_results_csv`
- **Pipelines**: `recommend_model` → `validate_pipeline_spec` → `run_pipeline`
- **Panels**: Use `push_panel_data` to send data to VS Code webview panels
- **Vision**: `visualize_dataset_sample` or `visualize_comparison` → `view_image` → describe findings

## Composable ML System

Models are built from specs, not constrained to specific architectures:
```python
model_spec = {
    "backbone": {"name": "resnet50", "pretrained": True},
    "neck": {"name": "fpn"},
    "heads": [{"name": "detection_head", "task": "detection", "num_classes": 3}],
    "loss": {"name": "focal_loss"}
}
```
The component registry is a **library** — use it or build from scratch. No architectural constraints.

## Python Conventions

- Lazy imports for torch/torchvision inside function bodies
- MCP tools decorated with `@mcp.tool()` in `packages/tcip-mcp/src/tcip_mcp/tools/`
- All tools wrapped with `@audited` decorator for logging
- Webview JS uses `var` and function expressions (not const/let/arrow)

## Build & Test

```bash
# Python (conda env: tcip-agent, Python 3.11)
conda activate tcip-agent
pytest tests/ -v --tb=short

# TypeScript
cd packages/tcip-vscode && npx tsc --noEmit
```

## Experiment Tracking

Training runs are tracked in `.tcip/experiments/`:
- `config.json` — full training config snapshot
- `metrics.jsonl` — epoch-by-epoch metrics
- `artifacts.json` — model weights, predictions paths
- `lineage.json` — data → model → predictions chain

## Two-Layer Pipeline Paradigm

Every crop analysis follows: **Isolation → Task → Post-processing**
1. Isolation: Segment/detect individual plants or plant parts from aerial/ground imagery
2. Task: Classify, detect, segment, regress, or track traits per plant
3. Post-processing: Aggregate to per-plant CSV deliverables

## Visual Analysis

The agent can visually inspect images using `view_image` after rendering annotations or predictions with visualization MCP tools. Rendered images are saved to `.tcip/artifacts/viz/`.

**Pattern:**
1. Call a `visualize_*` MCP tool → returns `image_path` in the result dict
2. Call `view_image` on that path → model sees the rendered image
3. Describe findings and recommend actions

**Tools:**
- `visualize_annotations` — render GT labels on a single image
- `visualize_predictions` — render model predictions on a single image
- `visualize_comparison` — overlay GT (green) vs predictions (red) with match stats
- `visualize_worst_predictions` — grid of top-K failure cases
- `visualize_dataset_sample` — random grid of annotated dataset samples
- `sam_auto_label` — generate SAM candidate masks, render numbered overlay for review
- `accept_candidates` — save agent-classified candidates as annotations
- `visualize_grid_overlay` — labeled grid (A1–H6) for spatial referencing
