# Phase 1 — Python MCP Server

## Goal
Standalone Python package exposing all domain tools via MCP protocol (stdio transport).
Testable independently with any MCP client before the Rust agent exists.

## What gets ported

### Pipeline modules (write from scratch, guided by prior design)
- `registry/crops.yml` — trait registry data (YAML, source of truth)
- `pipelines/data/` — dataset loading, formats, splitting, augmentation, tiling
- `pipelines/models/` — backbone/neck/head builders, losses, NMS
- `pipelines/training/` — multi-stage trainer, HPO, schedulers, checkpointing
- `pipelines/evaluation/` — metrics per task type
- `pipelines/inference/` — predictor, tiled inference
- `pipelines/postprocessing/` — aggregation, filtering, CSV export
- `artifact_manager.py` — run directory structure, metrics logging
- `model_registry.py` — model version tracking

**NOT ported**: `schema.py` (16 enums + 9 Pydantic models) and `registry.py` (validation loader).
See [09-schema-registry-redesign.md](../adaptation/09-schema-registry-redesign.md).
The MCP tools load YAML directly with `yaml.safe_load()` and return JSON dicts.
No intermediate Python types for domain values.

### Shared `tcip_annotation` package
The headless annotation/review engines become a shared library (`tcip_annotation`)
imported by both the MCP server and the GUI:
- `state.py` → AppState (annotation/review data model, no GUI)
- `annotation/engine.py` → AnnotationEngine (headless CRUD, undo/redo, spatial index)
- `review/engine.py` → ReviewEngine (matching, review state persistence)
- `label_io.py` → YOLO format parsing/writing
- `matching.py` → IoU-based GT vs prediction matching

### New code
- `server.py` — MCP server entry point, tool registration
- `tools/` — MCP tool handlers wrapping the ported modules

## Package structure (monorepo)

All three packages live in one repository:

```
tcip/                             # Monorepo root
├── registry/
│   └── crops.yml                 # Trait registry (YAML, source of truth)
├── skills/                       # Skill files (markdown, see adaptation/10-skills-architecture.md)
│   ├── pipeline-design.md
│   ├── model-selection.md
│   ├── training-config.md
│   ├── image-processing.md
│   ├── annotation-workflow.md
│   ├── crop-science.md
│   └── evaluation-method.md
├── packages/
│   ├── tcip-annotation/          # Shared annotation engine library
│   │   ├── pyproject.toml
│   │   └── src/tcip_annotation/
│   │       ├── __init__.py
│   │       ├── engine.py         # Headless annotation CRUD, undo/redo
│   │       ├── review_engine.py  # Matching, review state persistence
│   │       ├── label_io.py       # YOLO format parsing/writing
│   │       ├── matching.py       # IoU-based GT vs prediction matching
│   │       ├── state.py          # Annotation/review data model
│   │       └── utils.py          # EXIF, geometry helpers
│   ├── tcip-mcp/                 # MCP server package
│   │   ├── pyproject.toml
│   │   └── src/tcip_mcp/
│   │       ├── __init__.py
│   │       ├── server.py         # MCP server entry, tool registration
│   │       ├── tools/
│   │       │   ├── __init__.py
│   │       │   ├── registry_tools.py   # list_crops, get_trait_info, etc.
│   │       │   ├── data_tools.py       # load_dataset, validate, split
│   │       │   ├── annotation_tools.py # load/save annotations, compute_matches
│   │       │   ├── training_tools.py   # validate_config, launch, check_status
│   │       │   ├── inference_tools.py  # run_inference
│   │       │   ├── model_tools.py      # register, list, get_best
│   │       │   └── project_tools.py    # update_project_state, get_project_state
│   │       ├── pipelines/
│   │       │   ├── data/               # Dataset loading, splitting, augmentation
│   │       │   ├── models/             # Backbone/neck/head builders
│   │       │   ├── training/           # Multi-stage trainer, HPO
│   │       │   ├── evaluation/         # Metrics per task type
│   │       │   ├── inference/          # Predictor, tiled inference
│   │       │   └── postprocessing/     # Aggregation, CSV export
│   │       ├── artifact_manager.py     # Run directory management
│   │       └── model_registry.py       # Model version tracking
│   └── tcip-gui/                 # GUI package (Phase 3+)
│       └── ...
├── tcip-agent/                   # Rust agent (Phase 2+)
│   └── ...
└── tests/
    ├── test_registry_tools.py
    ├── test_data_tools.py
    ├── test_annotation_tools.py
    ├── test_training_tools.py
    └── test_inference_tools.py
```

## MCP tool definitions

### Registry (ReadOnly)
| Tool | Input | Output |
|------|-------|--------|
| `list_crops` | none | crop names + trait counts |
| `get_crop_traits` | crop_name | all traits grouped by status |
| `get_trait_info` | crop, trait | full pipeline config |
| `find_traits_by_task` | crop, ml_task | matching traits |
| `find_traits_by_sensor` | crop, sensor_type | matching traits |
| `get_registry_summary` | none | aggregate stats |

### Data (WorkspaceWrite)
| Tool | Input | Output |
|------|-------|--------|
| `load_dataset` | path, format | summary: image count, class dist |
| `validate_data_quality` | images_dir, labels_dir | issues list |
| `split_dataset` | path, ratios, stratified | train/val/test manifests |

### Annotation (WorkspaceWrite)
| Tool | Input | Output |
|------|-------|--------|
| `load_annotations` | folder_path | annotation state summary |
| `save_annotations` | folder_path, annotations | success/failure |
| `compute_matches` | gt_path, pred_path, iou_thresh | TP/FP/FN counts + details |

### Training (FullAccess)
| Tool | Input | Output |
|------|-------|--------|
| `validate_config` | config_dict | compatibility check result |
| `launch_training` | config_dict | run_id |
| `check_training_status` | run_id | stage, epoch, metrics, ETA |
| `get_training_results` | run_id | final metrics, best checkpoint |
| `launch_hpo` | config, search_space | hpo_run_id |

### Inference (WorkspaceWrite)
| Tool | Input | Output |
|------|-------|--------|
| `run_inference` | model_path, images_dir, config | prediction summary + files written |
| `export_results` | run_id, output_format | CSV path |

### Model Management (WorkspaceWrite)
| Tool | Input | Output |
|------|-------|--------|
| `register_model` | run_id, model_id, tags | success |
| `list_models` | trait_name | model versions |
| `get_best_model` | trait_name, metric | model info + path |

### Project State (WorkspaceWrite)
| Tool | Input | Output |
|------|-------|--------|
| `get_project_state` | project_dir | current stage, models, datasets |
| `update_project_state` | project_dir, updates | success |

## Test criteria (Phase 1 complete when)

1. `python -m tcip_mcp` starts MCP server on stdio
2. MCP client sends `tools/list` → gets all ~23 tools back
3. `list_crops` → returns 6 crops with correct trait counts
4. `get_trait_info("hazelnut", "catkin_05per_date")` → returns phenology trait with detection pipeline config
5. `load_dataset` with real hazelnut annotations → correct class distribution
6. `compute_matches` with hazelnut GT + model predictions → TP/FP/FN counts
7. `validate_config` with a training config → passes validation
8. All ported pipeline module tests pass (data, models, training, evaluation)
9. All ported annotation engine tests pass (engine, matching, label I/O)

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "mcp",           # MCP Python SDK
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy",
    "pillow>=9.0",
    "shapely>=2.0",
    "ray[tune]>=2.0",
    "optuna>=3.0",
    "tensorboard>=2.10",
]
```

## Estimated scope
- Pipeline modules: ~3000 LOC (data, models, training, evaluation, inference, postprocessing)
- Annotation engine: ~800 LOC (shared `tcip_annotation` package)
- New MCP server glue: ~500 LOC
- No schema.py/registry.py — tools query YAML directly
- Tests: pipeline + annotation + MCP-layer tests
