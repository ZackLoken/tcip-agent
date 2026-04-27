# TCIP Agent

Agentic ML/CV system for tree crop breeding programs. Automates pipeline design, training, annotation, review, inference, and per-plant CSV delivery.

## Architecture

Three-process system:

1. **Copilot / Claude agent** (`.github/agents/tcip.agent.md`) — ML/CV engineer persona with progressive domain knowledge via agent skills
2. **Python MCP server** (`packages/tcip-mcp/`) — 40+ domain tools over MCP stdio (data, annotation, training, inference, pipelines, active learning, experiments)
3. **FastAPI + React GUI** (`packages/tcip-web/`) — browser app with Annotate / Review / Training / Tuning / Inference / Results tabs; mirrors the yolo-annotator desktop UX. Backend persists state at `<project_root>/.tcip/state/`; agent and browser both mutate it through the same routes.

Supporting library: `packages/tcip-annotation/` — Headless annotation/review engine (label I/O, matching, AnnotationEngine, ReviewEngine, SAM wrapper).

## Build & Test

```bash
# Python (conda env: tcip-agent, Python 3.11)
conda activate tcip-agent
pytest tests/ -v --tb=short

# Frontend (Vite + React + TS + Tailwind + Konva)
cd packages/tcip-web/frontend
npm install
npm run typecheck
npm run build      # outputs into ../static/

# Backend
python -m tcip_web   # http://127.0.0.1:8765
```

## Key Directories

| Path | Contents |
|------|----------|
| `.github/agents/` | Copilot custom agent definition |
| `.github/skills/` | Agent skills — progressive domain knowledge (loaded when relevant) |
| `.github/prompts/` | Prompt files — slash commands for common tasks |
| `packages/tcip-mcp/` | MCP server — tools across data, annotation, training, inference, etc. |
| `packages/tcip-annotation/` | Label I/O, matching, AnnotationEngine, ReviewEngine, SAM wrapper |
| `packages/tcip-web/` | FastAPI backend + React frontend (the GUI) |
| `tests/` | Python test suite |
| `data/` | Sample images, labels, predictions |

## Crops

Six tree crop species: `black_locust`, `chestnut`, `currant`, `elderberry`, `hazelnut`, `persimmon`.

Traits are defined in per-crop agent skills under `.github/skills/crops/`. Always verify trait names against these files — never assume.

## Conventions

- **Frontend**: React + TypeScript (strict). Source under `packages/tcip-web/frontend/src/`; production build emits to `packages/tcip-web/static/`.
- **Python imports**: Lazy imports for heavy deps (torch, torchvision) inside function bodies
- **MCP tools**: Decorated with `@mcp.tool()`, registered in `packages/tcip-mcp/src/tcip_mcp/tools/`
- **Audit logging**: All tools wrapped with `@audited` decorator, logs to `.tcip/audit.jsonl`
- **Experiments**: Training runs tracked in `.tcip/experiments/` with config, metrics, artifacts, lineage

## Composable ML System

The ML pipeline uses a composable architecture with NO constraints on model architectures:
- `pipelines/components/` — Registered backbones, necks, heads, losses
- `pipelines/registry.py` — `ComponentRegistry` for plugin-style registration (library, not a constraint)
- `pipelines/composer.py` — `compose_model()` builds models from spec dicts
- `pipelines/training/generic_trainer.py` — Generic training loop
- `pipelines/inference/generic_predictor.py` — Generic inference
- `pipelines/data/datasets.py` — Dataset builders (classification, detection, segmentation, point cloud)

## MCP Tools

The system provides 40+ MCP tools organized by domain:
- **Data**: `load_dataset`, `validate_data_quality`, `split_dataset`
- **Annotation**: `load_annotations`, `save_annotations`, `evaluate_detections`, `run_matching`, `sam_predict`, `push_panel_data`
- **Training**: `validate_config`, `launch_training`, `check_training_status`, `run_hpo`, `get_worst_predictions`
- **Inference**: `run_inference`, `export_predictions_yolo`, `export_results_csv`
- **Models**: `list_available_models`, `register_model`, `get_best_model`
- **Pipelines**: `list_components`, `recommend_model`, `validate_pipeline_spec`, `run_pipeline`
- **Projects**: `init_project`, `create_session`, `get_project_status`, `export_project`
- **Experiments**: `create_experiment`, `log_metrics`, `compare_experiments`, `get_experiment_lineage`
- **Active Learning**: `score_unlabeled`, `get_review_queue`
