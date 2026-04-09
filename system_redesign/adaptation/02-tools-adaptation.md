# 02 — Tools Adaptation

## What claw-code has

40 tool specs across 8 categories. Three-layer registry (built-in, plugin, runtime/MCP).
Permission mode per tool. ToolSpec with name, description, JSON Schema, required permission.

## What carries over as-is

- **ToolSpec structure.** Name + description + input schema + permission level. Universal.
- **GlobalToolRegistry pattern.** Central dispatch by name. Keep it.
- **Permission-per-tool.** Each tool declares its minimum permission mode.
- **Three-layer architecture.** Built-in + plugin + MCP. We use the same layers.

## Tools we keep from claw-code (native, in Rust)

| Tool | Why |
|------|-----|
| `bash` | Agent needs to run shell commands (git, python scripts, system tasks) |
| `read_file` | Agent reads configs, code, data files |
| `write_file` | Agent writes configs, generated code |
| `edit_file` | Agent modifies existing files |
| `glob_search` | Find files by pattern |
| `grep_search` | Search file contents |
| `WebFetch` | Look up documentation, papers |
| `canvas_control` | Send display commands to GUI canvas (load image, show predictions, etc.) |

These 8 are the "general-purpose agent + GUI bridge" tools. They stay native in Rust.

## Tools we drop

| Tool | Why |
|------|-----|
| Worker* (8 tools) | Multi-worker orchestration is overkill for single-workstation MVP |
| Team*, Cron* (5) | Same — no multi-agent team coordination needed yet |
| LSP | We're not doing general code editing across languages |
| NotebookEdit | We're replacing Jupyter with our own training pipeline |
| REPL, PowerShell | bash covers this |
| RemoteTrigger | No remote MCP in MVP |
| EnterPlanMode, ExitPlanMode | Can add later if useful |
| Sleep | Not needed |

That cuts 40 → ~12 native tools. The rest come via MCP.

## New tools via Python MCP server

These are the domain-specific tools exposed by the Python MCP server:

### Registry queries (ReadOnly)
- `list_crops` — overview of crops and trait counts
- `get_crop_traits` — all traits for a crop
- `get_trait_info` — full pipeline config for a trait
- `find_traits_by_task` — filter by ML task
- `find_traits_by_sensor` — filter by sensor type

### Data management (WorkspaceWrite)
- `load_dataset` — load annotations, return class distribution
- `validate_data_quality` — check annotation coverage, balance
- `split_dataset` — stratified train/val/test split

### Annotation (WorkspaceWrite)
- `load_annotations` — load YOLO format labels for a folder
- `save_annotations` — write annotations to YOLO format
- `compute_matches` — IoU-based GT vs prediction matching

### Training (FullAccess)
- `validate_config` — check training config before launch
- `launch_training` — start training run, return run_id
- `check_training_status` — poll progress (stage, epoch, metrics)
- `get_training_results` — final metrics, best checkpoint
- `launch_hpo` — start hyperparameter optimization

### Inference (WorkspaceWrite)
- `run_inference` — run model on images, return predictions
- `export_results` — generate per-plant CSV

### Model management (WorkspaceWrite)
- `register_model` — add model to registry
- `list_models` — list models for a trait
- `get_best_model` — best model by metric

### Project state (WorkspaceWrite)
- `get_project_state` — read current project state (crop, trait, pipeline stage)
- `update_project_state` — update project state after completing a pipeline step

## Tool count summary

| Layer | Count | Examples |
|-------|-------|---------|
| Native (Rust) | ~8 | bash, read/write/edit_file, glob, grep, WebFetch, canvas_control |
| MCP (Python) | ~22 | registry queries, training, inference, annotation, project state |
| User interaction | ~3 | SendUserMessage, AskUserQuestion, TodoWrite |
| Meta | ~3 | Agent, ToolSearch, StructuredOutput |
| **Total** | **~36** | |
