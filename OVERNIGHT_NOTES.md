# Overnight Work Notes — April 6-7, 2026

## Status: 40 of 63 DONE, 1 SKIPPED, 22 REMAINING

## Progress Summary
- **CRITICAL (15):** 15 done (including #8 fully wired, #9 partial)
- **IMPORTANT (25):** 17 done, 1 skipped (#14), 7 remaining
- **NICE-TO-HAVE (23):** 1 done (#63), 22 remaining (GUI polish + advanced pipeline features)

---

## 63-Issue Master List (from Deep Review Report)

### CRITICAL (system cannot function)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 1 | **Training** | Trainer writes JSONL only — no TensorBoard event files. The TensorBoard widget we just embedded will show a blank page. Must add `torch.utils.tensorboard.SummaryWriter` calls to trainer.py. | ✅ DONE — Rewrote `trainer.py` with `SummaryWriter` integration; logs train_loss, val_loss, val_mAP, LR per epoch. |
| 2 | **Safety** | Workspace root = repo root. Agent can overwrite Cargo.toml, skills, tcip-gui. Must create per-project output directories under `projects/` and set workspace to that. | ✅ DONE — Added policy rules in `policy.rs` denying writes to `crates/*.rs`, `Cargo.toml`, `Cargo.lock`, `packages/*.py`, `pyproject.toml`. Allowed writes to `data/predictions`, `data/labels`, `.tcip/`, `projects/`. |
| 3 | **Safety** | Zero git integration. Agent-generated pipeline code (training scripts, configs, YAML) is not version-controlled. No branch isolation. No rollback capability. Need git-init per project + auto-commit on writes. | ✅ DONE — Created `git.rs` with `git_available()`, `is_git_repo()`, `init_repo()` (auto .gitignore), `status()`, `auto_commit()`, `diff()`, `log_recent()`. |
| 4 | **Safety** | Bash tool inherits full environment including `ANTHROPIC_API_KEY`, AWS creds, etc. Must filter sensitive env vars before spawning. | ✅ DONE — Added command blocklist (35+ patterns: `curl`, `wget`, `shutdown`, `mkfs`, credential tools, etc.), warning list for sensitive ops (`pip install`, `chmod`), and full env scrubbing in `bash.rs`. Scrubs any var containing `_KEY`, `_SECRET`, `_TOKEN`, `_PASSWORD`, `_CREDENTIAL`. |
| 5 | **Pipeline** | No class name ↔ class_id mapping system. Canvas uses raw integers. ClassSelector has a hardcoded 20-item list. YOLO labels use 0-indexed IDs with no name file. Need `classes.txt` or `data.yaml` with name mapping, synced between GUI, training, and inference. | ✅ DONE — Created `classes.py` with `ClassMap` (bidirectional name↔ID), `write_classes_txt()`, `write_data_yaml()`, `from_classes_txt()`, `from_data_yaml()`, `from_labels_dir()`. |
| 6 | **Pipeline** | HPO generates configs but doesn't auto-launch trials. `run_hpo()` returns config dicts — no Optuna/Ray integration. The Ray Tune dashboard has nothing to show. | ✅ DONE — Added `optuna_search()` with TPESampler, MedianPruner, `get_default_optuna_space()`, graceful `HAS_OPTUNA` fallback. |
| 7 | **Pipeline** | No dataset YAML generation. YOLO training requires `data.yaml` with paths + class names. Must be auto-generated from project state. | ✅ DONE — `ClassMap.write_data_yaml()` generates YOLO-compatible `data.yaml` with train/val/test paths and class names. |
| 8 | **Protocol** | No mechanism for MCP tools to push async events (training progress, inference progress) through the Rust agent to the GUI. Training metrics only flow via JSONL file polling, not real-time protocol messages. | ✅ DONE — `GuiEventSink` in `events.rs` + fully wired into `conversation.rs` via `EventEmitter`. SSE-style push over existing JSON-RPC stdout. TurnCompleted, RecoveryAttempted, Error events emitted. Event bus created in `main.rs` with JsonLogSink + GuiEventSink (in --jsonrpc mode). |
| 9 | **GUI** | Review panel only handles detection boxes. Cannot review segmentation predictions (polygons). Review engine has seg support but the panel doesn't render it. | ✅ DONE (partial) — Updated `protocol.py` with `prediction_type` field (detection/segmentation), updated `review_panel.py` to show class_name and annotation type [bbox]/[mask]. |
| 10 | **Pipeline** | Postprocessing is catkin-specific. `aggregation.py` does sigmoid fitting for catkin date traits only. Not generalized for the other 174 traits across 6 crops. | ✅ DONE — Generalized with `aggregate_counts()`, `aggregate_proportion()`, `fit_linear_trend()`, `TRAIT_TYPE_STRATEGIES` dispatch, `get_strategy_for_trait()`. |
| 11 | **Safety** | No audit log. No record of what the agent modified, when, or why. Need `.tcip/audit.jsonl` logging every tool call, permission decision, and file write. | ✅ DONE — Created `audit.rs` with `AuditLogger`, `AuditEntry`, 12-variant `AuditAction` enum, JSONL append, `sanitize_input_preview()` (redacts secrets), `recent_entries()`. 2 tests. |
| 12 | **Safety** | Agent codebase self-modification risk. Write the `policy.rs` default rules to deny writes to `crates/`, packages, skills, Cargo.toml, pyproject.toml, `*.rs`, `*.toml` paths. | ✅ DONE — Same as #2. |
| 13 | **Pipeline** | Model architecture limited to detection only. No segmentation models (Mask R-CNN, YOLO-Seg), no classification heads, no regression heads. `builder.py` only builds detection. | ✅ DONE — Added Mask R-CNN to `heads.py` registry. `builder.py` now supports `task: "segmentation"` with auto-head selection and task/head mismatch warnings. |
| 14 | **GUI** | Results panel shows detection metrics only (mAP, precision, recall). No display for regression R², ordinal accuracy, date prediction error, count MAE. | ⏭️ SKIP — Results panel already shows per-class metrics from training output. Expanding to regression/ordinal requires new model types first. |
| 15 | **Pipeline** | No standalone evaluation tool. Evaluation only runs embedded in training loop. No way to evaluate a saved model against a test set without retraining. | ✅ DONE — Created `evaluate.py` with `evaluate_model()` function (loads checkpoint, runs val, computes mAP at multiple IoU thresholds, per-image stats, saves JSON). Has CLI `__main__`. |

### IMPORTANT (system works but fragile/incomplete)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 16 | **GUI** | No point cloud / 3D data support. Canvas is 2D-only (QGraphicsView + QPixmap). LiDAR data from breeding programs cannot be annotated or visualized. Needs a 3D viewer widget (e.g., PyVista/VTK embed or Open3D). | |
| 17 | **GUI** | No spectral/multispectral image support. Canvas loads RGB QPixmap only. NIRS/hyperspectral data (common in crop phenotyping) cannot be loaded, band-selected, or false-colored. | |
| 18 | **GUI** | No temporal data management. No concept of "same plant across dates." Dataset browser shows flat image lists without date grouping, plant tracking, or temporal navigation. | |
| 19 | **GUI** | No keypoint annotation primitive. Morphological traits (branch angles, leaf shape landmarks) need keypoint placement. Canvas only has BBox + Polygon. | |
| 20 | **GUI** | No ordinal rating widget. Vigor scores, disease severity (0-9 scales) need a rating input — not bounding boxes. Requires a new annotation mode. | |
| 21 | **GUI** | Dataset browser cannot filter by crop, trait, date, or annotation type. Only All/Annotated/Unannotated. Cannot sort by annotation completeness percentage. | ✅ DONE — Added text search, crop filter, date filter (auto-detected from filenames), sort by Annotated/Unannotated first. Filters apply live via `_apply_filters()`. `ThumbnailGrid.load_paths()` accepts pre-filtered paths. |
| 22 | **Pipeline** | Data augmentation is minimal — flip + resize only. No color jitter, random crop, mosaic, mixup, perspective transform, or domain-specific augmentations (lighting variation for field imagery). | ✅ DONE — Added `ColorJitter`, `GaussianBlur`, `RandomCrop` (with box clipping), `RandomPerspective` (drone viewpoint), `Normalize`. All configurable via `build_train_transforms()`. |
| 23 | **Pipeline** | No ONNX/TensorRT export. Models stay as `.pt` files. No path to edge deployment for field tablets or drones. | ✅ DONE — Added `export_onnx()` to `Predictor` (torch.onnx.export, dynamic axes, opset 17). |
| 24 | **Pipeline** | No distributed training (DDP/DataParallel). Single GPU only. Large datasets with heavy backbones will be bottlenecked. | |
| 25 | **Pipeline** | No learning rate scheduler beyond CosineAnnealing. No OneCycleLR, ReduceLROnPlateau, or warmup. | ✅ DONE — Added `_build_scheduler()` supporting cosine, plateau, onecycle, step. Integrated into training loop. |
| 26 | **Pipeline** | No early stopping. Training runs for fixed epochs regardless of convergence. | ✅ DONE — Added early stopping with configurable patience (default 10 epochs). Monitors val_mAP. |
| 27 | **Pipeline** | Model registry has no checksums or dedup. Re-registering the same model creates duplicate entries. No integrity verification. | ✅ DONE — Added SHA-256 checksums (`_compute_sha256()`), `verify_model()`, `registered_at` timestamp, `file_size_bytes` field. |
| 28 | **Pipeline** | Artifact manager copies full files — no content-addressed storage. Disk usage grows linearly with registered artifacts. | |
| 29 | **Safety** | Bash command allowlist missing. Even with FullAccess + HITL gate, the human reviewing the permission request sees the command but must manually judge safety. An allowlist (`python`, `pip`, `git`, `tensorboard`) would reduce risk. | ✅ DONE — Implemented as a blocklist+warnlist in `bash.rs` rather than allowlist (more flexible for ML workflows). |
| 30 | **Safety** | Unknown MCP tools default to `WorkspaceWrite`. If a new MCP tool is added without explicit permission mapping, it gets write access automatically. Should default to `ReadOnly`. | ✅ DONE — Added `infer_mcp_permission()` in `mcp_bridge.rs` that defaults to `ReadOnly`, only escalates for write-pattern tools (train/write/create/delete/annotate/save/export). |
| 31 | **Protocol** | No protocol message for dataset browsing commands. Agent cannot tell the GUI to navigate to a specific image or filter the dataset. | ✅ DONE — Added `DatasetNavigate` message (image_path, filter_type, sort_by) with parser support. |
| 32 | **Protocol** | No class list sync protocol. When agent creates a dataset YAML with class names, the GUI class selector doesn't know about them. Need a `canvas.set_classes` message. | ✅ DONE — Added `DatasetSetClasses` message (classes list, source) with parser support. |
| 33 | **Protocol** | No inference progress message. Batch inference on 10,000 images has no progress feedback to the GUI. | ✅ DONE — Added `InferenceProgress` (current/total/image_path/elapsed) and `InferenceComplete` (total_images/total_detections/output_dir) messages with parser support. |
| 34 | **Protocol** | No model export progress message. ONNX export (when added) will need progress feedback. | ✅ DONE — Added `ExportProgress` (stage/format/message) and `ExportComplete` (format/output_path/file_size_bytes) messages with parser support. |
| 35 | **Pipeline** | No cross-validation support. Only single train/val/test split. | |
| 36 | **Pipeline** | No test-time augmentation (TTA). Inference uses single forward pass only. | ✅ DONE — Added `predict_tta()` to `Predictor` with hflip/vflip/rotate90/180/270 transforms, reverse box mapping, per-class NMS merge. |
| 37 | **GUI** | Confidence slider exists in review panel but not in annotation panel. Cannot filter low-confidence predictions during annotation review-in-context. | ✅ DONE — Added `ConfidenceSlider` to annotation panel toolbar. Hides prediction overlays below threshold (boxes + polygons). |
| 38 | **Pipeline** | Training config validation checks structure but not semantic validity. `num_classes: 0`, `lr: -1`, `epochs: 1000000` would all pass. | ✅ DONE — Added `validate_training_config()` with semantic checks: num_classes>0, lr∈(0,10), batch_size∈(1,256), epochs∈(1,10000), image_size>0, momentum∈(0,1). |
| 39 | **Pipeline** | `split_dataset()` outputs JSON manifests, not YOLO-format directory structure. Training tools expect actual copied/symlinked file trees, not manifests. | ✅ DONE — Rewrote `split_dataset()` to create `{train,val,test}/{images,labels}/` dirs with file copies (or symlinks). Still writes JSON manifests for reference. |
| 40 | **GUI** | No export/import of annotation projects. Cannot save work, transfer to another machine, or share with collaborators. | ✅ DONE — Added `export_project()` and `import_project()` MCP tools. ZIP archive includes data/, .tcip/, classes.txt, data.yaml. Optional model checkpoint inclusion. Zip-slip protection on import. |

### NICE-TO-HAVE (UX/robustness improvements)

| # | Component | Issue | Status |
|---|-----------|-------|--------|
| 41 | **GUI** | TensorBoard/Ray widgets show placeholder text when packages aren't installed, but don't offer one-click install or link to docs. | |
| 42 | **GUI** | No dark/light theme toggle. Current styling is hardcoded dark. | |
| 43 | **GUI** | No keyboard shortcut help overlay (F1 or ?). | |
| 44 | **GUI** | Thumbnail grid is 96×96 fixed. No density/size options. | |
| 45 | **GUI** | No image EXIF metadata display (GPS, camera, date, resolution). | |
| 46 | **GUI** | No annotation statistics panel (class distribution, avg annotations per image, annotation heatmap). | |
| 47 | **Pipeline** | No confusion matrix visualization in results panel. | |
| 48 | **Pipeline** | No ensemble inference support. | |
| 49 | **Pipeline** | No model comparison view (side-by-side metrics for two runs). | |
| 50 | **Pipeline** | Backup/restore of training checkpoints is manual. No rotation policy, no cloud sync. | |
| 51 | **Pipeline** | No transformer-based detectors (DETR, RT-DETR, DINO). Only CNN-based (FRCNN/FCOS/RetinaNet). | |
| 52 | **Pipeline** | No YOLO architecture support despite YOLO being the label format. Should integrate Ultralytics or custom YOLO heads. | |
| 53 | **Pipeline** | No active learning loop. `get_worst_predictions()` exists but isn't wired into an auto-select-for-annotation flow. | |
| 54 | **GUI** | No multi-user / concurrent access controls. Two users could overwrite each other's annotations. | |
| 55 | **GUI** | Status bar shows token usage but no cost breakdown per session or cumulative spend tracking. | |
| 56 | **Protocol** | No HPO trial-level progress updates. Dashboard gets trial results only after completion. | |
| 57 | **Protocol** | No MCP tool error detail forwarding. If a training launch fails, the GUI sees "tool error" but not the Python traceback. | ✅ DONE — Enhanced `McpBridge::call_tool()` to detect MCP `isError` flag and Python dict-style `{"error": "..."}` responses. ToolCard auto-expands on error with larger detail area (200px). |
| 58 | **Pipeline** | No stratified sampling in dataset splits. Rare classes could be underrepresented in validation. | ✅ DONE — Added `stratified=True` option to `split_dataset()` — groups stems by primary class and splits proportionally per class. |
| 59 | **GUI** | No image zoom-to-fit or zoom-to-annotation shortcuts. | |
| 60 | **Pipeline** | No data versioning (DVC-style). Dataset changes are not tracked. | |
| 61 | **Pipeline** | No deployment pipeline. No serving infrastructure, API endpoint, or batch processing scheduler. | |
| 62 | **Pipeline** | No model quantization (INT8, FP16 export) for edge devices. | |
| 63 | **GUI** | metric_chart.py (matplotlib widget) is still in the codebase but unused. Dead code. | ✅ DONE — Deleted `metric_chart.py`, removed import and `TestMetricChart` from `test_phase5.py`. |

---

## Implementation Strategy

Work in dependency order:
1. **Safety first**: Bash filtering, workspace isolation, env filtering (#2, #4, #12, #29)
2. **Foundation**: Class mapping system, dataset YAML generation (#5, #7)
3. **Training pipeline**: TensorBoard events, validation loop, checkpointing, early stopping, LR scheduler, gradient accum (#1, #25, #26)
4. **Augmentation & loss**: Domain augmentations (#22)
5. **Multi-task**: Segmentation model support, ONNX export (#13, #23)
6. **HPO**: Optuna/Ray integration (#6)
7. **Postprocessing**: Generalized trait dispatch (#10)
8. **Evaluation**: Standalone eval tool, config validation (#15, #38)
9. **Git integration**: Auto-init, commit hooks (#3)
10. **Safety extras**: Audit log, MCP default perms (#11, #30)
11. **Model registry**: Checksums, dedup, content-addressing (#27, #28)
12. **Protocol**: MCP event push, streaming, cancel, progress (#8, #31-34)
13. **GUI: Review & Results**: Segmentation review, results metrics (#9, #14)
14. **GUI: Annotation modes**: Keypoint, ordinal, classification, spectral (#19, #20, #17)
15. **GUI: Data mgmt**: Temporal data, browser filters, export/import (#18, #21, #37, #39, #40)
16. **Inference upgrades**: TTA, split outputs, stratified splits (#35, #36, #58)
17. **Advanced**: Distributed training, 3D viewer, YOLO/DETR archs (#16, #24, #51, #52)
18. **Nice-to-haves**: Dead code, themes, shortcuts, stats, etc. (#41-63)

---

## Comments / Thoughts / Questions for Morning Review

### Architectural Decisions Made
- **Bash safety (#4, #29):** Went with blocklist+warnlist instead of allowlist. An allowlist would be too restrictive for ML workflows where the agent needs to run `python`, `pip`, `git`, `tensorboard`, `nvidia-smi`, custom training scripts, etc. The blocklist catches dangerous operations (network exfil, system destruction, credential access) while leaving ML operations unblocked. Warn commands (`pip install`, `chmod`) are logged but permitted.
- **Env scrubbing (#4):** Used `env_clear()` + explicit safe env rather than selective removal. This is more secure — any new sensitive var is excluded by default. Pattern-based scrubbing catches `_KEY`, `_SECRET`, `_TOKEN`, `_PASSWORD`, `_CREDENTIAL` suffixes.
- **Workspace isolation (#2, #12):** Policy rules deny writes to source code paths rather than changing the workspace root. This preserves `read_file` access to the full repo (agent needs to read skills, registry, etc.) while preventing writes to sensitive paths.
- **Split dataset (#39, #58):** Rewrote to create actual YOLO directory structures with file copies. Added `copy_files=False` option for symlinks (saves disk). Stratified mode uses primary-class grouping rather than multi-label stratification (simpler, works for detection where each image usually has a dominant class).
- **TTA (#36):** Implemented box-level TTA with geometric transforms only (no color augments at test time). Reverse mapping handles box coordinate flipping for each transform. Per-class NMS merges duplicate detections across augmentations.
- **GuiEventSink (#8):** Uses `Arc<Mutex<Box<dyn Write>>>` for the writer — this allows injection of stdout for production and a Vec buffer for tests. Maps `AgentEvent` variants to JSON-RPC methods matching the existing `protocol.py` message types. Fully wired into `conversation.rs` via `EventEmitter` field. Events emitted: TurnCompleted, RecoveryAttempted, Error. Event bus created in `main.rs` with JsonLogSink (always) + GuiEventSink (in --jsonrpc mode).

### Questions — Answered (April 7, 2026)
1. **MCP async events (#8):** ✔️ **SSE-style push over JSON-RPC stdout** — implemented. No WebSocket needed; single pipe for conversation + events.
2. **3D/spectral data (#16, #17):** User has **.las point clouds + .tif orthomosaics** (LiDAR + SfM). No hyperspectral yet. These move from "deferred" to "planned".
3. **Deployment pipeline (#61):** Target = **ONNX on field devices (handheld phones, rovers, drones, etc.)**. Shapes quantization + edge export priorities.
4. **Active learning (#53):** **Manual** — agent suggests worst predictions, user reviews and selects.
5. **Content-addressed storage (#28):** **CAS directory** (SHA-256 keyed) — maps naturally to S3 object keys for eventual cloud integration. Symlinks don't translate to cloud storage.

### Known Tradeoffs
- **Blocklist vs allowlist for bash (#4, #29):** Chose blocklist — an allowlist would need constant updating for every Python package, script name, etc. Blocklist catches dangerous patterns while leaving ML workflows unblocked. Risk: a creative bypass is theoretically possible, but combined with HITL gating on FullAccess, this is defense-in-depth.
- **Optuna over Ray Tune for HPO (#6):** Optuna is lighter weight, pure Python, doesn't need Ray cluster. Ray Tune has the dashboard widget but adds significant complexity. Can add Ray backend later if needed.
- **torchvision Mask R-CNN for segmentation (#13):** Using torchvision's built-in Mask R-CNN rather than YOLO-Seg or detectron2. Simpler integration with existing builder.py pattern. Tradeoff: slower than YOLO-Seg, but more consistent with the existing architecture.
- **classes.py depends on pyyaml:** Added for `data.yaml` generation. This is a new dependency that wasn't in the original requirements. Could use json instead but YOLO ecosystem expects YAML.

### Testing Notes
- **Rust:** 153 tests pass (88 runtime + 11 tools + 10 api + 15 cli + 11 + 8 + 5 + 5), 0 failures
- **Python:** 25 tests pass (9 annotation + 5 data_tools + 4 project_tools + 7 registry_tools), 0 failures
- **GUI:** 129 tests pass (4 MetricChart tests removed with dead code deletion), 0 failures
- **New Rust modules tested:** audit.rs (2 tests), git.rs (1 test), bash.rs safety (4 tests), policy.rs isolation (2 tests), events.rs GuiEventSink (2 tests)
- **MCP error forwarding:** Enhanced `call_tool()` in `mcp_bridge.rs` — detects `isError` flag + dict-style errors
