# TCIP architecture and API

This document states what exists at the current commit: what each module owns, the real
dependency graph, the public surface an adopter may rely on, the on-disk formats, and the
cross-layer seam inventory. Its factual sentences are written to be mechanically
checkable: each states one fact a script can verify (a path exists, a symbol is defined
in a file, a count equals N, X imports Y, a route is registered by a file).

Status: provisional. A CI check over this document's factual sentences is planned and not
yet wired; until it runs, treat any sentence that disagrees with the code as a defect in
this document and correct the document.

Marker convention: a sentence followed by an HTML comment beginning `queued:` describes
the surface as it is today, while a recorded maintainer decision queues a change to that
surface; the comment names the decision record. Markers are removed as each change lands
or is rejected. Rendered views hide these comments; `grep -n "queued:"` lists them.

Sections:

1. Module ownership and dependency graph
2. Public surface
3. On-disk formats
4. Seam inventory


## Module ownership and dependency graph

Source: regenerated module inventory at `docs/audit/phase5/archdoc/module-inventory-head.json` / `.md`, produced by `build_module_inventory.py` (a copy of `docs/audit/phase0/module-inventory/build_module_inventory.py` kept beside its output in `docs/audit/phase5/archdoc/` and run there against HEAD 2670cebf). Every count in this section is read from that regenerated inventory, not from the Phase 0 inventory at `docs/audit/phase0/module-inventory/module-inventory.json`.

HEAD 2670cebf has 323 modules across the five scanned roots (93962 total lines):

| Package (root) | Modules | Lines |
|---|---|---|
| tcip-mcp | 94 | 42609 |
| tcip-annotation | 12 | 3879 |
| tcip-web | 34 | 10570 |
| tcip-web-frontend | 145 | 29998 |
| scripts | 38 | 6906 |

`tcip-mcp`, `tcip-annotation`, and `tcip-web` are the three Python packages under `packages/`; `scripts` is `scripts/` at the repo root (not an installed package); `tcip-web-frontend` is the TypeScript/TSX tree under `packages/tcip-web/frontend/src`. These counts are the `counts.python_by_root` and `counts.typescript_total` fields of `module-inventory-head.json`.

differs from phase0 record: the Phase 0 inventory (`docs/audit/phase0/module-inventory/module-inventory.json`) recorded python_total 149 (tcip-mcp 85, tcip-annotation 11, tcip-web 30, scripts 23) and typescript_total 123. The regenerated inventory at the HEAD named above has python_total 178 (tcip-mcp 94, tcip-annotation 12, tcip-web 34, scripts 38) and typescript_total 145; the tables below name every module of both inventories, and `scripts/check_architecture_doc.py` refuses a source file under a covered root that no row names.

## tcip-mcp

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-mcp/src/tcip_mcp/__init__.py | TCIP MCP Server: domain tools for the phenotyping platform. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/__main__.py | Entry point: ``python -m tcip_mcp``. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/agent_identity.py | Which agent harness this MCP server process serves, declared at the handshake, and the session it minted; projected onto every audit line, statement record and HTTP push. | 0 | 7 |
| packages/tcip-mcp/src/tcip_mcp/audit.py | Audit logging decorator for MCP tools, the log each event's scope routes it to, and the refusal a call raises when its own entry cannot be appended. | 3 | 28 |
| packages/tcip-mcp/src/tcip_mcp/class_registry.py | The dataset's class registry, subjects, their attributes, and the deterministic name→id assignment a training run uses (and records, so predictions stay decodable). | 1 | 11 |
| packages/tcip-mcp/src/tcip_mcp/dataset_layout.py | Canonical dataset-layout resolver: the single source of truth for where an image's ground-truth labels and model predictions live on disk. | 4 | 30 |
| packages/tcip-mcp/src/tcip_mcp/experiments.py | Experiment tracking for ML training runs. | 4 | 15 |
| packages/tcip-mcp/src/tcip_mcp/identity.py | The platform's recorded-actor convention, in one place. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/model_registry.py | Model registry, track trained models and their performance. | 2 | 12 |
| packages/tcip-mcp/src/tcip_mcp/operationalization.py | Per-project trait-operationalization records: what a delivered number means, who confirmed it, and the precondition every crossing delivery door checks. | 8 | 9 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/__init__.py | Pipeline sub-package: data, models, training, evaluation, inference, postprocessing. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/__init__.py | Active learning pipeline: scorer and selector modules. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/helpers.py | Shared active-learning helpers used by the AL MCP tools. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/scorer.py | Active learning scorers: rank unlabeled images by informativeness. | 2 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/selector.py | Active learning selector: pick next images to annotate. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/band_stats.py | Display band statistics, the 8-bit stretch every band render goes through, and the RGB composite it stacks into. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py | Block-aware calibration/holdout: validate a detection operating point directly against a mosaic's own reserved calibration/test bands (see ``training_tools._spatial_single_source_split``'s four-way split, ``reserve_calibration_fraction``), for a raster training source too large or too singular to hold whole images out from. | 15 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/__init__.py | Components sub-package: composable ML primitives. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/backbones.py | ``BackboneWrapper``: the interface a backbone must expose to the necks and detectors here. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py | 2D object-detector builders: plain torchvision detector factories. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/heads.py | Task-specific heads: each knows its loss, metric, and output format. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/losses.py | Loss functions for bespoke models: plain importable classes + a name->class map. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/necks.py | Neck modules: adapt backbone features for downstream heads. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/__init__.py | Data pipeline: dataset loading, augmentation, tiling, splitting. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/augmentations.py | Data augmentation transforms for all task types. | 1 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/band_groups.py | Sensor-agnostic band-group correlation: sibling single-band raster files that are really one logical multi-band capture (some multispectral drone sensors write one file per band instead of one multi-band file per image), and the ``.bandgroup`` manifest that records a found group. | 0 | 15 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py | Multi-task datasets with standardized interfaces. | 12 | 10 |  <!-- queued: P5-224 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/samplers.py | Task-aware data samplers: class-imbalance handling plus read-locality ordering. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/splits.py | Group-aware, annotation-stratified train/val splitting. | 5 | 11 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/tiling.py | Sliding-window tiling geometry for small-object detection (SAHI-style). | 0 | 8 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py | Tier-A data/model derivations, read the artifact in hand, compute the value. | 7 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/display_bounds.py | Pixel bounds for what the platform serves to a screen or writes as an agent-facing artifact. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/__init__.py | Review -> retrain feedback: materialize curated datasets (W5) and reconstruct a review-confirmed calibration reference from review verdicts (W1). | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/materialize.py | Materialize a curated detection dataset from human review verdicts. | 8 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/review_calibration.py | Reconstruct a calibration reference from human review verdicts. | 4 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/verdicts.py | Reading one stored review verdict entry: the action vocabulary and the boxes it carries. | 1 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/image_utils.py | Shared image utilities for the composable ML pipeline (channel-aware). | 3 | 28 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/__init__.py | Inference pipeline: model loading and batch prediction. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py | Generic predictor for any bespoke ``model_source`` checkpoint. | 9 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/predictor.py | Model-kind contract + the predictor factory. | 5 | 11 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/__init__.py | Measurement primitives: morphology on a *validated* mask (a first-class toolkit primitive). | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/mask_geometry.py | Mask-geometry: dimensional measurements on a validated binary/instance mask. | 3 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/scale_calibration.py | Deriving and validating a physical per-pixel scale against real physical measurements. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py | ``build_model``, the one indirection between a config/checkpoint and an ``nn.Module``. | 2 | 12 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py | The one model-side contract: the measurement boundary, as a behavioral check, not a mold. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py | Resolve the detection operating point (conf/NMS/max_dets/tile) per dataset, at runtime. | 7 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/overviews.py | External overview pyramids (.ovr sidecars) for large rasters. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/__init__.py | Postprocessing pipeline: temporal aggregation and CSV export. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/aggregation.py | Per-plant aggregation, temporal/spatial aggregation of per-image results. | 4 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py | CSV export for per-plant phenotyping results. | 7 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/orthomosaic_mapping.py | Georeferencing for a whole-mosaic GeoTIFF. | 1 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py | Canonical phenology measurement, the one implementation of a trait's positive-fraction milestones. | 4 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/plant_mapping.py | Plant-ID mapping across image capture dates. | 0 | 9 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/proposal.py | Annotation-proposal engines: a method-neutral seam for auto-labeling. | 2 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/raster_source.py | Raster reading: one open-and-read surface for every image source this platform decodes. | 4 | 18 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py | Named reference grid over a raster's native pixel frame. | 3 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/region_completeness.py | Per-cell content digest for the region-completeness store (:func:`tcip_mcp.dataset_layout.region_completeness_path`): detects an annotation edited or deleted inside an attested cell after attestation. | 5 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py | Runtime parameter resolution, the "derive, don't pin" currency. | 9 | 28 |  <!-- queued: P5-230 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/pipelines/schemas.py | Pydantic v2 config schemas for structural/type validation. | 0 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/__init__.py | Training pipeline: trainer, progressive unfreezing, HPO. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py | The audited training envelope + ``TrainContext``. | 16 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py | Task-aware evaluation metrics + composite selection objective. | 11 | 10 |  <!-- queued: P5-227 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/generic_trainer.py | Task-agnostic training loop for a bespoke ``model_source`` model. | 9 | 6 |  <!-- queued: P5-232 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/hpo.py | HPO, hyperparameter optimization on Ray Tune. | 1 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/optimizer_factory.py | Optimizer factory with differential learning rate support. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py | The subprocess entry point ``launch_training`` spawns to run one bespoke training run's actual body, dataset/loader construction, the audited envelope, ``run_training_envelope()``, in an isolated OS process, so a leak/OOM/hang in one run can't take down the launching process or any other concurrent run's process. | 9 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_manager.py | TensorBoard process management for training and HPO runs. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py | Prediction-bucket immutability: never silently overwrite predictions a human reviewed. | 6 | 8 |
| packages/tcip-mcp/src/tcip_mcp/project_paths.py | Stable resolution of the platform state root, independent of a process's cwd. | 0 | 24 |
| packages/tcip-mcp/src/tcip_mcp/project_record.py | The project record: the one document every project carries, holding its authored site. | 0 | 5 |
| packages/tcip-mcp/src/tcip_mcp/project_status.py | Per-project status pointer: a small, persisted summary of recent activity. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/server.py | MCP server entry point: register all domain tools and run on stdio. | 19 | 19 |
| packages/tcip-mcp/src/tcip_mcp/statements.py | Comparable-value and content-hash primitives shared by every statement kind. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/store_catalogue.py | The whole store catalogue in one import: every module that registers a store, package-only so account_for reaches it with no repo root on sys.path. | 35 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/__init__.py | Tool sub-package: each module registers tools with the MCP server. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py | Annotation tools, load, save, and evaluate name-based annotations via MCP. | 15 | 1 |  <!-- queued: P5-233 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/tools/bundle.py | The shared membership accounting archive_project and import_project both compose from: derives every root a project tree is or holds and classifies each file into bookkeeping, a claimed record/log, a blob, or unaccounted. | 6 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py | Data management tools: load datasets, validate quality, split data. | 12 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/experiment_tools.py | Experiment tracking MCP tools: create, log, compare, and trace experiments. | 4 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/feedback_tools.py | Review -> retrain feedback MCP tools. | 11 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py | Inference MCP tools: run models on images, export results. | 22 | 4 |  <!-- queued: P5-225 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/tools/ingest_tools.py | Image ingestion: turn a raw folder of photos into a structured TCIP project. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py | Meta-loop tools for self-improvement. | 4 | 4 |
| packages/tcip-mcp/src/tcip_mcp/tools/model_tools.py | Model management tools, registry, listing, comparison. | 7 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/operationalization_tools.py | The agent's statement tool for trait operationalizations; it can state, never confirm. | 4 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/trait_spec_authoring_tools.py | The agent's authoring tool for a trait spec that does not yet exist; it can author, never confirm. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/orthomosaic_tools.py | Orthomosaic MCP tools: per-plant delivery from a persisted whole-raster prediction bucket plus a plant-locations CSV. | 12 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/phenology_tools.py | Phenology MCP tools, the agent-facing surface for the per-plant phenology pipeline. | 19 | 4 |  <!-- queued: P5-235 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py | Project management tools. | 10 | 7 |
| packages/tcip-mcp/src/tcip_mcp/tools/scale_tools.py | Physical per-pixel scale calibration: the delivery-gating producer for ``resolve_scale.json``. | 10 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py | Training MCP tools, config validation, launch training, HPO, status. | 24 | 8 |  <!-- queued: P5-223 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/tools/vision_tools.py | Vision tools: render annotations and predictions for visual analysis. | 23 | 2 |  <!-- queued: P5-231 merge-or-split -->
| packages/tcip-mcp/src/tcip_mcp/traits.py | Trait knowledge, the human-defined *semantics* of each measurable trait (Tier C). | 4 | 21 |
| packages/tcip-mcp/src/tcip_mcp/utils/__init__.py | Shared low-level utilities for tcip-mcp. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/web_client.py | HTTP client for MCP tools to push state to the tcip-web backend. | 2 | 9 |
| packages/tcip-mcp/src/tcip_mcp/workspace.py | Workspace resolver: where TCIP projects live on disk. | 1 | 13 |

## tcip-annotation

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-annotation/src/tcip_annotation/__init__.py | Headless annotation library: canonical name-based per-image JSON labels + a single-file COCO. | 8 | 4 |
| packages/tcip-annotation/src/tcip_annotation/annotation_engine.py | AnnotationEngine: Annotation CRUD, spatial index, undo/redo. | 2 | 1 |
| packages/tcip-annotation/src/tcip_annotation/format_io.py | Annotation I/O for the two on-disk formats: the canonical per-image JSON and a single-file COCO. | 2 | 5 |
| packages/tcip-annotation/src/tcip_annotation/json_io.py | Per-image JSON: the canonical on-disk label format (ground truth + predictions). | 1 | 32 |
| packages/tcip-annotation/src/tcip_annotation/mask_contours.py | Mask -> polygon rings: the one contour extractor behind every mask-derived shape. | 0 | 3 |
| packages/tcip-annotation/src/tcip_annotation/matching.py | Geometry helpers and GT-vs-prediction matching engine. | 1 | 3 |
| packages/tcip-annotation/src/tcip_annotation/review_engine.py | ReviewEngine: review logic, detection walk-through, accept/reject. | 3 | 7 |
| packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py | SAM2 wrapper for interactive segmentation. | 2 | 6 |
| packages/tcip-annotation/src/tcip_annotation/state.py | Annotation and review data model. | 0 | 21 |
| packages/tcip-annotation/src/tcip_annotation/utils.py | Shared utilities: image orientation, geometry helpers. | 0 | 4 |
| packages/tcip-annotation/src/tcip_annotation/verdicts.py | The review verdict's action vocabulary, declared once. | 0 | 3 |
| packages/tcip-annotation/src/tcip_annotation/viz.py | Visualization rendering: draws annotations and predictions on images. | 2 | 1 |

## tcip-store

Counts in this table are import edges inside `packages/tcip-store/src`, counted the same way as every other table here. This package is not one of the five roots the regenerated module inventory scanned, so its rows are not cross-checked against that JSON.

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-store/src/tcip_store/__init__.py | The storage seam's public surface: keys, errors, store declarations, and the module-level operations. | 5 | 0 |
| packages/tcip-store/src/tcip_store/adoption.py | Moving a root's existing record and log files into a database, exclusively and atomically, or refusing before it writes, including the stores a database beside them has never held. | 5 | 0 |
| packages/tcip-store/src/tcip_store/binding.py | Which backend a process binds at its entry point: the database unless `TCIP_STORE_BACKEND` names the file backend, and a refusal for any other name. | 3 | 0 |
| packages/tcip-store/src/tcip_store/errors.py | Every typed refusal the seam raises, absence and corruption included. | 1 | 6 |
| packages/tcip-store/src/tcip_store/export.py | Writing a root's database-held records and logs back out as the file layout, and the per-store counters that say when those files are behind. | 4 | 0 |
| packages/tcip-store/src/tcip_store/file_backend.py | The filesystem backend: identity to path, atomic replace, file locks, append-only logs, blobs, and the conform rail's refusal of record writes, and of a colliding blob write, to a root a database holds. | 3 | 6 |
| packages/tcip-store/src/tcip_store/layout_claims.py | Which store could own which path under a root, as data: the layout vocabulary, one claim per record and log store, the template-directed walk the conform rail and the adoption planner share, and the anchored match a blob write is checked against. | 3 | 3 |
| packages/tcip-store/src/tcip_store/model.py | Identity and value types the seam speaks on every backend: Key, Version, Versioned, LogPage, Capabilities. | 0 | 5 |
| packages/tcip-store/src/tcip_store/registry.py | The store catalogue: kind, key shape, the canonical JSON codec each kind encodes through and the exemption a store must state to carry another, concurrency policy, durability, enumeration, and the layout claim a store outside the platform table must declare, plus whether the store is frozen and its schema_version ceiling. | 4 | 6 |
| packages/tcip-store/src/tcip_store/schema_version.py | The version-field accept rule: absence reads as version 1, 1 accepts, and a present version above a frozen store's ceiling or one that is not a plain integer refuses, naming the document, the ceiling and the store. | 2 | 3 |
| packages/tcip-store/src/tcip_store/sqlite_backend.py | The database backend: one verified WAL database per root for records and logs, checked against each kind of root it serves on that kind's first use, with blobs forwarded to the file backend. | 5 | 3 |
| packages/tcip-store/src/tcip_store/store.py | The bound-backend surface and the rules that must mean the same thing on every backend. | 3 | 2 |
| packages/tcip-store/src/tcip_store/values.py | What a value must be before a store will carry it, and how a producer says it is not. | 0 | 1 |

## tcip-web

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/src/tcip_web/__init__.py | TCIP Web: FastAPI server for the ML pipeline. | 0 | 0 |
| packages/tcip-web/src/tcip_web/__main__.py | Entry point: ``python -m tcip_web``. | 2 | 0 |
| packages/tcip-web/src/tcip_web/agent_bash_guard.py | PreToolUse Bash guard for the fenced in-app agent terminal. | 1 | 0 |
| packages/tcip-web/src/tcip_web/agent_fence_rules.py | What the in-app agent fence protects, declared once for both shell guards. | 0 | 2 |
| packages/tcip-web/src/tcip_web/agent_learning_capture.py | SessionEnd capture hook: the soft backstop for the self-learning loop. | 1 | 1 |
| packages/tcip-web/src/tcip_web/agent_powershell_guard.py | PreToolUse PowerShell guard for the fenced in-app agent terminal. | 1 | 0 |
| packages/tcip-web/src/tcip_web/agent_session_start.py | SessionStart ritual hook: inject the session-start ritual directive naming the active project. | 2 | 0 |
| packages/tcip-web/src/tcip_web/app.py | FastAPI application: REST API for MCP tools + WebSocket for GUI state sync. | 10 | 3 |
| packages/tcip-web/src/tcip_web/identity.py | Current-user identity for provenance stamping (created_by / accepted_by). | 0 | 5 |  <!-- queued: P5-329 unwired -->
| packages/tcip-web/src/tcip_web/jobstore.py | Persistence + memory-cap helpers for the web's async job registries. | 1 | 7 |
| packages/tcip-web/src/tcip_web/label_annotations_cache.py | The mtime-and-size-keyed label-document parse memo shared by the classes, dataset and review routes. | 1 | 3 |
| packages/tcip-web/src/tcip_web/paths.py | Path resolution helpers with traversal protection. | 3 | 13 |
| packages/tcip-web/src/tcip_web/routes/__init__.py | Route modules for the tcip-web FastAPI backend. | 16 | 1 |
| packages/tcip-web/src/tcip_web/routes/_body_common.py | The empty JSON body a route with only path parameters declares, forcing a browser onto a preflighted request instead of a simple one. | 0 | 3 |
| packages/tcip-web/src/tcip_web/routes/_coverage_models.py | The view-coverage record's viewing context, declared once so `routes/coverage.py` and `routes/images.py` agree on its shape, and so `scripts/generate_frontend_types.py` can render it for the browser instead of the browser hand-transcribing it. | 1 | 4 |
| packages/tcip-web/src/tcip_web/routes/_metrics_common.py | The shape both metric routes serve, from whichever log the caller resolved. | 0 | 2 |
| packages/tcip-web/src/tcip_web/routes/annotate.py | Annotation label CRUD routes for the Annotate tab. | 9 | 1 |
| packages/tcip-web/src/tcip_web/routes/canvas.py | Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back. | 2 | 2 |
| packages/tcip-web/src/tcip_web/routes/classes.py | Class registry routes. | 6 | 2 |
| packages/tcip-web/src/tcip_web/routes/coverage.py | View-coverage routes: the reference grid over a raster and the per-image record of two per-cell facts: which cells were served to the browser at native resolution (a delivery fact) and which cells were swept in the viewport at or above the breeder's own working scale (a sweep fact). | 11 | 2 |
| packages/tcip-web/src/tcip_web/routes/dataset.py | Dataset discovery + selection routes. | 5 | 2 |
| packages/tcip-web/src/tcip_web/routes/fs.py | Local-filesystem directory browsing for the frontend's folder picker. | 1 | 1 |
| packages/tcip-web/src/tcip_web/routes/images.py | Image serving: the one path pixels reach the browser through. | 11 | 3 |  <!-- queued: P5-234 merge-or-split -->
| packages/tcip-web/src/tcip_web/routes/inference.py | Inference routes: async tiled runs + live progress WebSocket. | 15 | 2 |
| packages/tcip-web/src/tcip_web/routes/meta.py | Meta-loop routes: surface Claude's friction reports and retrospectives. | 2 | 1 |
| packages/tcip-web/src/tcip_web/routes/projects.py | Workspace project discovery + the active-project marker. | 4 | 1 |
| packages/tcip-web/src/tcip_web/routes/results.py | Results routes: plant-mapping, per-plant phenology curves, CSV export, the operationalization record surface, the trait-spec statement surface, and the read-only delivery-event list. | 13 | 1 |
| packages/tcip-web/src/tcip_web/routes/review.py | Review routes: compute matches, walk detections, record actions, save GT. | 17 | 2 |  <!-- queued: P5-228 merge-or-split -->
| packages/tcip-web/src/tcip_web/routes/sessions.py | Session-tracking routes: annotation_stats.json equivalent. | 2 | 3 |
| packages/tcip-web/src/tcip_web/routes/terminal.py | Agent terminal routes: the HTTP/WS surface over :mod:`tcip_web.terminal`. | 3 | 5 |
| packages/tcip-web/src/tcip_web/routes/training.py | Training routes: validate config, launch, list runs, live metrics stream. | 9 | 2 |
| packages/tcip-web/src/tcip_web/routes/tuning.py | HPO / Tuning routes: launch + list + per-trial visibility. | 8 | 2 |
| packages/tcip-web/src/tcip_web/state.py | In-memory GUI state + debounced persistence to ``.tcip/state/gui.json``. | 1 | 6 |
| packages/tcip-web/src/tcip_web/trust_boundary.py | The network trust boundary: which connections the backend serves and which names it answers to. | 0 | 5 |
| packages/tcip-web/src/tcip_web/terminal.py | Embedded agent terminal: run the real Claude Code CLI in a PTY. | 3 | 3 |

## tcip-web-frontend

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/frontend/src/App.tsx | (none found) | 28 | 1 |
| packages/tcip-web/frontend/src/api/classes.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/classes.ts | Dataset class-registry + per-image-status API helpers. | 2 | 14 |
| packages/tcip-web/frontend/src/api/client.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/client.ts | Typed REST client for the tcip-web backend. | 6 | 41 |
| packages/tcip-web/frontend/src/api/devProxy.generated.ts | Dev-server proxy prefixes, generated by scripts/generate_frontend_routes.py from the routes the FastAPI app registers. | 0 | 0 |
| packages/tcip-web/frontend/src/api/http.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/http.ts | Shared fetch helpers. | 0 | 13 |
| packages/tcip-web/frontend/src/api/inference.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/inference.ts | Inference + Results API helpers for the Inference and Results tabs. | 4 | 6 |
| packages/tcip-web/frontend/src/api/meta.ts | Meta-loop API helpers: Claude's friction reports and retrospectives. | 2 | 1 |
| packages/tcip-web/frontend/src/api/routes.ts | Every backend path the browser calls, generated from the routes the FastAPI app registers. | 0 | 10 |
| packages/tcip-web/frontend/src/api/sessions.ts | Session-tracking API helpers (annotation_stats.json on disk). | 2 | 4 |
| packages/tcip-web/frontend/src/api/streams.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/terminal.ts | REST client for the embedded agent terminal (the real Claude Code CLI in a PTY). | 2 | 2 |
| packages/tcip-web/frontend/src/api/training.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/training.ts | Training-tab specific REST + WebSocket helpers. | 4 | 6 |
| packages/tcip-web/frontend/src/api/tuning.ts | Tuning (HPO) API helpers for the Tuning tab. | 3 | 1 |
| packages/tcip-web/frontend/src/api/types.generated.ts | Types generated by scripts/generate_frontend_types.py from the pydantic models that declare them (routes/_coverage_models.py, routes/coverage.py, routes/review.py, tcip_web.state.GuiVocabulary, tcip_mcp.web_client, tcip_web.jobstore). | 0 | 18 |
| packages/tcip-web/frontend/src/api/ws.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/ws.ts | WebSocket client that subscribes to GuiState snapshots + panel events. | 5 | 2 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.test.tsx | (none found) | 7 | 0 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.tsx | Annotate-tab context toolbar. | 10 | 2 |
| packages/tcip-web/frontend/src/components/BandPicker.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/BandPicker.tsx | (none found) | 2 | 3 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.tsx | Shared Konva Stage wrapper with pan + zoom state managed in the store. | 6 | 4 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageMinimap.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageMinimap.tsx | Coverage minimap for a multi-cell raster: a small thumbnail of the whole image with per-cell swept shading, region-completeness markers, in-view cell borders, and the current viewport rectangle; clicking or dragging jumps the canvas to the cell under the pointer, double-clicking toggles that cell's completeness for the active subject. | 3 | 2 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.ts | Discrete zoom levels (5% .. | 0 | 3 |
| packages/tcip-web/frontend/src/components/CollapsibleSection.tsx | The app's collapsible-section primitive: one chevron glyph and one trigger+content unit. | 1 | 6 |
| packages/tcip-web/frontend/src/components/ColorPickerModal.tsx | Dark color picker: SI palette + basic palette + hex input, resolving to a hex string. | 0 | 1 |
| packages/tcip-web/frontend/src/components/DeliveryEventsPanel.tsx | What has shipped from this project: one row per completed delivery, read-only. | 1 | 1 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.tsx | A titled chrome bar over an iframe, for the tools the platform runs beside the app (TensorBoard, Ray's dashboard). | 0 | 3 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.tsx | (none found) | 0 | 2 |
| packages/tcip-web/frontend/src/components/HelpOverlay.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/HelpOverlay.tsx | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/components/ProjectBreadcrumb.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/ProjectBreadcrumb.tsx | Status-bar project breadcrumb: three fast-tracks in the lower-right corner: project name → a dropdown of recent projects (jump straight in), date → a dropdown of this project's dates (switch without the workspace), Switch Project → the full workspace (all projects). | 5 | 2 |
| packages/tcip-web/frontend/src/components/ProjectPicker.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/ProjectPicker.tsx | The front door. | 4 | 2 |
| packages/tcip-web/frontend/src/components/SeasonRail.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/SeasonRail.tsx | Season rail: the app's signature. | 0 | 2 |
| packages/tcip-web/frontend/src/components/StatementPanel.tsx | The generalized confirmation surface for a statement record: the agent states, the breeder confirms or withdraws, and a moved/superseded record re-renders what is on file rather than what was last shown. | 4 | 1 |
| packages/tcip-web/frontend/src/components/StatusBar.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/StatusBar.tsx | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/components/TabBanner.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/TabBanner.tsx | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/components/TerminalRail.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/TerminalRail.tsx | The agent rail: the real Claude Code CLI, embedded. | 4 | 2 |
| packages/tcip-web/frontend/src/components/Toasts.tsx | (none found) | 1 | 1 |
| packages/tcip-web/frontend/src/components/TopBar.tsx | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.ts | Mirror the active tab into the backend GUI state so view_gui_state reports the tab the human actually sees. | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.ts | The breeder's band selection for the image `bandsInfo` describes, held once per band-set signature (the image's band names, in order) rather than per component, so a composite chosen in one tab is the one the other renders over the same band set, and a detour through a differently-banded image (or a plain colour photo) neither applies the old selection nor destroys it. | 2 | 3 |
| packages/tcip-web/frontend/src/hooks/useCoverageGrid.ts | The coverage lattice for the open raster, fetched once the base serve shows the raster is larger than one display-bounded serve (Served-Size below the native dims). | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useCoverageTracking.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/hooks/useCoverageTracking.ts | Wires the CoverageTracker into the Annotate tab: resets on the (image, subject, date, dataset, grid) identity, hydrates from the stored record, feeds it viewport passes and the viewing context, and exposes the swept set for the minimap plus the Complete warning facts. | 6 | 2 |
| packages/tcip-web/frontend/src/hooks/useDisclosure.ts | Open/closed state for a collapsible region, optionally remembered across sessions. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.ts | A staged agent request that follows the dataset selection until the breeder edits it. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useImageBands.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageBands.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/hooks/useImageNav.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageNav.ts | Single source of truth for image navigation. | 2 | 5 |
| packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.ts | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.ts | (none found) | 0 | 3 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/hooks/usePrefetchAdjacentImages.ts | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/hooks/useRegionCompleteness.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/hooks/useRegionCompleteness.ts | Wires the region-completeness store into the Annotate tab: fetches every subject's attestation record for the open raster, exposes the active subject's own complete cells separate from every other subject's, and posts a double-click toggle. | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useRegionServes.ts | The cell-aligned region serves the current viewport needs when the user zooms past the base bitmap's resolution on a large raster. | 6 | 2 |
| packages/tcip-web/frontend/src/index.css.test.ts | Compiles index.css through PostCSS + Tailwind (same pipeline as the real build) and asserts the keyboard focus-visible ring rules exist on the shared component classes. | 1 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.ts | Drive the Annotate tab to a specific (subject, date, image, mode) in response to the agent's `annotate_focus` event. | 3 | 2 |
| packages/tcip-web/frontend/src/lib/bandSelection.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/bandSelection.ts | (none found) | 2 | 7 |
| packages/tcip-web/frontend/src/lib/canvasSync.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/canvasSync.ts | Live canvas-state sync: lets the agent see exactly what the canvas shows. | 3 | 4 |
| packages/tcip-web/frontend/src/lib/coverage.test.ts | (none found) | 0 | 0 |
| packages/tcip-web/frontend/src/lib/coverage.ts | Pure helpers over the coverage lattice a raster's grid route serves. | 2 | 9 |
| packages/tcip-web/frontend/src/lib/coverageTracker.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/coverageTracker.ts | Session accumulator for the per-image view-coverage record. | 2 | 2 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.ts | Stop the browser's own ctrl+wheel page zoom over the app, so the canvas' zoom is the only zoom. | 0 | 2 |
| packages/tcip-web/frontend/src/lib/datasetUiState.ts | Per-(project, date, subject/model) UI state, so switching dates/projects and returning lands you back where you were. | 3 | 3 |
| packages/tcip-web/frontend/src/lib/editGeometry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/editGeometry.ts | Pure geometry for in-place box/polygon editing, shared by the Annotate and Review tabs' editors. | 1 | 2 |
| packages/tcip-web/frontend/src/lib/imageLoader.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/imageLoader.ts | Shared image loader for /api/images serves. | 1 | 7 |
| packages/tcip-web/frontend/src/lib/imageStatus.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/imageStatus.ts | (none found) | 1 | 4 |
| packages/tcip-web/frontend/src/lib/labelProblemToast.ts | Toast a dataset selection's label_problem, shared by every path that installs a new selection through /dataset/select. | 1 | 3 |
| packages/tcip-web/frontend/src/lib/labelSerde.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/labelSerde.ts | The single mapping between the unified name-based label file (one Annotation list per image) and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). | 1 | 3 |
| packages/tcip-web/frontend/src/lib/openProject.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/openProject.ts | Opening a workspace project = pointing the GUI at it (project root = dataset root) via /dataset/select. | 4 | 4 |
| packages/tcip-web/frontend/src/lib/paths.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/paths.ts | Normalize a user-pasted filesystem path. | 1 | 6 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts | (none found) | 0 | 0 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.ts | Pure hit-testing helpers for the annotate canvas' geometry (polygons and points). | 0 | 3 |
| packages/tcip-web/frontend/src/lib/recentProjects.ts | The last few projects the user opened, for the status-bar "project name" fast-track. | 0 | 2 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.ts | One reconnecting-WebSocket shape, shared by every socket this app opens. | 0 | 5 |
| packages/tcip-web/frontend/src/lib/reviewColors.ts | (none found) | 0 | 4 |
| packages/tcip-web/frontend/src/lib/reviewFocus.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/reviewFocus.ts | Drive the Review tab to a model's predictions on a specific frame/detection in response to the agent's `review_focus` event. | 2 | 3 |
| packages/tcip-web/frontend/src/lib/reviewGeometry.ts | The single source of a review detection's geometry. | 1 | 0 |
| packages/tcip-web/frontend/src/lib/runStatus.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/runStatus.ts | Statuses a training run or a sweep never leaves, so a poll keyed on one can stop. | 1 | 3 |
| packages/tcip-web/frontend/src/lib/statementFields.ts | Breeder-facing labels for every field a statement record carries. | 0 | 2 |
| packages/tcip-web/frontend/src/lib/viewGeometry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/viewGeometry.ts | Shared view math for the canvas: fit the view to a pixel rect and clamp pan offsets. | 2 | 7 |
| packages/tcip-web/frontend/src/main.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/guiStateShape.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/store/index.ts | (none found) | 3 | 54 |  <!-- queued: P5-238 merge-or-split -->
| packages/tcip-web/frontend/src/store/mergeSnapshot.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/store/store.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/store/tabRestore.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/store/terminalOpenPolicy.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/types.ts | Types mirroring the Python backend's GuiState and the name-based label schema. | 1 | 29 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx | (none found) | 26 | 2 |  <!-- queued: P5-236 merge-or-split -->
| packages/tcip-web/frontend/src/tabs/InferenceTab.test.tsx | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/tabs/InferenceTab.tsx | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/tabs/MetaTab.tsx | (none found) | 4 | 1 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.test.tsx | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.tsx | (none found) | 7 | 2 |
| packages/tcip-web/frontend/src/tabs/ReviewTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/ReviewTab.tsx | (none found) | 24 | 2 |  <!-- queued: P5-237 merge-or-split -->
| packages/tcip-web/frontend/src/tabs/RunMonitorLayout.tsx | The shell the Training and Tuning tabs share: a fixed-width scrolling sidebar of runs beside a detail region. | 0 | 2 |
| packages/tcip-web/frontend/src/tabs/TrainingTab.tsx | (none found) | 9 | 1 |
| packages/tcip-web/frontend/src/tabs/TuningTab.tsx | (none found) | 9 | 1 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.ts | Plain-language requests the run tabs stage for the agent, editable before they're sent. | 0 | 3 |
| packages/tcip-web/frontend/src/tabs/chartTheme.ts | Recharts takes literal colour strings (not Tailwind classes), so the field-station tokens are mirrored here as hex. | 0 | 2 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.ts | Metric-stream helpers for the Training tab (kept out of the .tsx so they're unit-testable). | 1 | 2 |
| packages/tcip-web/frontend/src/test/setup.ts | Extends Vitest's `expect` with jest-dom matchers (toBeInTheDocument, etc.) and registers automatic cleanup after each test. | 0 | 0 |

## scripts

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| scripts/_paths.py | Shared path resolution for the one-off analysis scripts: no machine-specific hardcoding. | 0 | 5 |
| scripts/_store_bootstrap.py | Re-exports the store catalogue from tcip_mcp.store_catalogue, and which roots a project's records live in. | 3 | 2 |
| scripts/adopt_store.py | Move a root's existing record and log files into a store database. | 1 | 0 |
| scripts/calibrate_operating_point.py | Calibrate + held-out validate a detection operating point over a labeled split. | 7 | 0 |
| scripts/check_architecture_citations.py | Verify ARCHITECTURE.md's file:line citations against the code they quote, for CI. | 0 | 0 |
| scripts/check_architecture_doc.py | Verify ARCHITECTURE.md's module-ownership tables against the tree, for CI. | 0 | 0 |
| scripts/check_dataset_identity.py | Check a dataset's on-disk content against its recorded identity: detect changed / moved data. | 3 | 0 |
| scripts/compute_disagreements.py | Summarize GT-vs-prediction disagreements per image at several conf thresholds. | 1 | 0 |
| scripts/conform_cal_holdout_locks.py | Conform every pre-existing `cal_holdout_split_lock` record under a root to carry `split_manifest_dir`. | 3 | 0 |
| scripts/conform_dataset_registry_paths.py | Conform a project's dataset registry onto the relative-path row: rewrite each entry's stored path through the same identity-based rule `register_dataset` now uses. | 3 | 0 |
| scripts/conform_metrics_marker.py | Stamp the ``metrics_logged`` marker onto every experiment a root's status record predates. | 1 | 0 |
| scripts/conform_project_site.py | Write or correct one project's authored site: the record ``init_project``/``ingest_images`` themselves cannot reach for a project whose name does not fit the workspace scheme, and the one deliberate overwrite for a site typed wrong once or a record damaged by hand. | 1 | 0 |
| scripts/conform_registry_experiment_id.py | Conform a project's registry entries to carry ``experiment_id``, for an entry registered before the producer-binding field existed. | 2 | 0 |
| scripts/conform_view_coverage_viewing.py | Conform a dataset's stored `view_coverage` records to the current `CoverageViewing` shape. | 2 | 0 |
| scripts/cross_family_ask.py | Pose one identical question to several agent harnesses and record comparable answers. | 0 | 0 |
| scripts/distill_learnings.py | Distill worksheet: gather one project's learning record in one place. | 2 | 0 |
| scripts/doctor.py | Data-state doctor: scan a live project for state inconsistencies code audits can't see. | 8 | 0 |
| scripts/drop_annotation_stats_image_status.py | Conform a project's ``annotation_stats`` record to drop its dead ``image_status`` key. | 1 | 0 |
| scripts/drop_trait_spec_provenance.py | Conform a project's trait-spec records to drop the retired free-text ``provenance`` field, and remove the stale copies an earlier YAML-to-record conform step left behind. | 1 | 0 |
| scripts/export_store.py | Write a root's database-held records and logs back out as files. | 1 | 0 |
| scripts/foreground_fn_candidates.py | Compute foreground-only high-confidence FN candidates per image. | 1 | 0 |
| scripts/gate_baseline.py | Parse ci.yml's jobs and run the steps it declares, so a local pass means CI would pass too. | 0 | 0 |
| scripts/generate_frontend_routes.py | Generate the browser's route-path module from the backend's registered routes. | 1 | 0 |
| scripts/generate_frontend_types.py | Generate the browser's coverage-record types from the pydantic models that declare them. | 9 | 0 |
| scripts/generate_frozen_manifest.py | Render the store registry's freeze classifications into the shipped frozen-formats.json; --check holds CI to the committed edition. | 2 | 0 |
| scripts/inspect_baseline_weights.py | Print framework / model metadata from the baseline weights.pt. | 1 | 0 |
| scripts/inspect_gps_exif.py | Print GPS EXIF for a sample of images per acquisition date. | 1 | 0 |
| scripts/list_tools.py | Print the live MCP tool registry (count + names). | 1 | 0 |
| scripts/plant_aware_group_splits.py | Plant-aware group-key derivation for ``make_splits``, over per-stem georeferenced rasters. | 3 | 0 |
| scripts/prove_test_fails_before.py | Prove a test actually fails against the code it was written to catch. | 0 | 0 |
| scripts/render_candidates_tile.py | Render a tile showing GT (green) and only the FN candidates (numbered red). | 1 | 0 |
| scripts/restamp_dataset_fingerprint.py | Restamp a bare legacy dataset fingerprint onto the formula-version-prefixed form, through register_dataset's own path. | 4 | 0 |
| scripts/scan_dataset.py | Scan a folder for images, labels, and predictions, through the demoted `scan_dataset` function. | 1 | 0 |
| scripts/shp_to_plant_csv.py | Convert a plant-locations shapefile into ``read_plant_csvs``' CSV schema. | 1 | 0 |
| scripts/smoke_fence_e2e.py | Live smoke: does the real fenced `claude` refuse to edit platform internals? | 3 | 0 |
| scripts/smoke_phenology_e2e.py | Live e2e smoke: the agent's phenology pipeline on real geolocated images. | 4 | 0 |
| scripts/smoke_terminal_e2e.py | One-shot smoke: the embedded agent terminal against the real `claude` CLI. | 2 | 0 |
| scripts/verify_citations.py | Check that literature citations point at real code, real papers, and real sentences. | 0 | 0 |
| scripts/verify_claims.py | List every claim-shaped sentence this change *adds* to comments and docstrings. | 0 | 0 |
| scripts/verify_doc_examples.py | Verify that code examples in skills and source docstrings actually work. | 0 | 0 |
| scripts/verify_skill_tools.py | Guardrail: hold every tool name in agent-facing prose to the registry. | 1 | 0 |
| scripts/verify_skill_traits.py | Guardrail: flag every trait-like token in a crop/domain SKILL.md that is not in crops.yml. | 1 | 0 |
| scripts/watch_agent_chat.py | Read the in-app TCIP agent chat from the orchestrating Claude Code session. | 0 | 0 |

## Package-level dependency rules holding at HEAD 2670cebf

The following sentences are checked against every in-repo Python import edge in `module-inventory-head.json` (an edge is counted only when both the importing file and the imported file resolve to a file inside this repo; stdlib and third-party imports are excluded by `build_module_inventory.py`, see docstring at `docs/audit/phase0/module-inventory/build_module_inventory.py:9-20`).

- No module under `packages/tcip-mcp` imports from `tcip-web`.
- No module under `packages/tcip-mcp` imports from `scripts`.
- No module under `packages/tcip-annotation` imports from `tcip-mcp`.
- No module under `packages/tcip-annotation` imports from `tcip-web`.
- No module under `packages/tcip-annotation` imports from `scripts`.
- No module under `packages/tcip-web` imports from `scripts`.

Non-zero cross-package edge counts at HEAD:

- `scripts` -> `tcip-annotation`: 3 import edges.
- `scripts` -> `tcip-mcp`: 21 import edges.
- `scripts` -> `tcip-web`: 5 import edges.
- `tcip-mcp` -> `tcip-annotation`: 43 import edges.
- `tcip-web` -> `tcip-annotation`: 7 import edges.
- `tcip-web` -> `tcip-mcp`: 71 import edges.

`packages/tcip-web/frontend/src` (`tcip-web-frontend`) has zero in-repo import edges to any Python module in any of the four Python roots: `build_module_inventory.py` resolves a TypeScript specifier only against a relative path or the `@/` alias into `packages/tcip-web/frontend/src` itself (`docs/audit/phase0/module-inventory/build_module_inventory.py:301-321`), so no specifier in the frontend source tree can resolve to a file outside that tree.

## Modules with zero importers (121)

A module counts as zero-importer when no other module in its own scanned tree resolves an in-repo import to it (`imported_by_count == 0` in the regenerated inventory). This includes package entry points (`__init__.py`, `__main__.py`), CLI scripts under `scripts/` invoked as processes rather than imported, and every TypeScript `*.test.ts`/`*.test.tsx` file, none of which are expected to have an in-repo importer.

| Root | Module path |
|---|---|
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__main__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/components/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/components/backbones.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/components/heads.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/components/necks.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/data/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/inference/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/training/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/tools/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/utils/__init__.py |
| tcip-web | packages/tcip-web/src/tcip_web/__init__.py |
| tcip-web | packages/tcip-web/src/tcip_web/__main__.py |
| tcip-web | packages/tcip-web/src/tcip_web/agent_bash_guard.py |
| tcip-web | packages/tcip-web/src/tcip_web/agent_powershell_guard.py |
| tcip-web | packages/tcip-web/src/tcip_web/agent_session_start.py |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/classes.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/client.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/devProxy.generated.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/http.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/inference.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/streams.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/training.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/ws.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/AnnotateToolbar.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/BandPicker.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/CanvasStage.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/CoverageMinimap.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/zoom.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/EmbeddedTool.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ErrorBoundary.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/HelpOverlay.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ProjectBreadcrumb.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ProjectPicker.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/SeasonRail.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/StatusBar.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/TabBanner.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/TerminalRail.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useCoverageTracking.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageBands.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageNav.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useRegionCompleteness.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/index.css.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/annotateFocus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/bandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/canvasSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/coverage.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/coverageTracker.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/editGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/imageLoader.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/imageStatus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/labelSerde.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/openProject.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/paths.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reviewFocus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reviewGeometry.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/runStatus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/viewGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/main.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/guiStateShape.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/mergeSnapshot.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/store.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/tabRestore.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/terminalOpenPolicy.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/AnnotateTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/InferenceTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/ResultsTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/ReviewTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/test/setup.ts |
| scripts | scripts/adopt_store.py |
| scripts | scripts/calibrate_operating_point.py |
| scripts | scripts/check_architecture_citations.py |
| scripts | scripts/check_architecture_doc.py |
| scripts | scripts/check_dataset_identity.py |
| scripts | scripts/compute_disagreements.py |
| scripts | scripts/conform_cal_holdout_locks.py |
| scripts | scripts/conform_metrics_marker.py |
| scripts | scripts/conform_project_site.py |
| scripts | scripts/conform_registry_experiment_id.py |
| scripts | scripts/conform_view_coverage_viewing.py |
| scripts | scripts/cross_family_ask.py |
| scripts | scripts/distill_learnings.py |
| scripts | scripts/doctor.py |
| scripts | scripts/drop_annotation_stats_image_status.py |
| scripts | scripts/drop_trait_spec_provenance.py |
| scripts | scripts/export_store.py |
| scripts | scripts/foreground_fn_candidates.py |
| scripts | scripts/gate_baseline.py |
| scripts | scripts/generate_frontend_routes.py |
| scripts | scripts/generate_frontend_types.py |
| scripts | scripts/generate_frozen_manifest.py |
| scripts | scripts/inspect_baseline_weights.py |
| scripts | scripts/inspect_gps_exif.py |
| scripts | scripts/list_tools.py |
| scripts | scripts/plant_aware_group_splits.py |
| scripts | scripts/prove_test_fails_before.py |
| scripts | scripts/render_candidates_tile.py |
| scripts | scripts/shp_to_plant_csv.py |
| scripts | scripts/smoke_fence_e2e.py |
| scripts | scripts/smoke_phenology_e2e.py |
| scripts | scripts/smoke_terminal_e2e.py |
| scripts | scripts/verify_citations.py |
| scripts | scripts/verify_claims.py |
| scripts | scripts/verify_doc_examples.py |
| scripts | scripts/verify_skill_tools.py |
| scripts | scripts/verify_skill_traits.py |
| scripts | scripts/watch_agent_chat.py |


## Public surface

Every tool, route, symbol and line in this section is the one standing at HEAD
f943c12d4693409bcf2a0c1a4229d26b61d34cc1.

## 1. MCP tools

`packages/tcip-mcp/src/tcip_mcp/server.py:9` defines `mcp = MCPServer("tcip-pipeline")`.
`python scripts/list_tools.py` is the authority for the registered tool list and count; a
count written here drifts, so none is. Every registered tool's name matches an `@mcp.tool()`
decorator site in `packages/tcip-mcp/src/tcip_mcp/tools/*.py` and each such site is
immediately followed by an `@audited` decorator (some spelled `@audited(scope_arg=...)`, so a
check must match the decorator name rather than the whole line).

Tables below group by defining module. Column "line" is the `def`/`async def` line.
Docstring is the function's docstring first line, verbatim.

### annotation_tools.py (8 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `read_annotations` | `annotation_tools.py:92` | yes | Load the ground-truth labels and predictions for a single image. |
| `save_annotations` | `annotation_tools.py:141` | yes | Write an image's annotations to its single per-image label file (all subjects, one file). |
| `score_predictions` | `annotation_tools.py:443` | yes | Score on-disk predictions against on-disk ground truth (COCOeval). |
| `segment_prompt` | `annotation_tools.py:479` | yes | Turn an interactive prompt (points, a box, or grid cells) into mask polygon rings, via an engine. |
| `push_panel_data` | `annotation_tools.py:575` | yes | Push structured data to a TCIP GUI panel via the tcip-web backend. |
| `focus` | `annotation_tools.py:609` | yes | Drive the live GUI to a (subject, date) frame, the Annotate tab or the Review tab. |
| `stage_proposals` | `annotation_tools.py:851` | yes | Stage model-/agent-proposed shapes to ``predictions/<model>/<date>/<stem>.json`` for canvas |
| `write_class_map` | `annotation_tools.py:1043` | yes | Author the dataset's nested class registry, a thin wrapper over ``class_registry``. |

### data_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `validate_data_quality` | `data_tools.py:299` | yes | Run quality checks on a dataset (any supported annotation format). |  <!-- queued: P5-18 unify -->
| `make_splits` | `data_tools.py:449` | yes | Compute a leakage-free, annotation-stratified train/val/calibration split. |

### experiment_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `create_experiment` | `experiment_tools.py:11` | yes | Create a new experiment to track a training run. |
| `get_experiment` | `experiment_tools.py:41` | yes | Read an experiment record. |
| `list_experiments` | `experiment_tools.py:83` | yes | Enumerate every experiment the store holds a status record for. |
| `compare_experiments` | `experiment_tools.py:106` | yes | Side-by-side comparison of multiple experiments. |

### feedback_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `materialize_review_dataset` | `feedback_tools.py:166` | yes | Build a curated detection dataset from human review verdicts. |
| `prioritize_review_queue` | `feedback_tools.py:286` | yes | Order un-reviewed images for the next review batch. |  <!-- queued: P5-51 merge-or-split -->

### inference_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `force_redraw_cal_holdout_split` | `inference_tools.py:505` | yes | Deliberately redraw a locked calibration/holdout split. |
| `run_inference` | `inference_tools.py:706` | yes | Run a trained model on images. |
| `export_predictions` | `inference_tools.py:1763` | yes | Run inference and save predictions as COCO/JSON prediction file(s). |  <!-- queued: P5-36 unify -->
| `tabulate_counts` | `inference_tools.py:2069` | yes | Run inference and export a CSV summary of detection counts per image. |  <!-- queued: P5-37 merge-or-split -->

### ingest_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `ingest_images` | `ingest_tools.py:172` | yes | Copy raw images into a structured project, bucketed by the capture date each file states. |

### meta_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `claude_reports` | `meta_tools.py:237` | yes | Log structured friction when you get stuck, confused, or surprised. |
| `load_project_memory` | `meta_tools.py:306` | yes | Read one project-memory corpus into context so context isn't lost between sessions. |
| `project_retrospective` | `meta_tools.py:393` | yes | Write an end-of-project retrospective to markdown. |
| `record_distillation_pass` | `meta_tools.py:496` | yes | Record that you reviewed this project's friction/retrospectives (e.g. via |

### model_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_model` | `model_tools.py:18` | yes | Register a trained model in the project model registry. |  <!-- queued: P5-23 merge-or-split -->
| `list_registered_models` | `model_tools.py:78` | yes | List models in the project registry. |
| `select_best_model` | `model_tools.py:126` | yes | Get the best registered model by an explicit metric, no default is assumed. |

### operationalization_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `state_trait_operationalization` | `operationalization_tools.py:19` | yes | Record what this trait's delivered number means, in the breeder's terms, for one delivery. |

### trait_spec_authoring_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `author_trait_spec` | `trait_spec_authoring_tools.py:18` | yes | Register a trait that does not yet exist, and record why, in the breeder's terms. |

### orthomosaic_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `deliver_orthomosaic_plant_counts` | `orthomosaic_tools.py:36` | yes | Per-plant detection counts from a persisted orthomosaic prediction bucket + plant CSV(s). |

### phenology_tools.py (5 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `build_plant_mapping` | `phenology_tools.py:27` | yes | Assign each geolocated image to a plant, then persist the mapping under this project. |
| `update_trait_spec_fields` | `phenology_tools.py:156` | yes | Update one or more fields on an already-registered trait's spec. |
| `calibrate_classifier_operating_point` | `phenology_tools.py:435` | yes | Calibrate and validate the trait's positive-class classifier against held-out GT. |
| `calibrate_ordinal_regression_operating_point` | `phenology_tools.py:605` | yes | Calibrate and validate a trait's ordinal-rank or continuous-value prediction against a |
| `compute_phenology` | `phenology_tools.py:804` | yes | Per-plant phenology milestones from classified predictions + a plant mapping. |  <!-- queued: P5-43 unify -->

### project_tools.py (7 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_dataset` | `project_tools.py:162` | yes | Record a dataset's identity so a delivered number can be traced to the exact data behind it. |
| `init_project` | `project_tools.py:272` | yes | Initialise a TCIP project directory. |
| `set_active_project` | `project_tools.py:306` | yes | Set the workspace's active project so the GUI opens it. |
| `view_gui_state` | `project_tools.py:395` | yes | The live GUI session the human is looking at: active project, dataset, date, trait, tab, and the |
| `inspect_project` | `project_tools.py:442` | yes | Get an overview of a TCIP project. |
| `archive_project` | `project_tools.py:582` | yes | Export an annotation project as a portable ZIP archive. |  <!-- queued: P5-07 demote-to-script -->
| `import_project` | `project_tools.py:809` | yes | Import an annotation project from a ZIP archive. |  <!-- queued: P5-08 demote-to-script -->

`tools/bundle.py` (not a tool module: no `@mcp.tool()` sites) is the one membership accounting
`archive_project` and `import_project` both compose from, `account_for(tree)`. It derives every
root a project tree is or holds (the fixed `.tcip` structure, plus every `split_manifest.json`/
`curated_manifest.json` anchor under placement constraints that raise `AnchorMisplaced` when one
sits at the tree root, under `.tcip`, under a blob home, or under/above another derived root),
then classifies every file by precedence: bookkeeping, a record or log claimed by exactly one
derived root's own layout (`tcip_store.adoption.plan_root`; two roots claiming one file is a
`collisions` entry on the result, never a specificity tie), a recognized blob home, or
unaccounted. `archive_project` bundles the record/log and blob classes; `import_project` refuses
on any bookkeeping, collided, undecodable or unaccounted member before adopting or moving
anything.

### scale_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `calibrate_physical_scale` | `scale_tools.py:84` | yes | Derive and validate a physical per-pixel scale, and stamp it into ``pred_dir``'s |

### training_tools.py (8 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `preflight_config` | `training_tools.py:68` | yes | Validate a training configuration before launching. |
| `launch_training` | `training_tools.py:521` | yes | Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder. |
| `check_training_status` | `training_tools.py:749` | yes | Check the status of a training run. |
| `list_training_runs` | `training_tools.py:885` | yes | List every training run this platform can currently account for. |
| `cancel_training` | `training_tools.py:899` | yes | Request graceful cancellation of a running training run. |
| `inspect_compute_resources` | `training_tools.py:928` | yes | Report the host's current compute headroom, a fact to reason with before launching |  <!-- queued: P5-31 demote-to-script -->
| `run_hpo` | `training_tools.py:1319` | yes | Run hyperparameter optimization on Ray Tune, training each trial for real. |
| `evaluate_model` | `training_tools.py:2586` | yes | Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json. |

### vision_tools.py (6 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `visualize` | `vision_tools.py:295` | yes | Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid. |
| `render_failure_cases` | `vision_tools.py:517` | yes | Find and render the worst predictions for failure analysis. |  <!-- queued: P5-45 demote-to-script -->
| `propose_annotations` | `vision_tools.py:750` | yes | Propose candidate annotations on an image for review, using a chosen auto-labeling engine. |
| `accept_proposals` | `vision_tools.py:951` | yes | Assign classes to reviewed proposals and stage them as predictions for canvas review. |
| `capture_live_canvas` | `vision_tools.py:1081` | yes | Render exactly what the human's GUI canvas shows right now: image, shapes, viewport. |
| `overlay_reference_grid` | `vision_tools.py:1211` | yes | Render image with a labeled reference-grid overlay for spatial referencing. |

8 + 2 + 4 + 2 + 4 + 1 + 4 + 3 + 1 + 1 + 5 + 7 + 1 + 8 + 1 + 6 = 58 tools across 16 modules.

## 2. HTTP routes and WebSocket endpoints

`packages/tcip-web/src/tcip_web/app.py` builds the FastAPI app, registers 6 HTTP routes
and 2 WebSocket routes directly, then calls `register_all(app)` from
`packages/tcip-web/src/tcip_web/routes/__init__.py`, which `include_router`s 16 route
modules under `routes/`, each with a fixed prefix. Verified: `routes/__init__.py`,
`routes/_metrics_common.py`, and `routes/_body_common.py` define no routes of their own (0
`@router.*` decorator sites in any of the three); `_metrics_common.py` holds `metrics_response`,
the response shape `training.py` and `tuning.py` both answer in, and `_body_common.py` holds
`EmptyBodyPayload`, the empty body model six path-parameter-only routes now declare so the
browser must send a preflighted request rather than reaching the handler as a simple one.

Total HTTP routes at HEAD: 89 (6 on `app.py` plus 83 across the 16 route modules, both counts
obtained this session by grepping `@app.get/post(` and `@router.get/post(` and summing);
websocket routes are counted separately, below, and excluded from this total. Each per-router
heading's own route count (and their sum, 86) includes any websocket route it lists, since
`routes/inference.py`, `routes/terminal.py` and `routes/training.py` each carry one; net of
those three, the 16 modules hold the 83 HTTP routes counted here.

Total WebSocket routes at HEAD: 5 (`/ws/state`, `/ws/panel/{panel}` on `app.py`;
`/api/terminal/ws/{session_id}` on `routes/terminal.py`; `/api/inference/jobs/{job_id}/stream`  <!-- queued: P5-124 unify -->
on `routes/inference.py`; `/api/training/runs/{run_id}/stream` on `routes/training.py`).

Tables below group by defining module. Column "line" is the handler's `def`/`async def` line,
the same convention the tool tables use; the `@router.*`/`@app.*` decorator carrying the method
and path sits directly above it. Every method, path, and handler name below is the one
registered at HEAD.

### app.py (not under a router prefix)

| method | path | handler | line |
|---|---|---|---|
| GET | `/api/state` | `get_state` | `app.py:204` |
| POST | `/api/state/tab` | `set_active_tab` | `app.py:213` |
| WS | `/ws/state` | `state_ws` | `app.py:221` |
| GET | `/health` | `health` | `app.py:315` |
| GET | `/` | `index` | `app.py:323` |
| POST | `/api/events/{panel}` | `post_panel_event` | `app.py:387` |
| GET | `/api/events/{panel}/recent` | `get_recent_panel_events` | `app.py:438` |  <!-- queued: P5-82 delete -->
| WS | `/ws/panel/{panel}` | `panel_ws` | `app.py:447` |

### routes/annotate.py, prefix `/api/annotate` (3 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/labels` | `load_labels` | `routes/annotate.py:171` |
| POST | `/labels` | `save_labels` | `routes/annotate.py:195` |
| POST | `/open` | `open_image` | `routes/annotate.py:253` |  <!-- queued: P5-58 delete -->

### routes/canvas.py, prefix `/api/canvas` (1 route)

| method | path | handler | line |
|---|---|---|---|
| POST | `/state` | `push_canvas_state` | `routes/canvas.py:70` |

### routes/classes.py, prefix `/api/classes` (6 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/load` | `load_classes` | `routes/classes.py:97` |
| POST | `/save` | `save_classes` | `routes/classes.py:161` |
| GET | `/image_status` | `get_image_status` | `routes/classes.py:295` |
| POST | `/image_status` | `set_image_status` | `routes/classes.py:305` |
| POST | `/image_status/bulk` | `set_image_status_bulk` | `routes/classes.py:336` |
| POST | `/image_status/derive` | `derive_image_status` | `routes/classes.py:366` |

### routes/coverage.py, prefix `/api/coverage` (5 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/grid` | `get_grid` | `routes/coverage.py:92` |
| GET | `` (root) | `get_coverage` | `routes/coverage.py:143` |
| POST | `` (root) | `post_coverage` | `routes/coverage.py:189` |
| GET | `/completeness` | `get_completeness` | `routes/coverage.py:279` |
| POST | `/completeness` | `post_completeness` | `routes/coverage.py:324` |

### routes/dataset.py, prefix `/api/dataset` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/tree` | `get_dataset_tree` | `routes/dataset.py:154` |  <!-- queued: P5-83 unify -->
| GET | `/images` | `list_images` | `routes/dataset.py:200` |  <!-- queued: P5-84 delete -->
| POST | `/select` | `select_dataset` | `routes/dataset.py:222` |
| POST | `/nav` | `set_current_image` | `routes/dataset.py:319` |

### routes/fs.py, prefix `/api/fs` (1 route)

| method | path | handler | line |
|---|---|---|---|
| GET | `/list` | `list_dir` | `routes/fs.py:108` |

### routes/images.py, prefix `/api/images` (5 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `` (root) | `serve_image` | `routes/images.py:499` |
| GET | `/dimensions` | `get_dimensions` | `routes/images.py:714` |
| GET | `/bands` | `get_bands` | `routes/images.py:734` |
| POST | `/overviews` | `build_image_overviews` | `routes/images.py:867` |
| GET | `/overviews/status` | `get_overview_job` | `routes/images.py:891` |

### routes/inference.py, prefix `/api/inference` (4 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| POST | `/launch` | `launch_inference` | `routes/inference.py:456` |  <!-- queued: P5-105 delete -->
| GET | `/jobs` | `list_jobs` | `routes/inference.py:555` |
| GET | `/jobs/{job_id}/preview` | `get_preview` | `routes/inference.py:560` |
| POST | `/jobs/{job_id}/cancel` | `cancel_job` | `routes/inference.py:574` |
| WS | `/jobs/{job_id}/stream` | `stream_job` | `routes/inference.py:584` |

### routes/meta.py, prefix `/api/meta` (2 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/reports` | `get_reports` | `routes/meta.py:36` |
| GET | `/retrospectives` | `get_retrospectives` | `routes/meta.py:58` |

### routes/projects.py, prefix `/api/projects` (2 routes)  <!-- queued: P5-88 unify -->

| method | path | handler | line |
|---|---|---|---|
| GET | `` (root) | `list_projects` | `routes/projects.py:100` |
| POST | `/active` | `set_active_project` | `routes/projects.py:142` |  <!-- queued: P5-90 move-to-gui-or-automatic -->

### routes/results.py, prefix `/api/results` (15 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/plant_mapping/build` | `build_plant_mapping` | `routes/results.py:178` |
| POST | `/plant_mapping/load` | `load_plant_mapping` | `routes/results.py:270` |  <!-- queued: P5-129 delete -->
| GET | `/plant_mapping/list` | `list_plant_mappings` | `routes/results.py:289` |
| POST | `/per_plant_curves` | `per_plant_curves` | `routes/results.py:561` |  <!-- queued: P5-130 merge-or-split -->
| POST | `/onset_dates` | `onset_dates` | `routes/results.py:595` |  <!-- queued: P5-131 merge-or-split -->
| POST | `/export_csv` | `export_csv` | `routes/results.py:629` |
| GET | `/traits` | `list_traits` | `routes/results.py:1054` |
| GET | `/operationalization` | `get_operationalization` | `routes/results.py:753` |
| GET | `/operationalizations` | `list_operationalizations` | `routes/results.py:769` |
| POST | `/operationalization/confirm` | `confirm_operationalization` | `routes/results.py:807` |
| GET | `/trait-spec-statement` | `get_trait_spec_statement` | `routes/results.py:909` |
| GET | `/trait-spec-statements` | `list_trait_spec_statements` | `routes/results.py:925` |
| POST | `/trait-spec-statement/confirm` | `confirm_trait_spec_statement` | `routes/results.py:967` |
| GET | `/delivery-events` | `list_delivery_events` | `routes/results.py:1033` |
| GET | `/models/registered` | `registered_models` | `routes/results.py:1083` |

### routes/review.py, prefix `/api/review` (11 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/matches` | `compute_image_matches` | `routes/review.py:451` |
| POST | `/action` | `record_action` | `routes/review.py:577` |
| POST | `/mark_complete` | `mark_complete` | `routes/review.py:713` |
| POST | `/backup_labels` | `backup_labels` | `routes/review.py:775` |
| POST | `/save_gt` | `save_gt` | `routes/review.py:796` |
| POST | `/validate_reference` | `validate_reference` | `routes/review.py:848` |
| GET | `/image_status` | `get_image_status` | `routes/review.py:1196` |
| GET | `/image_statuses` | `image_statuses` | `routes/review.py:1249` |
| GET | `/generation_conf` | `get_generation_conf` | `routes/review.py:1279` |
| POST | `/queue/launch` | `launch_priority_queue` | `routes/review.py:1461` |
| GET | `/queue/{job_id}` | `get_priority_queue_job` | `routes/review.py:1489` |

### routes/sessions.py, prefix `/api/sessions` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/image_event` | `image_event` | `routes/sessions.py:118` |
| POST | `/start` | `start_session` | `routes/sessions.py:186` |
| POST | `/end` | `end_session` | `routes/sessions.py:207` |
| GET | `/load` | `load_sessions` | `routes/sessions.py:223` |

### routes/terminal.py, prefix `/api/terminal` (4 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| GET | `/status` | `get_status` | `routes/terminal.py:253` |
| POST | `/sessions` | `create_session` | `routes/terminal.py:282` |
| GET | `/sessions` | `list_terminal_sessions` | `routes/terminal.py:305` |
| POST | `/sessions/{session_id}/restart` | `restart_session` | `routes/terminal.py:314` |
| WS | `/ws/{session_id}` (full path `/api/terminal/ws/{session_id}`) | `terminal_ws` | `routes/terminal.py:342` |

### routes/training.py, prefix `/api/training` (8 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| POST | `/validate` | `preflight_config_route` | `routes/training.py:54` |  <!-- queued: P5-104 delete -->
| POST | `/launch` | `launch_training_route` | `routes/training.py:68` |  <!-- queued: P5-113 delete -->
| GET | `/runs` | `list_runs_route` | `routes/training.py:88` |
| GET | `/runs/{run_id}` | `get_run` | `routes/training.py:101` |
| POST | `/runs/{run_id}/tensorboard` | `launch_run_tensorboard` | `routes/training.py:108` |
| POST | `/runs/{run_id}/cancel` | `cancel_run_route` | `routes/training.py:128` |
| GET | `/runs/{run_id}/metrics` | `get_run_metrics` | `routes/training.py:143` |  <!-- queued: P5-110 delete -->
| POST | `/compare` | `compare_runs_route` | `routes/training.py:166` |
| WS | `/runs/{run_id}/stream` (full path `/api/training/runs/{run_id}/stream`) | `training_stream_ws` | `routes/training.py:245` |

### routes/tuning.py, prefix `/api/tuning` (9 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/launch` | `launch_hpo` | `routes/tuning.py:320` |
| GET | `/sweeps` | `list_sweeps` | `routes/tuning.py:351` |
| GET | `/sweeps/{sweep_id}` | `get_sweep` | `routes/tuning.py:369` |
| GET | `/sweeps/{sweep_id}/trials` | `list_trials` | `routes/tuning.py:402` |
| GET | `/sweeps/{sweep_id}/trials/{trial_id}/metrics` | `get_trial_metrics` | `routes/tuning.py:435` |
| GET | `/ray-dashboard` | `get_ray_dashboard` | `routes/tuning.py:462` |
| POST | `/sweeps/{sweep_id}/tensorboard` | `launch_sweep_tensorboard` | `routes/tuning.py:548` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard` | `launch_trial_tensorboard` | `routes/tuning.py:565` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop` | `stop_trial_tensorboard` | `routes/tuning.py:579` |

### 11 routes with no located frontend caller

Per phase0's `web-surface.md`, these 11 registered backend routes had no caller found
under `packages/tcip-web/frontend/src/` by literal-path grep. Not re-derived this
session; restated from phase0 as the brief instructs, with each route's line verified
above against HEAD.

| method | path | note (from phase0) |
|---|---|---|
| GET | `/health` | not fetched from `frontend/src`; a liveness endpoint |
| GET | `/` | loaded by the browser's own navigation, not via fetch/XHR from app code |
| POST | `/api/events/{panel}` | posted by MCP tools (`tcip_mcp.web_client`), not by the browser |
| GET | `/api/events/{panel}/recent` | no caller found |
| GET | `/api/state` | no caller found |
| GET | `/api/images/dimensions` | no caller found |  <!-- queued: P5-72 delete -->
| GET | `/api/inference/jobs/{job_id}/preview` | no caller found; `inferenceApi` has no `preview` function |  <!-- queued: P5-125 delete -->
| GET | `/api/review/image_status` | singular form; no caller found (plural `/api/review/image_statuses` is called) |  <!-- queued: P5-141 delete -->
| POST | `/api/review/save_gt` | no caller found; `api.review` has no `saveGt` function |  <!-- queued: P5-139 delete -->
| GET | `/api/terminal/sessions` | no caller found; `terminalApi` has no list function |  <!-- queued: P5-99 delete -->
| POST | `/api/training/compare` | no caller found; `trainingApi` has no `compare` function |  <!-- queued: P5-111 delete -->

## 3. tcip-annotation importable public symbols

`packages/tcip-annotation/src/tcip_annotation/__init__.py:40-76` defines `__all__`.
Counted this session by parsing the module's AST: 27 entries.

differs from phase0 record: `docs/audit/phase0/surface-formats/annotation-api.md` states
in prose ("re-exports the following 26 names") that `__all__` has 26 names; its own table
under "Package-level exports" (lines 40-66 of that file) in fact lists 27 rows, matching
the HEAD count of 27. The HEAD fact is 27; the phase0 document's summary sentence
undercounts its own table by one.

| name | re-exported from | `__init__.py` line |
|---|---|---|
| `Annotation` | `state` | 3 |
| `AnnotationState` | `state` | 3 |
| `BBox` | `state` | 3 |
| `Point` | `state` | 3 |
| `Polygon` | `state` | 3 |
| `bbox_of` | `state` | 3 |
| `read_annotations` | `json_io` | 12 |
| `write_annotations` | `json_io` | 12 |
| `to_coco_dataset` | `json_io` | 12 |
| `detect_format` | `format_io` | 17 |
| `load_annotations_any` (alias of `format_io.load_annotations`) | `format_io` | 17-19 |
| `save_annotations_any` (alias of `format_io.save_annotations`) | `format_io` | 17-20 |
| `parse_coco_annotations` | `format_io` | 21 |
| `write_coco` | `format_io` | 22 |
| `compute_matches` | `matching` | 24 |
| `compute_classified_trait_matches` | `matching` | 24 |
| `box_iou` | `matching` | 27 |
| `polygon_iou` | `matching` | 28 |
| `point_in_polygon` | `matching` | 29 |
| `mask_to_polygon_rings` | `mask_contours` | 32 |
| `cell_fields` | `sam_wrapper` | 38 |
| `grid_to_pixel` | `sam_wrapper` | 38 |
| `auto_mask` | `sam_wrapper` | 38 |
| `AnnotationEngine` | `annotation_engine` | 33 |
| `ReviewEngine` | `review_engine` | 34 |
| `ReviewDetection` | `review_engine` | 34 |
| `ReviewContext` | `review_engine` | 34 |

Package no-dependency claim, restated from phase0 and not re-derived this session: no
`import tcip_mcp` / `import tcip_web` statement exists anywhere in
`packages/tcip-annotation/src/tcip_annotation/`; `packages/tcip-annotation/CLAUDE.md`
states the same rule and is loaded automatically when reading files in the package.

Names not re-exported in `__all__` but importable directly from their defining submodule
(restated from phase0, not re-derived this session): `json_io.ANNOTATIONS_KEY`,
`json_io.UNLABELED`, `json_io.target_class_id`,
`format_io.AnnotFormat`, `format_io.load_annotations`, `format_io.save_annotations`,
`mask_contours.DEFAULT_EPSILON_FRAC`, `annotation_engine.Snapshot`,
`annotation_engine.UNDO_DEPTH`, `review_engine.REVIEW_SHARD_DIRNAME`,
`sam_wrapper.checkpoint_path`, `sam_wrapper.predict_from_point`,
`sam_wrapper.predict_from_points`, `sam_wrapper.predict_from_box`,
`sam_wrapper.column_label`, `sam_wrapper.column_index`, `utils.auto_orient_image`,
`utils.get_image_dimensions`, and every public name in `viz.py`
(`COLOR_PALETTE`, `render_detections`, `render_segmentations`, `render_comparison`,
`render_grid`, `render_candidates`, `render_grid_overlay`, `render_canvas_state`).

## 4. Entry points

`python -m tcip_mcp`: `packages/tcip-mcp/src/tcip_mcp/__main__.py:1-5` imports `main`
from `tcip_mcp.server` and calls it: `packages/tcip-mcp/src/tcip_mcp/server.py:65`
(`def main()`). `server.py:9` defines `mcp = MCPServer("tcip-pipeline")`, the object every
`@mcp.tool()` decorator in `packages/tcip-mcp/src/tcip_mcp/tools/*.py` registers against
(`python scripts/list_tools.py` lists them; the count is never written down, since it drifts).

`python -m tcip_web`: `packages/tcip-web/src/tcip_web/__main__.py` defines `main()`
(line 61), reads `TCIP_WEB_HOST` / `TCIP_WEB_PORT` (default `127.0.0.1:8765`, lines 22-23,
67-68), writes the bound port under the workspace root's `.tcip/state/web_port.txt`
(`_write_port_file`, line 43), and starts the app via `uvicorn.run("tcip_web.app:app", ...)`
(line 72). The port write resolves under the workspace root, not the platform-state root, so
it needs no pin first; the app pins the platform-state root at the lifespan's startup or,
when a request is served before the lifespan has run, at that first request, never at import:
`bind_startup_root` (`packages/tcip-web/src/tcip_web/app.py:91`) reads the workspace's
active-project marker (`tcip_mcp.project_paths.pin_project_root(from_marker=True)`) the first
time either the lifespan (`app.py:50`) or the startup middleware (`app.py:131`) calls it, so
every way the app is served (this entry point, a bare `uvicorn tcip_web.app:app`,
`--lifespan off`, the reloader's child) pins a root before anything resolves a `.tcip` path. Exposure is a property
of the accepted connection rather than the configured bind host, so this entry point always
binds the requested host and port; whether an arrival through a non-loopback address is served
is decided per request by `tcip_web.trust_boundary.TrustBoundaryMiddleware` (`trust_boundary.py:
282`), which refuses one unless `TCIP_WEB_ALLOW_INSECURE=1` is set (`insecure_opt_in`,
`trust_boundary.py:128`).

`.mcp.json` (repo root): declares two MCP servers. `tcip` launches
`conda run -n tcip-agent --no-capture-output python -m tcip_mcp`. `claude-context`
launches `.claude/hooks/claude_context_launch.cmd` with `EMBEDDING_PROVIDER`,
`OLLAMA_HOST`, `EMBEDDING_MODEL`, `MILVUS_ADDRESS`, `HYBRID_MODE`, and
`CLAUDE_CONTEXT_BACKGROUND_SYNC` set in its `env` block.

`scripts/` (repo root): a non-API surface, not imported by `tcip_mcp`, `tcip_web`, or
`tcip_annotation` package code; each file is a standalone script invoked directly
(`python scripts/<name>.py`). 34 files in `scripts/`, excluding `README.md` and
`__pycache__` (33 `.py` files and one `.ps1` file):
`_paths.py`, `_store_bootstrap.py`, `adopt_store.py`, `calibrate_operating_point.py`,
`check_architecture_citations.py`, `check_architecture_doc.py`, `check_dataset_identity.py`,
`compute_disagreements.py`, `conform_view_coverage_viewing.py`, `cross_family_ask.py`,
`distill_learnings.py`, `doctor.py`, `drop_trait_spec_provenance.py`, `export_store.py`,
`foreground_fn_candidates.py`, `gate_baseline.py`, `generate_favicon.ps1`,
`generate_frontend_routes.py`, `generate_frontend_types.py`, `inspect_baseline_weights.py`,
`inspect_gps_exif.py`, `list_tools.py`, `plant_aware_group_splits.py`,
`prove_test_fails_before.py`, `render_candidates_tile.py`, `shp_to_plant_csv.py`,
`smoke_fence_e2e.py`, `smoke_phenology_e2e.py`, `smoke_terminal_e2e.py`, `verify_citations.py`,
`verify_claims.py`, `verify_doc_examples.py`, `verify_skill_traits.py`,
`watch_agent_chat.py`.


## On-disk formats

25 formats are inventoried in `docs/audit/phase0/surface-formats/ondisk-formats.md` and
`ondisk-formats.json` (`formats` array, length 25). Every writer/reader symbol:line citation
below names where that symbol stands at HEAD f943c12d, which for many is no longer the line the
phase0 record gave.
Where a phase3 seam record (`docs/audit/phase3/seams/`, adjudicated from
`docs/audit/phase2/seam-coverage/seam-coverage.json`) covers a format's writer/reader
agreement, its seam id, coverage verdict, and implementation-sharing judgment
(`phase0_implementation`: `once, shared` | `mixed` | `written twice`) are cited. A format with
no matching name among the 67 seam ids in `seam-coverage.json` is marked "no seam record".

## 1. Annotation JSON, canonical per-image label file

Path: `<dataset_root>/annotations/[<date>/]<stem>.json` (ground truth);
`<dataset_root>/predictions/<model>/[<date>/]<stem>.json` (predictions, identical schema).

Writers: `tcip_annotation.json_io.write_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:703`;
`tcip_annotation.format_io.save_annotations` (`fmt="json"`),
`packages/tcip-annotation/src/tcip_annotation/format_io.py:283`;
`tcip_annotation.review_engine.ReviewEngine.save_gt`,
`packages/tcip-annotation/src/tcip_annotation/review_engine.py:848`;
`tcip_mcp.prediction_buckets.stage_prediction_shapes`,
`packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:254`.

Readers: `tcip_annotation.json_io.read_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:517`;
`tcip_annotation.format_io.load_annotations`,
`packages/tcip-annotation/src/tcip_annotation/format_io.py:349`;
`tcip_mcp.dataset_layout.subjects_on_date`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:1026`.

A prediction record's `created_by` is one spelling, `resolution.prediction_producer`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:1084`, so every checkpoint-backed writer
stamps the same `model:<checkpoint-stem>@<sha256-prefix>` producer identity rather than each door
composing its own string.

Seam S17 ("Canonical per-image annotation JSON schema"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: every writer and the shared `json_io.read_annotations`
reader are exercised in round-trip tests
(`tests/test_tcip_web_routes.py:264,297,398,420,462`, `tests/test_review_engine.py:372`,
`tests/test_review_channel.py:103`, `tests/test_name_based_annotation_schema.py:200,246`). Gap
recorded by S17: no test cross-writes through one production writer and cross-reads through a
different consumer path in the same test.

## 2. Assembled COCO dataset JSON, interop format

Path: caller-supplied, single dataset-level `.json` file, not per-image.

Writers: `tcip_annotation.format_io.write_coco`,
`packages/tcip-annotation/src/tcip_annotation/format_io.py:275`;
`tcip_annotation.format_io.save_annotations` (`fmt="coco"`), `format_io.py:283`;
`tcip_annotation.json_io.to_coco_dataset` (returns dict, performs no file I/O),
`packages/tcip-annotation/src/tcip_annotation/json_io.py:413`.

Readers: `tcip_annotation.format_io.parse_coco_annotations`, `format_io.py:226`;
`tcip_annotation.format_io.load_annotations` (`fmt="coco"` or auto-detected via `detect_format`,
`format_io.py:68`), `format_io.py:263`.

No seam id in `seam-coverage.json`'s 67-entry inventory names the COCO write/read agreement
directly. S19 ("Annotation format detection scope (json, coco)") is the nearest seam but covers
only `detect_format`'s refusal behavior on an unrecognized store, not COCO writer/reader schema
agreement; `phase0_implementation: once, shared` for S19 (`tests/test_mcp_tools_integration.py:185`,
`tests/test_format_io.py:198,209`).

## 3. `classes.json`, class registry

Path: `<dataset_root>/classes.json`.

Writer: `tcip_mcp.class_registry.replace_registry`,
`packages/tcip-mcp/src/tcip_mcp/class_registry.py:346`, the one write both registry doors call
(the GUI's `save_classes` and the tool's `write_class_map`,
`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:1043`).

Readers: `tcip_mcp.class_registry.read_registry`, `class_registry.py:213`;
`tcip_mcp.dataset_layout.list_subjects` (delegates to `class_registry`),
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:744`.

`assign_class_ids`, `class_registry.py:445`, derives the training-time name-to-id map from this
file's declared attribute order; no integer id is stored in the file itself.
`attribute_schema_digest`, `class_registry.py:162`, hashes a subject's attribute
name/type/values for the `image_status_digest.json` staleness stamp (format 6).

Seam S20 ("classes.json class registry"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_name_based_annotation_schema.py:84` writes the
registry through the real `write_registry` and reads `num_classes` back through the real training
loader's `class_registry.assign_class_ids` call (`packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:120`).
Gap: no test drives the actual `/api/classes/save` HTTP route in the same test as the training-side
read.

## 4. `dataset.json`, dataset identity

Path: `<dataset_root>/dataset.json`.

Writer: `tcip_mcp.tools.project_tools.register_dataset`,
`packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:162`.

Reader: `tcip_mcp.pipelines.resolution.dataset_fingerprint` (recompute-on-read is the stated
authority; the stored value is a cache),
`packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:628`.

Seam S26 ("dataset.json identity and fingerprint"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_dataset_identity_recording.py:78` calls the
real `register_dataset` writer, then the real `training_tools._dataset_identity` reader, both of
which call the identical `dataset_fingerprint` function, and asserts they agree. Gap: the seam's
named third consumer, `scripts/check_dataset_identity.py`, is never executed by any test.

## 5. `image_status.json`, confirmed-negative / human-Complete store

Path: `<dataset_root>/.tcip/state/image_status.json`.

Writers: `set_image_status`,
`packages/tcip-web/src/tcip_web/routes/classes.py:305`; `set_image_status_bulk`,
`routes/classes.py:336`; `tcip_mcp.tools.data_tools._apply_negative_carry`
(split-materialized copy; every confirmed negative is read by the admission
(`trainable_stems`) before the split's manifest or file tree is written, then
attributed to a split by
`negative_carry = _compute_negative_carry(label_map, bare_parts, image_map, subject, only_date)`
`packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py:692`, then applied by
`_apply_negative_carry(negative_carry, out_dir, subject)`
`packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py:889`).

Readers: `tcip_mcp.pipelines.data.datasets.confirmed_negative_names`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:442`; `_status_bucket_for`,
`packages/tcip-web/src/tcip_web/routes/sessions.py:264`;
`tcip_mcp.class_registry._sweep_schema_change`,
`packages/tcip-mcp/src/tcip_mcp/class_registry.py:284`, which enumerates every bucket of a
subject whose attribute schema is about to change so the confirmations under it can be stamped
before the outgoing digest is gone.

`IMAGE_STATUSES = ("complete", "partial", CONFIRMED_NEGATIVE, "unannotated")`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:781`, imported by the web route module.

The token a Complete stores here is subject-scoped before it ever reaches a writer: `mark_complete`,
`packages/tcip-web/src/tcip_web/routes/review.py:713`, derives it from the GT file through
`annotations_hold_subject`, scoped to the confirmed subject, and the browser posts that value on
through `set_image_status`.

Seam S22 ("image_status.json confirmed-negative store"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_tcip_web_classes_routes.py:132,149`,
`tests/test_confirmations_travel_with_dataset.py:65,95`, `tests/test_doctor.py:48`.
`tests/test_status_digest_stamp_writer.py` writes statuses through the real
`/api/classes/image_status` and `/image_status/bulk` routes and reads them back through the real
`confirmed_negative_names`. Gap: whether `doctor.py` is ever driven against a route-written store
has not been re-verified; MCP-side readers test membership through
`dataset_layout.is_confirmed_negative` against the vocabulary declared beside the resolver.

## 6. `image_status_digest.json`, attribute-schema staleness stamp

Path: `<dataset_root>/.tcip/state/image_status_digest.json`.

Writers: `_stamp_digest`, `packages/tcip-web/src/tcip_web/routes/classes.py:265`, called from
`set_image_status`/`set_image_status_bulk` at confirmation time; and
`tcip_mcp.class_registry._sweep_schema_change`,
`packages/tcip-mcp/src/tcip_mcp/class_registry.py:284`, called through `replace_registry`
(`packages/tcip-mcp/src/tcip_mcp/class_registry.py:346`) by both registry writers,
`save_classes` (`packages/tcip-web/src/tcip_web/routes/classes.py:161`) and `write_class_map`
(`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:1043`), before the new registry lands.
A status and its stamp are two transactions, status first, so unstamped confirmations
legitimately exist; the outgoing registry is the last moment their digest is recoverable, so the
sweep records it there and they read as predating the change instead of as made under the new
vocabulary. Both writers reach the store through the one transactional writer
`dataset_layout.stamp_image_status_digests`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:920`, whose `only_unstamped` argument keeps
the sweep from re-dating a stamp the confirmation-time writer already set.

Reader: `tcip_mcp.pipelines.data.datasets.confirmed_negative_names`'s quarantine logic,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:394`; a name whose stamp no longer
matches the registry's current schema is dropped as `quarantined_stale_definition` rather than
trained as a negative (`datasets.py:296`).

Seam S23 ("image_status_digest.json attribute-schema stamp"), verdict `both-sides-restated`,
`phase0_implementation: once, shared`: `tests/test_confirmations_travel_with_dataset.py:119-266`,
`tests/test_coco_training_assembly.py:591-680`. The shared function across both sides is
`attribute_schema_digest`, `class_registry.py:162`. Both sides run their real implementations
end to end in `tests/test_status_digest_stamp_writer.py` (the HTTP status routes produce the
sidecar, `confirmed_negative_names` reads it) and in
`tests/test_schema_change_staleness_sweep.py` (both registry writers run the sweep,
`confirmed_negative_names` reads the result). Reader-only tests elsewhere still hand-write the
`{bucket: {image_name: digest}}` shape, reusing `attribute_schema_digest` for the value.

## 7. `view_coverage.json`, reference-grid coverage record, advisory

Path: `<dataset_root>/.tcip/state/view_coverage.json`.

Path/shape definition: `tcip_mcp.dataset_layout.view_coverage_path`, `dataset_layout.py:482`,
naming the stored record's own shape as `tcip_web.routes._coverage_models.CoverageRecord`.
Writer and reader: `routes/coverage.py`'s `post_coverage` (`:187`) and `get_coverage` (`:143`),
each validating the stored record against `CoverageRecord` before merging into or serving it.

Seam S24 ("view_coverage.json advisory coverage record"), verdict: single. `_coverage_models.py`
declares `GridGeometry`, `StatsSource`, `WorkingScaleBar`, `CoverageViewing` and `CoverageRecord`
once; `routes/coverage.py`'s `CoveragePayload` types its `grid` and `viewing` fields against those
models; `scripts/generate_frontend_types.py` renders `frontend/src/api/types.generated.ts` from
them, held current by `tests/test_generated_frontend_types.py`; the browser imports the generated
types (`lib/coverageTracker.ts`) rather than hand-declaring its own. `tests/test_coverage_routes.py`'s
`test_post_from_a_plain_rgb_view_round_trips` and `test_post_from_a_composite_view_round_trips`
round-trip through the real `TestClient` route, proving the server accepts and stores the shape the
browser is meant to send. The narrower gap stays open: each body is a Python dict transcribed by
hand (`tests/test_coverage_routes.py:361-420`) rather than the value `coverageTracker.postNow`
itself produces, and the two suites (this one and `coverageTracker.test.ts`, which exercises
`postNow` only against a mocked `post`) run independently, so a drift between what the tracker
actually builds and what this test believes it builds would not be caught by either.

## 8. `region_completeness.json` and `region_completeness_digest.json`

Path: `<dataset_root>/.tcip/state/region_completeness.json` and
`region_completeness_digest.json`, siblings of `image_status.json`.

Path/shape definition: `tcip_mcp.dataset_layout.region_completeness_path`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:565`;
`region_completeness_digest_path`, `dataset_layout.py:604`; shape guard
`normalize_region_completeness_store`, `dataset_layout.py:659`.

Writer/reader named by phase0 but not independently opened this session:
`pipelines/region_completeness.py` as the digest sidecar's writer.

Seam S25 ("region_completeness.json attestation store"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_coverage_routes.py:449`,
`tests/test_block_calibration.py:126,164,203`. The shared functions across the HTTP-route side and
the calibration-gate side are `region_completeness_path` (`dataset_layout.py:565`),
`normalize_region_completeness_store` (`dataset_layout.py:659`) and `status_bucket`
(`dataset_layout.py:693`). Gap: `test_block_calibration.py`'s
`_attest_regions_complete` helper bypasses the HTTP route, writing the store via the same shared
functions the route calls internally rather than via a POST to `/api/coverage/completeness`, so a
bug confined to the route's own HTTP layer would not be caught by the calibration-gate tests.

## 9. `.tcip/audit.jsonl`, append-only tool-call log

Path: `.tcip/audit.jsonl` under the root the entry's scope names: the pinned platform state
root, via `resolve_state`, for a platform event, and the dataset root for an event that changed
a record travelling with the data. That path is what the file backend places the log at and what
`scripts/export_store.py` writes back out; on the default database backend the rows live in that
root's `.tcip/store.db` until they are exported.

Writers: the `@audited` decorator, `packages/tcip-mcp/src/tcip_mcp/audit.py:174`, which records
a call in the platform log unless the tool declares `@audited(scope_arg=...)` naming the
argument that carries the dataset it mutates a record of, resolved by `dataset_scope_of`, line
147 (through the same canonicalizer the tool body uses, when the declaration passes one as
`scope_via`); `record_event`, same file, line 121, which is what code that is not an `@audited`
MCP tool emits through (the training envelope's open/close events, and each GUI route's own
`_audit` helper). Both address the log through `audit_log_key`, same file, line 84, and append
through the storage seam. What a failed append means is where the two part: `record_event` warns
and returns, through `_write_entry`, same file, line 103, because its callers bracket work rather
than follow a mutation; the decorator raises `MutationCommittedWithoutAuditLine`, line 57,
because its append runs after the tool body and a warning there invites a blind retry of a
mutation already on disk. Dataset-scoped entries carry a `scope` field naming their root;
platform entries keep the original shape.

Reader: no production code parses the log's entries. The only production consumer is
`archive_project` (`tools/project_tools.py`), which bundles the file as a claimed ROOT-layout
record through the shared membership accounting (`tcip_mcp.tools.bundle.account_for`) without
opening it. Rows are otherwise read only through the storage seam's `read_log`, which decodes
each line into an opaque mapping, and by tests that access named keys, so a new per-entry field
is additive for every existing consumer.

Seam S06 ("Append-only audit log .tcip/audit.jsonl"), verdict `one-side-only`,
`phase0_implementation: mixed`: `tests/test_audit_experiments.py:25,57`,
`tests/test_tcip_web_routes.py:480,813,845`, `tests/test_tcip_web_results_routes.py:534`,
`tests/test_tcip_web_classes_routes.py:394,440`. Each writer is exercised through the real append
and checked for its own tool name landing in the file. A GUI route's `_audit` helper composes no
entry of its own: it calls `record_event` with a `source` of `"gui"` and the scope its event
belongs to, and `tests/test_audit_row_core_field_agreement.py:26` runs a real route and a real
platform write against one log and holds the two rows to the same core fields.

## 10-15. `.tcip/experiments/<experiment_id>/`, eight sub-formats

Path root: `.tcip/experiments/<experiment_id>/`, resolved via `experiments_dir()`,
`packages/tcip-mcp/src/tcip_mcp/experiments.py:49`, against the pinned platform state root. Each
member is a store of its own with its own key constructor beside it, and every writer and reader
below addresses the member through that key rather than composing a path. Eight members are
declared; the numbered range 10-15 carries six of them, and `env.json` and `validations.jsonl`
are listed here with the rest rather than taking numbers of their own.

- `config.json` (`config_key`, `experiments.py:116`): written by `create_experiment`,
  `experiments.py:406` (`def create_experiment(`), `overwrite_config_if_pristine`,
  `experiments.py:474` (rewrites only while the record is still pristine, no metrics logged), and
  three best-effort merges the subprocess worker patches into the durable record after a run's
  own resolution is known, each a thin mutator over the one shared terminal-refusing procedure
  `_patch_experiment_config`, same file line 31: `_patch_experiment_config_tiling`,
  `packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py:72`
  (`def _patch_experiment_config_tiling(`), `_patch_experiment_config_id_map`, same file line 90
  (`def _patch_experiment_config_id_map(`), and `_patch_experiment_config_split`, same file line
  113 (`def _patch_experiment_config_split(`). Read by `get_experiment`, `experiments.py:1416`
  (`def get_experiment(`), and `compare_experiments`, `experiments.py:1531`.
- `status.json` (`status_key`, line 140): written by `create_experiment` (397), `update_status`,
  `experiments.py:533` (`def update_status(`), `stamp_run_identity` (`experiments.py:666`),
  `_touch_heartbeat`, `experiments.py:862` (`def _touch_heartbeat(`). Read by `get_experiment`
  (1288), `reconstruct_run_status`, `experiments.py:809` (`def reconstruct_run_status(`),
  `resolve_experiment_dir_for_run` (`experiments.py:690`). `state` is terminal-locked once
  `"completed"`/`"failed"`.
- `lineage.json` (`lineage_key`, line 164): written by `create_experiment` (397), `complete_run`,
  `experiments.py:600` (`def complete_run(`, the run's own `model_weights`/`model_weights_sha256`
  digest, sealed into the transaction that completes the run) and `update_lineage`,
  `experiments.py:1191` (every other field; refuses `model_weights`/`model_weights_sha256` as
  `complete_run`'s alone). Read by `get_experiment` (1288) and `get_experiment_lineage`,
  `experiments.py:1597`.
- `artifacts.json` (`artifacts_key`, line 187): written by `create_experiment` (397),
  `complete_run` (592, the `model_weights` entry: `path`, `sha256`, `recorded`) and
  `record_artifact`, `experiments.py:1163`. Read by `get_experiment` (1288).
- `metrics.jsonl` (`metrics_key`, line 254, append-only): written by
  `log_metrics`, `experiments.py:897`. Read by `read_metrics`, `experiments.py:881`, which
  `get_experiment` (1288, paginated) and `reconstruct_run_status` (770, last row only) go through.
- `env.json` (`env_key`, line 210): the library versions, seed and model kind a run is
  reproducible from, written once by the training envelope,
  `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:355`. No accessor in this module
  reads it back; it is provenance a reviewer reads directly.
- `split.json` (`split_key`, line 232): written by `training_tools._persist_split_manifest`,
  `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:1729`
  (`def _persist_split_manifest(`). Read by `read_split_manifest`,
  `experiments.py:1058`, which `pipelines/block_calibration.py` and `pipelines/operating_point.py`
  both take the manifest from. Every run, bound to a manifest or not, records `date`, the labels
  directory's own capture date, `manifest_date_key`'s empty string for a flat tree (never `null`:
  a selection-disjointness check comparing dates must tell a flat run's own date apart from a
  caller that derived no date to compare at all), so it can scope itself to one date without
  re-deriving it from the config. When the run named a split manifest (`data.split.manifest_dir`),
  this record's `manifest_binding` carries the binding's counts (`assigned`, `train_bound`,
  `val_bound`, `calibration_bound`, `calibration_unadmitted`, `other_dates`) and its two content
  hashes (never the manifest's own member lists, which live in the `split_manifest` record itself,
  see §26); `calibration_bound`
  counts stems the manifest's `calibration` side names for this date, placed on neither loader
  whether or not the run currently admits them, and `calibration_unadmitted` counts the subset the
  run no longer admits at all, so a review of this one record shows the manifest's held-out side
  was honored rather than silently folded into training or selection. Beside `manifest_binding`,
  never inside it, `label_digests` carries the per-stem facts the count-shaped binding block
  deliberately excludes: `at_split` (copied from the manifest's own members block for this date),
  `at_run` (the same per-stem digests recomputed at bind time) and `manifest_sha256` (the digest
  of the manifest record the run bound to), so a calibration can name a label that moved between
  the draw and now without the durable experiment config, a checkpoint's embedded config or a
  trial's resolved config ever carrying a per-stem digest.
- `validations.jsonl` (`validations_key`, line 272, append-only): the claims earned against this
  run's evidence. Written only by the module-private `_append_validation`, `experiments.py:997`
  (no public raw appender; the storage seam's generic append remains reachable and is a stated
  residual), which refuses a row missing any of `_VALIDATION_FIELDS` (`experiments.py:945`),
  `train_disjointness` among them: `{"checked": bool, "group_check": str | None}` for the four
  documents whose gate runs the check, `null` for `resolve_scale`. `selection_disjointness` is the
  parallel field for whether the calibration used to validate a checkpoint's own reference was kept
  disjoint from that checkpoint's own selection (val) side: `{"applicable": bool, "reason": str |
  None, "checked": bool, "unresolvable": bool, "leaked_groups": list, "leaked_stems": list,
  "group_check": str | None, "labels_moved_draw_to_run": list | None, "labels_moved_run_to_now":
  list | None, "calibration_labels_moved": list | None, "manifest_redrawn": bool | None,
  "calibration_labels_dir": str | None}` for the four documents `resolver_selection_disjointness`
  covers, `null` for `resolve_scale`; `applicable` is `False` when no split manifest is in play (no
  `split_manifest_dir` named and no `manifest_binding` on the checkpoint's own run, a within-image
  `spatial_strip` split, an empty `val`, an `external` group_by, no `calibration_date` derived at
  all, or a `calibration_date` other than the run's own `split.json` `date`), each such case
  carrying its own `reason`; `unresolvable` marks
  the one case the ruling refuses rather than skips, a named manifest with no experiment record to
  read a selection side from. `verify_stamp_binding` (line 1399) requires the five label-movement
  keys present, `null` admitted, on an applicable row it would otherwise pass: an applicable,
  checked, no-leak row missing any of them floors, the same as a leak does, so a row earned before
  the keys existed cannot read as cleared. Read by `read_validations`, `experiments.py:1022`,
  `find_validation`, `experiments.py:1037` (matching rows by recomputed `validation_digest`,
  `experiments.py:940`), and included whole by `get_experiment` (1201). The one member appendable
  after a terminal state, because a validation is a statement made about a run after it ended.

Seam S07 ("Experiment record .tcip/experiments/<id>/", covering config/status/lineage/artifacts),
verdict `both-sides-one-implementation`, `phase0_implementation: mixed`:
`tests/test_calibration_holdout_disjointness.py:130`,
`tests/test_training_subprocess_isolation.py:100,330,349,362`. Gap: no test verifies that a
subprocess worker's best-effort config patches respect the terminal-state immutability lock
`overwrite_config_if_pristine` enforces on its own writers.

Seam S08 ("metrics.jsonl row format"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_tcip_web_training_routes.py:63,65`,
`tests/test_provenance_spine.py:143`, `tests/test_model_registry_metrics.py:70`. Gap: no test
writes `metrics.jsonl` through the real `log_metrics` writer and reads it back through the real
metrics route or the training-stream websocket in the same test; the reader-side tests hand-write
rows directly. The route reads the log through the seam and shapes the answer with
`_metrics_common.metrics_response`, so there is no second parse of a row to disagree with.

Seam S30 ("split.json train/val manifest"), verdict `both-sides-one-implementation`,
`phase0_implementation: written twice`: `tests/test_calibration_holdout_disjointness.py:124,305,357`,
`tests/test_block_calibration.py:75`. These tests call the real writer
`_persist_split_manifest` and the real readers (`operating_point.py`'s disjointness check,
`block_calibration.py`'s `resolve_block_calibration_records`) against the same file. Gap: nothing
exercises a mismatch between `dataset_hash`/`dataset_id`/`dataset_fingerprint` values recorded by
the writer and any downstream consumer keying off them. `manifest_binding`, the block a run bound
to a named `split_manifest` record (format 26) carries here, is exercised by
`tests/test_split_manifest_binding.py`'s own writer/reader pair, not by this seam's tests.

## 16. `.tcip/models/registry.json`, trained-model registry

Path: `<project_path>/.tcip/models/registry.json`.

Writer: `_register_entry`, `packages/tcip-mcp/src/tcip_mcp/model_registry.py`, the one write both
registration modes share, replacing one entry by name inside one storage-seam transaction on the
key `registry_index_key` mints so a concurrent registrar's entries are not clobbered.
`ModelRegistry.register_model` wraps it for explicit mode (`experiment_id=None`, its own hash of
`checkpoint_path`); `register_model_from_experiment` (`experiments.py`) calls it directly with the
digest a completed run's own `complete_run` already recorded, and `experiment_id` set to that run.
An entry a run bound this way (`experiment_id` non-null) refuses a replace by anything but that
run (`EntryOwnedByRun`); the retired `experiment:<id>` tag convention is written by nothing any
more. `register_model`/`_register_entry` take `metrics_source` as a required keyword, `"trainer"` /
`"training_source"` / `"caller"` / `None`, naming which path produced `metrics` without claiming
anyone verified it; only the two production callers set it, `register_model_from_experiment`
(reading whether the run's config carries `training_source`) and the `register_model` tool's
explicit mode (`tools/model_tools.py:18`, `"caller"` when `metrics` is non-empty).
`register_model_from_experiment` no longer falls back to the run's `metrics.jsonl` log for a
checkpoint with no metrics dict; such a registration carries `metrics={}` and
`metrics_source=None`, and the load failure that produced it is logged rather than swallowed.
An entry that predates the `metrics_source` key also predates `experiment_id`: conform it with
`scripts/conform_registry_experiment_id.py` first (the eviction rail refuses a pre-`experiment_id`
entry's replace by name), then re-register it through `register_model` to add `metrics_source`.

Readers: `read_registry_index`, `model_registry.py:53`, the read path for anything outside the
module (`scripts/doctor.py:125`, `"metrics_source"`), and the entry-by-entry accessors built on
it: `ModelRegistry.list_models`, line 319; `get_model`, line 325; `best_model`, line 332;
`verify_model`, line 293. `best_model` takes `metric_key` and `higher_is_better` as required
keywords, no default and no name heuristic, and by default ranks only entries whose
`metrics_source` is `"trainer"` (`include_unverified=True` also ranks the rest). The
`select_best_model` tool (`tools/model_tools.py:123`) resolves `higher_is_better` from
`evaluation.HIGHER_IS_BETTER_BY_METRIC` (`pipelines/training/evaluation.py:109`) when the caller
states none, the single declared-direction mapping `resolve_selection_metric`
(`pipelines/training/generic_trainer.py`) also reads for the trainer's own checkpoint selection.

Seam S27 ("Trained-model registry .tcip/models/registry.json"), verdict `one-side-only`,
`phase0_implementation: once, shared`: `tests/test_lifecycle_wiring.py:7`,
`tests/test_model_registry_metrics.py:8,40,54,70`, `tests/test_provenance_spine.py:70,84,94,111`,
`tests/test_tcip_web_results_routes.py:593,605`. Gap: no test registers a real model and then
calls `GET /api/results/models/registered`, or runs an inference launch end to end through the web
route's identity-stamp block, to confirm the GUI-visible entry matches the MCP-registered one; the
only two web-route tests check a 403-confinement case and an empty-registry case.

## 17. Prediction buckets, verdict-guarded prediction directories

Path: `<dataset_root>/predictions/<model_name>/[<date>/]`, via
`tcip_mcp.dataset_layout.prediction_dir`.

Writer: `stage_prediction_shapes`, `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:254`, the
underlying per-image files written via `tcip_annotation.json_io.write_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:703` (format 1's writer).
`resolve_prediction_bucket`, `prediction_buckets.py:223`, resolves a `(dataset_root, model_name,
date)` triple to a writable directory; `resolve_writable_bucket`, `prediction_buckets.py:191`,
redirects to the next free `<model_name>@r2`/`@r3` variant once any image in a bucket has a
recorded review verdict; `BucketHasVerdicts`, `prediction_buckets.py:157`, is raised instead when
`overwrite=True` is requested against a verdicted bucket.

Readers: `bucket_stems`, `prediction_buckets.py:22`, walks each dir through
`tcip_annotation.json_io.prediction_documents`, which excludes every provenance stamp named in
that module's own `SIDECAR_FILENAMES`, so a stamp added for a new measurement dimension is
excluded here too; `verdict_count`, `prediction_buckets.py:144`, delegates
to `tcip_annotation.review_engine.ReviewEngine.verdict_count_for_images` against the store
`review_state_dir_of`, line 33, names.

Seam S29 ("Prediction-bucket immutability"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_prediction_bucket_resolution.py:39,48`,
`tests/test_review_channel.py:287,312,325`, `tests/test_export_predictions_bucket_handling.py:60`,
`tests/test_orthomosaic_tools.py:241-248`, `tests/test_tcip_web_routes.py:898`. Gap: the seam's
fourth named caller, `vision_tools.py:1033` inside `accept_proposals`'s `except BucketHasVerdicts`
block, has no test coverage; every `accept_proposals` test writes into a fresh, verdict-free
bucket.

## 18. `operating_point.json`, prediction-bucket provenance sidecar

Path: `<prediction_bucket_dir>/operating_point.json`, sibling of the bucket's per-image files,
one of the stamp filenames `tcip_annotation.json_io.SIDECAR_FILENAMES`
(`packages/tcip-annotation/src/tcip_annotation/json_io.py:242`) names and every bucket
enumeration excludes (format 17).

Writer: `write_sidecar`, `resolution.py:987`, the one write, under the stamp's own lock;
`update_sidecar`, line 809, is the one merge into an existing stamp, reading and writing inside
one transaction so a promotion cannot drop what the producing run recorded. Both refuse, through
one shared check (`_check_stamp_claim`, line 768), a stamp claiming validation with no
well-formed `validated_by` or no trait, so a producer cannot omit what every reader compares.
The stamp itself is built by `operating_point_stamp`, line 833, which requires every provenance
field of every producer, including the `validated_by` pointer with no default, and takes a
producer's own extras through `**fields`. The producers are the raster
export (stamped at `tools/inference_tools.py:1044`, written at `tools/inference_tools.py:1081`),
the image export (stamped at `tools/inference_tools.py:1380`, written at
`tools/inference_tools.py:1402`), the GUI's inference worker (stamped at
`routes/inference.py:289`, written at `routes/inference.py:331`, after every prediction file it
certifies is on disk), and the two phenology
deliveries (`tools/phenology_tools.py:403,572`). The raster export also records the identity of
the raster the bucket was produced on, `raster_content_identity`, on every run of that regime,
and `claim_scope_validated` when a trait triggered block calibration.

Reader: `read_operating_point_sidecar`, `resolution.py:1138`, which never raises: an unreadable
stamp floors the dimension it describes to unvalidated at every reconciler rather than taking down
a delivery gate. A stamp claiming validation is checked against the record it names by
`verify_stamp_binding`, line 1399, called from inside every reconciler so no delivery door can
reach a validated result without it; a failed binding floors every dimension that stamp carries
and the reason lands in the reconciler's `binding_notes`. The claim a stamp asserts is subset by
`claim_payload`, line 984, the one extractor the minting side and the reading side share. `routes/review.py:884` merges the review promotion's own validation fields in
through `update_sidecar`. `tools/orthomosaic_tools.py`'s `deliver_orthomosaic_plant_counts` reads
`raster_content_identity` back and refuses a delivery whose supplied raster does not match it,
content and georeferencing alike; a bucket recording no identity is refused rather than delivered.
`resolution.reconcile_claim_scope_validity` reads `claim_scope_validated` back for the delivery
gate.

Seam S28 ("operating_point.json prediction-bucket sidecar"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_delivery_gate.py:443,480`,
`tests/test_block_calibration.py:421,550`, `tests/test_detection_measurement_integrity.py:758,954`,
`tests/test_phenology.py:355,363`, `tests/test_phenology_tools.py:104`,
`tests/test_review_validation_affordance.py:181`. `tests/test_operating_point_sidecar_seam.py`
holds the writer and the reader together: the stamp carries the same keys whatever the producer
supplies (line 55), a write round-trips through the real reader (76), a merge is made against
what is stored rather than against a pre-lock read (93), and the declared stamp filenames cover
every declared document (183).

## 19. `.tcip/state/gui.json`, live GUI state snapshot

Path: `<project_root>/.tcip/state/gui.json`, addressed by `gui_snapshot_key`,
`packages/tcip-web/src/tcip_web/state.py:18`.

Writer: `StateStore._flush_sync`, `tcip_web/state.py:246`, debounced 0.5s after `mutate`/`replace`, which
resolves the destination at flush time so a project switch during the debounce window cannot write
one project's snapshot into another's. The store is declared `durable=False`: the snapshot is
rewritten every debounce cycle and losing the last one costs a re-selection, not history.

Reader: `StateStore.load_from_disk`, `tcip_web/state.py:259`.

`StateStore.mutate`, `tcip_web/state.py:30`, validates the merged mutation through `GuiState`
before holding it, raising `GuiMutationInvalid` (`tcip_web/state.py:29`) on a field that does not
validate or a key `GuiState` does not declare; `app.py`'s `_gui_mutation_invalid_handler`,
`packages/tcip-web/src/tcip_web/app.py:176`, answers a route that raises it with 400 and the
validation message rather than the 500 an unhandled `ValueError` would produce.

Seam S10 ("Live GUI state .tcip/state/gui.json"), verdict `both-sides-restated`,
`phase0_implementation: written twice`: `tests/test_active_context.py:33,41,54`,
`tests/test_tcip_web_state.py:20,38`. Gap: no test writes `gui.json` via the real
`StateStore._flush_sync` and then reads that exact file through the real MCP `view_gui_state`
tool; the MCP-side test hand-authors `gui.json` as a plain dict matching `GuiState`'s field names
instead.

## 20. `.tcip/state/project_status.json`, per-project activity pointer

Path: `<project_path>/.tcip/state/project_status.json`, addressed by `project_status_key`,
`packages/tcip-mcp/src/tcip_mcp/project_status.py:61`, on the store `PROJECT_STATUS_STORE`,
`project_status.py:46`; `project_status_path`, `project_status.py:71`, is the same address as a
path for a caller that needs one.

Writers: `record_report`, `project_status.py:137`; `record_retrospective`,
`project_status.py:153`; `record_distillation`, `project_status.py:173`; all via the shared
locked read-modify-write `_update`, `project_status.py:101`.

Reader: `read_project_status`, `project_status.py:77`.

No seam id in `seam-coverage.json`'s 67-entry inventory names `project_status.json`. S09 ("Web
job registries persisted to .tcip/state/<name>.json") names a structurally similar but distinct
store family (inference/tuning job registries), not this file.

## 21. `.tcip/state/review/*.json`, review-verdict shard store

Path: `<state_dir>/review/<sanitized_bucket>/<sanitized_img_name>.json`
(`REVIEW_SHARD_DIRNAME = "review"`, `packages/tcip-annotation/src/tcip_annotation/review_engine.py:84`).
The store is keyed `("bucket", "image")`, the bucket being the prediction bucket's path relative to
its dataset root as `bucket_key_of` spells it (`prediction_buckets.py:71`), folded into one
directory name by `bucket_dirname` (`review_engine.py:113`). A verdict recorded with no prediction
bucket keeps its shard directly under `review/`.
Real-world `state_dir` is `<dataset_root>/.tcip/state`, derived once by
`review_state_dir_of`, `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:110`;
`verdict_count`, `prediction_buckets.py:144`, opens a `ReviewEngine` on that root rather than
composing a state dir of its own.

Writer: `ReviewEngine._save_image`, `review_engine.py:310`, called by `mark_image_reviewed`
(`review_engine.py:356`), `unmark_image_reviewed` (`review_engine.py:398`),
`record_detection_action` (`review_engine.py:665`), `check_image_review_complete`
(`review_engine.py:802`); `save_review_state`, `review_engine.py:330`, flushes every shard.

Readers: `ReviewEngine.load_review_state`, `review_engine.py:286`, which enumerates the store's
keys (`review_engine.py:206`) at construction; `find_reviewed_entry`, `review_engine.py:549`,
and its spatial-hash cache `_build_reviewed_lookup`, `review_engine.py:530`.

Seam S16 ("ReviewEngine shard-store directory"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_review_channel.py:267-325`,
`tests/test_prediction_bucket_resolution.py:39,48`, `tests/test_tcip_web_routes.py:555`.
`review.py` derives its `state_dir` from the dataset root the request states, which is the root
`prediction_buckets.py`'s reader derives from, and both sides are driven across a project root and a
dataset root that are different directories in `tests/test_review_verdict_scope.py:160`: a verdict
recorded through the GUI route lands in the dataset-scoped store the MCP-side bucket-immutability
check counts, and the promotion reads that same store.

## 22. `.tcip/datasets.json`, project-level dataset identity registry

Path: `<project_root>/.tcip/datasets.json`.

Writer: `upsert_dataset`, `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:145`.

Reader: `read_datasets`, `project_tools.py:71`.

Shape: each entry's `path` is relative to `<project_root>` whenever the dataset sits under it by
filesystem identity (`registry_path_for`, `project_tools.py`), the project's own tree becoming
`"."`; absolute for a dataset outside it. Resolved on read through the one accessor,
`dataset_entry_path`, `project_tools.py`, which every consumer of an entry's path calls rather
than reading `path` off the entry directly. A project registered before this row is conformed by
`scripts/conform_dataset_registry_paths.py`, a one-off operator script, never a runtime
migration.

No seam id in `seam-coverage.json`'s 67-entry inventory names `.tcip/datasets.json` (distinct from
`dataset.json`, format 4, which S26 covers).

## 23. Workspace active-project marker (`.active`)

Path: `<workspace_root>/.active`, a workspace-root sibling, not inside `.tcip/`.

Writer: `set_active_project`, `packages/tcip-mcp/src/tcip_mcp/workspace.py:247`.

Readers: `read_active_project`, `workspace.py:158`; `resolve_project_path`, `workspace.py:239`.

Seam S02 ("Workspace root and the .active project marker"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_tcip_web_projects_routes.py:160`,
`tests/test_ingest_images.py:77,82`, `tests/test_set_active_project.py:16`,
`tests/test_agent_fence.py:142`, `tests/test_agent_ritual_hooks.py:41`,
`tests/test_active_context.py:33`. `agent_session_start.py`'s `_resolve_active` reads through
`workspace.py` itself, not a re-implementation:
`packages/tcip-web/src/tcip_web/agent_session_start.py:69` (`from tcip_mcp import workspace`)
imports it lazily inside the function, and
`packages/tcip-web/src/tcip_web/agent_session_start.py:75`
(`workspace.active_project_if_present(create=False)`) reads the marker through the same seam
`set_active_project` writes. Its tests write the marker through `workspace.set_active_project`,
`tests/test_agent_ritual_hooks.py:58`; `test_session_start_hook_runs_as_a_real_subprocess`,
`tests/test_agent_ritual_hooks.py:115`, runs the hook as a real subprocess against a marker
written that way, so the fresh-interpreter claim is measured rather than inferred.

## 24. Formats named but not exhaustively enumerated in phase0

`classifier_operating_point.json`, `ordinal_operating_point.json`, `regression_operating_point.json`
(sibling sidecars to `operating_point.json`, named in `pipelines/operating_point.py`, not opened
for this section); `pipelines/region_completeness.py`'s own digest-writing logic beyond the
`dataset_layout.py` cross-reference already given for format 8; any format defined inside
`pipelines/postprocessing/`, `pipelines/feedback/`, or `pipelines/data/splits.py`'s own materialized
tree beyond the `image_status.json` carry-over covered in format 5 (`split_manifest.json` is
covered in format 26). No seam id covers this placeholder entry since
it names no single format.

## 25. `.tcip/project.json`, per-project record (site)

Path: `<project_path>/.tcip/project.json`, addressed by `project_record_key`,
`packages/tcip-mcp/src/tcip_mcp/project_record.py:68`, on the store `PROJECT_RECORD_STORE`,
`project_record.py:36`; `project_record_path`, `project_record.py:78`, is the same address as a
path for a caller that needs one.

Writer: `record_site`, `project_record.py:169`, a create-only write: an absent record is written,
a present record with the same site is left as is, and a present record with a different or
unreadable site raises. The one deliberate overwrite is `scripts/conform_project_site.py
--replace`.

Readers: `read_record`, `project_record.py:117`; `site_fields`, `project_record.py:230`, is the
one reader every surface (the picker, `inspect_project`, the doctor) calls, and never raises.

No seam id in `seam-coverage.json`'s 67-entry inventory names `project.json`: the record is new.

## 26. `split_manifest.json`, a partition `make_splits` drew

Path: `<output_path>/split_manifest.json`, addressed by `split_manifest_key`,
`packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py:37`, under whatever directory the caller
asked the partition to be written to; no dataset resolver owns this layout.

Writer: `make_splits`, `data_tools.py:449`, when `output_path` is given or `materialize=True`.
The three sides are `splits.SPLIT_NAMES` (`train`, `val`, `calibration`); a manifest write states
all three ratios (`train_ratio`, `val_ratio`, `calibration_ratio`) non-zero, refusing whichever is
zero by name, since a manifest always draws all three sides. The draw refuses, before any write,
when the tree holds fewer foreground groups of `subject` (and `attribute`, when scoped) than the
three sides need at minimum (one each for `train`/`val`, two for `calibration`), counted through a
subject-scoped `count_label_lines` for the minimum pass on every manifest draw, independent of
`stratify_foreground` (which only gates the balancing pass's own foreground signal). The manifest
records `seed`, `group_by` (the resolved policy), `group_key_map` when one was supplied,
`dataset_fingerprint`, `subject`, `attribute` (`null` when none), `id_map` (the
`assign_class_ids` map the draw resolved), `members` (one block per capture date that admitted at
least one stem, keyed through `splits.manifest_date_key` (the date, or `""` for a flat tree),
holding `labels_root`, `images_root`, that date's own `dataset_hash`, and `label_digests` (the
per-stem sha256[:16] over each admitted stem's own label bytes, the same absent-file convention
`dataset_hash` uses); a date the draw searched
but admitted nothing writes no block, so a reader that finds none under a date treats it as one
the manifest never held, not one it holds empty), `splits` (`train`/`val`/`calibration`
identities, `<date>/<stem>`, the bare `<stem>` under a flat tree), `admission_counts` (the summed
`trainable_stems` counts across every date), `calibration_foreground_groups_by_date` and
`realized_ratios` (both described below). The answer (and the persisted manifest
record) also carry `calibration_foreground_groups_by_date`, a count for every date `members`
holds, `0` included, since the floor above is over the whole draw and one date's own calibration
slice can still land short of two foreground groups; the calibration door's own floor is where
that absence bites. Both also carry `realized_ratios`, each side's member share of the draw
actually delivered: at a floor-sized tree the minimum pass can consume every foreground group
before the balancing pass ever sees the caller's fractions, so the delivered shares can diverge
from the ratios asked for. A stats-only call (no `output_path`, no `materialize`) writes no
manifest, admits a zero or non-zero `calibration_ratio` either way, carries neither
`calibration_foreground_groups_by_date` nor `realized_ratios` (both manifest-write-only fields),
but answers `dataset_hashes_by_date` the same way, over whatever labels directories its plain
image/label scan found, plus a single
`dataset_hash` only when that scan found exactly one such directory; over more than one,
`dataset_hash` is `null` rather than one directory's hash blind to the rest.

Readers: `data_tools.read_split_manifest_dir`, `data_tools.py:55`, the one reader a training or
tuning run's `data.split.manifest_dir` resolves through (`training_tools._auto_train_val`) and a
manifest-restricted calibration resolves through (`splits.resolve_manifest_calibration_universe`),
which refuses by name when the record is absent, undecodable, not a mapping, lacks any of `seed`,
`group_by`, `dataset_fingerprint`, `subject`, `attribute`, `id_map`, `members`, `splits`,
`admission_counts` (the tuple `data_tools._SPLIT_MANIFEST_REQUIRED_KEYS`, kept beside the writer's
dict so the two cannot drift), holds a `members` block whose `label_digests` is missing or not a
non-empty mapping (a manifest drawn before the platform recorded per-stem digests binds nothing
here), lacks any name in `splits.SPLIT_NAMES` under `splits`, or whose
sides are not pairwise disjoint; `scripts/plant_aware_group_splits.py` reads no manifest back, it
only writes one through `make_splits`. `bind_manifest_stems` (`splits.py:403`) reads all three
sides for one capture date: a `calibration` member is placed on neither loader, whether or not the
run currently admits it, recorded as `calibration_bound`/`calibration_unadmitted` rather than
refused on; the `train`/`val` refusals (an admitted stem assigned to no side, a member the run no
longer admits, an empty side) are unchanged in shape.

No seam id in `seam-coverage.json`'s inventory names this record: it is new, and
`tests/test_split_manifest_binding.py` calls the real writer and the real consumer
(`_auto_train_val`'s manifest branch) against the same files.

## 27. `cal_holdout_split_lock`, `.tcip/artifacts/cal_holdout_split_<hash>.json`

Path: named for the identity hash it locks rather than a directory of its own, addressed by
`cal_holdout_lock_key`, `packages/tcip-mcp/src/tcip_mcp/pipelines/data/splits.py:934`, under the
scope root the split was drawn over (`cal_holdout_scope_root`).

Writer: `resolve_locked_cal_holdout_split`, `splits.py:1173`, locking on first draw for a given
identity hash; every later call for the same identity answers from the lock unchanged unless
`force_redraw=True`. The record carries `identity_hash`, `calibration`, `holdout`, `group_by`,
`group_key_map`, `seed`, `holdout_ratio`, `split_manifest_dir` (`null` for a whole-directory draw,
the identity hash otherwise being `dataset_hash` over the manifest's own `calibration` side rather
than the whole directory), and `redraw_history` (one entry per draw, each carrying its own
declared policy, including `split_manifest_dir`, and the old/new content hashes). `calibration`
and `holdout` here are the two halves `cal_holdout_split` cuts from whatever universe it is
given: the manifest's own `calibration` side under a manifest-restricted draw, the whole labelled
directory otherwise; the lock's own field names do not change with the source.

Readers: six callers draw a lock through this one function -
`inference_tools._calibrate_operating_point`, `inference_tools.force_redraw_cal_holdout_split`,
`phenology_tools.calibrate_ordinal_regression_operating_point`,
`measurement.scale_calibration.resolve_physical_scale`,
`scripts/calibrate_operating_point.py`, and `feedback.review_calibration.
resolve_operating_point_from_review` - each answering its own identity's lock, so a whole-directory
draw and a manifest-restricted draw over the same directory coexist as two distinct locks.

`scripts/conform_cal_holdout_locks.py` is the one-off operator fix that adds `split_manifest_dir:
null` to every lock (and each `redraw_history` entry) written before this key existed.
`tests/test_conform_cal_holdout_locks.py` covers it against a fixture root; the repo root's own
thirteen pre-existing locks are conformed once, outside the test suite, when this family lands.

No seam id in `seam-coverage.json`'s inventory names this record.

## Formats with a general path-resolution seam but no per-format seam entry above

Seam S14 ("dataset_layout.py as the on-disk path resolver"), verdict `both-sides-restated`,
`phase0_implementation: once, shared`, and seam S15 ("Per-image label filename convention"),
verdict `one-side-only`, `phase0_implementation: written twice`, both bear on how formats 1-8's
paths are derived rather than naming one format's own schema; they are cross-referenced here
rather than assigned to a single numbered format above. S15's gap: the frontend's
`AnnotateTab.tsx` composes the label path as a template literal independently of
`dataset_layout.annotation_path`, and no test cross-checks the frontend's composed string against
the Python resolver's output.


## Seam inventory

Source: `docs/audit/phase0/seams/seam-inventory.md` (67 seams, both endpoints) and
`docs/audit/phase3/seams/B01-adjudication.json` through `B12-adjudication.json` (the
per-seam `single_implementation.verdict` field: `duplicated`, `restated-in-test`, or
`single`). Every file:line citation below names the line the quoted fragment beside it stands
on at HEAD `f943c12d`, which `scripts/check_architecture_citations.py` holds to the tree. Where
a citation carried in the Phase 0 record no longer names the exact line of the symbol it
describes, the corrected HEAD line is given and the drift is noted.

The "Phase 3 verdict" column is `single_implementation.verdict` from the batch B01-B12
adjudication files, a semantic re-check of whether the two sides actually stay in
agreement, distinct from Phase 0's own structural "Implementation" label (`once,
shared` / `written twice` / `mixed`), which described only whether the code was
physically written once or twice, not whether that code enforces agreement in
practice. The two fields disagree for several seams (S14, S17, S19, S20, S26, S32,  <!-- queued: P5-275 unify -->
S33, S40, S59, S60, S66): Phase 0 recorded a single shared implementation at the  <!-- queued: P5-294 unify -->
structural level, which the Phase 3 adjudication accepts only when no second,
unshared restatement of the same fact exists elsewhere, whatever one function is  <!-- queued: P5-293 unify -->
the primary implementation. Both fields are
reported as-is below; the Phase 3 verdict is the one used for the seam count at the
end.

## S01. Platform state root pin (TCIP_PROJECT_ROOT)

Must agree: all three processes resolve `.tcip/` against the same directory.
Side A: `packages/tcip-mcp/src/tcip_mcp/project_paths.py:44` (`ENV_VAR = "TCIP_PROJECT_ROOT"`).
Side B: `packages/tcip-web/src/tcip_web/routes/images.py:168` (`_render_cache_dir` calls `project_paths.resolve_state_or`, the pinned resolver at `project_paths.py:68`, rather than reading the environment variable itself).
Phase 3 verdict: single.

## S02. Workspace root and the .active project marker

Must agree: every process names the same workspace directory and the same active project.
Side A: `packages/tcip-mcp/src/tcip_mcp/workspace.py:33` (`ACTIVE_MARKER = ".active"`).
Side B: `packages/tcip-web/src/tcip_web/agent_session_start.py:75` (`workspace.active_project_if_present(create=False)`, reading the marker through `workspace.py` rather than a loose file the SessionStart hook resolved itself).
Phase 3 verdict: single.

## S03. Backend port discovery file .tcip/state/web_port.txt

Must agree: the MCP process finds the port the web backend actually bound.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:61` (`BACKEND_PORT_STORE`, declared beside the reader because the reader cannot import `tcip_web`; `backend_port_key`, line 65, is the one address, read at line 112).
Side B: `packages/tcip-web/src/tcip_web/__main__.py:43` (`def _write_port_file(port: int) -> None:`, publishing through that same key at line 52, and logging rather than swallowing a failure, since the fallback silently misses an OS-picked port).
Phase 3 verdict: single.

## S04. Panel-event panel vocabulary (VALID_PANELS)  <!-- queued: P5-324 unify -->

Must agree: sender and receiver accept the same set of panel names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:168` (`VALID_PANELS = frozenset(`).
Side B: `packages/tcip-web/src/tcip_web/app.py:36` (`VALID_PANELS,`).
Phase 3 verdict: duplicated.

## S05. Panel event_type vocabulary  <!-- queued: P5-272 unify -->

Must agree: the Python poster, the FastAPI hub, and the browser handler use the same event_type strings.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:760` (`result = post_panel_event("app", "annotate_focus", payload)`).
Side B: `packages/tcip-web/src/tcip_web/app.py:410` (`if event.event_type == PANEL_EVENT_REVIEW_FOCUS:`).
Phase 3 verdict: single. The posted payload carries `active_subject` beside `subject` (`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:914`), the key both readers take (`packages/tcip-web/src/tcip_web/app.py:296`, `frontend/src/lib/annotateFocus.ts:21,54`), held by `tests/test_event_integration.py`'s producer-driven test, which posts the focus tool's own event and asserts the advisory state's `active_subject`.

## S06. Append-only audit log .tcip/audit.jsonl

Must agree: mutations from any process land in the log their scope names, a dataset's own for a record travelling with the data and the platform's otherwise, with the same entry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/audit.py:241` (`def audited(`, taking a declared `scope_arg` naming which tool argument carries the dataset a scoped tool mutates a record of) and `record_event`, line 121, the one emitter for code that is not an `@audited` tool; both address the log through `audit_log_key`, line 84, and differ only in what a failed append means: `record_event` warns through `_write_entry`, line 103, while the decorator refuses, since its append runs after the tool body.
Side B: `packages/tcip-web/src/tcip_web/routes/review.py:90` (`def _audit(scope: str, tool: str, arguments: dict) -> None:`, which calls `record_event` with the scope its event belongs to; `routes/results.py:54` and `routes/inference.py:84` do the same for their own roots).
Phase 3 verdict: single.

## S07. Experiment record .tcip/experiments/<id>/

Must agree: three processes agree on the experiment directory layout and immutability rules for each file.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:59` (`def experiment_dir(` plus the per-member key constructors, the one declaration of the record's path and member set).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py` (config patch goes through `store.transaction(config_key(...))`; every member writer takes its target from the experiments module's accessors).
Phase 3 verdict: single.

## S08. metrics.jsonl row format

Must agree: the writer's row shape is what the reader and the stream consumer expect.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:897` (`def log_metrics(`, the one writer; the trainer and the envelope hand rows to the context's epoch sink instead of opening the file).
Side B: `packages/tcip-web/src/tcip_web/routes/training.py:259` and `routes/tuning.py:286` (each route reads its own log through the seam's `read_log` and answers in the one shape `_metrics_common.metrics_response` builds; the training route's incremental tail reads the same log from a cursor).
Phase 3 verdict: single. An HPO trial with no experiment record still appends to its own trial log, one declared site in the epoch sink, pending the HPO store migration.

## S09. Web job registries persisted to .tcip/state/<name>.json  <!-- queued: P5-325 unify -->

Must agree: a job's own summary is written and reloaded against the root it launched under, not
whatever root this process happens to have pinned when either side runs.
Side A: `packages/tcip-web/src/tcip_web/jobstore.py:81` (`def job_registry_key(`, the one address each registry is written and reloaded through, on the store `JOB_REGISTRY_STORE` declared at `jobstore.py:59` (`JOB_REGISTRY_STORE = "job_registry"`); an explicit `root` composes the key directly, and only its absence falls back to `current_root`, which itself resolves `project_root`).
Side B: `packages/tcip-web/src/tcip_web/routes/inference.py` (inference job registry, calls `jobstore.persist_grouped`/`jobstore.load`).
Phase 3 verdict: duplicated.

## S10. Live GUI state .tcip/state/gui.json  <!-- queued: P5-284 unify -->

Must agree: the MCP agent reading GUI context parses the snapshot the web backend wrote.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:106` (`def gui_snapshot_key(`, the one address, declared beside `GUI_SNAPSHOT_STORE`, line 82; `packages/tcip-web/src/tcip_web/state.py:197` writes through it).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:411` (`gui = tcip_store.read(gui_snapshot_key(project_root), default=None)`, the MCP read through the same key).
Phase 3 verdict: single.

## S11. Live canvas state files canvas_live.json / canvas_shapes.json  <!-- queued: P5-274 unify -->

Must agree: browser payload, backend file writer, and MCP reader agree on the two-file split and the (image_path, tab) identity check.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:142` (`def canvas_meta_key(`, the meta document's one address, with `canvas_geometry_key`, line 144, addressing the geometry document; the two stores are declared as `CANVAS_META_STORE` and `CANVAS_GEOMETRY_STORE`, lines 111 and 112; `packages/tcip-web/src/tcip_web/routes/canvas.py:85` writes meta through the key, geometry first at line 78).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/vision_tools.py:1107` (`meta_doc = canvas_meta_key(root)`, the MCP read through the same keys, geometry at line 1088).
Phase 3 verdict: single.

## S12. Friction reports and retrospectives under .tcip/

Must agree: the GUI reader finds, decodes and orders what the MCP writer produced.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py:184` (`def report_documents(`, the one enumeration, decode and ordering of the friction reports, with `retrospective_documents`, line 211, doing the same for the retrospectives). Both stores are records, enumerated through the seam and ordered by the timestamp each document states (a report's own `timestamp` field, a retrospective's own `## Retrospective:` section headers), never by when the bytes landed. A report is one whole JSON document, not a line of a stream.
Side B: `packages/tcip-web/src/tcip_web/routes/meta.py:40` and `:62` (both routes import those MCP-side enumerators directly and present the rows they return, rather than walking a directory of their own).
Phase 3 verdict: single.

## S13. image_status carried in annotation_stats.json

Resolved: `annotation_stats.json` no longer carries an `image_status` block. Every writer
(`packages/tcip-web/src/tcip_web/routes/sessions.py`'s `_normalized`, `image_event`, `start_session`,
`end_session`) put an empty `{}` there because the shape guard defaulted an absent document to it,
and nothing ever read it back out: the dataset's real confirmed-negative store is
`image_status.json`, addressed by `image_status_key`, the seam this entry once asked to agree
with. The key stopped being written; a project's existing record still carrying it is conformed by
`scripts/drop_annotation_stats_image_status.py`, a one-off operator script.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:813` (`def is_confirmed_negative(`, the one membership predicate; `normalize_status_store` is the one store guard, called by `confirmed_negative_names` and the resolver's confirmations term instead of inline re-implementations).
Side B: `packages/tcip-web/src/tcip_web/routes/classes.py` (`set_image_status`, writing through the registered store).
Phase 3 verdict: single. `packages/tcip-web/src/tcip_web/routes/sessions.py:316` (`if is_confirmed_negative(status):`, session time classification) calls the same predicate against the real `image_status.json` rather than restating it.

## S14. dataset_layout.py as the on-disk path resolver

Must agree: agent writes and GUI reads resolve to the same files.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:129` (`def image_root(`, with `annotation_root`/`prediction_root` and the dated dir calls built on them, plus `bucket_subject_date` at line 475 as the published inverse of `status_bucket`).
Side B: `packages/tcip-web/src/tcip_web/routes/dataset.py` (`select_dataset` resolves every directory through the resolver; `scripts/doctor.py`, `data_tools`, `project_tools` and `annotation_tools` no longer re-spell the tree).
Phase 3 verdict: single.

## S15. Per-image label filename convention

Must agree: the browser's label path and the Python resolver's label path name the same file.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:966` (`def label_filename(`, with `annotation_path`/`prediction_path` built on it).
Side B: `packages/tcip-web/frontend/src/lib/paths.ts:45` (`labelPath`, the browser's one join site over the directories the backend resolves; a gate test pins the record extension against the resolver).
Phase 3 verdict: single. The browser still joins directory plus filename client-side at that one site; handing fully resolved per-image paths across the API would add a backend round trip to image navigation, an open owner question in the batch report.

## S16. ReviewEngine shard-store directory

Must agree: the verdict writer and the bucket-immutability reader look at the same review store.
Side A: `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:110` (`def review_state_dir_of(`, the one derivation of the store root; `verdict_count`, line 105, counts one bucket's verdicts through the store the engine writes into).
Side B: `packages/tcip-annotation/src/tcip_annotation/review_engine.py:171` (`REVIEW_VERDICTS_STORE`, which owns the shard layout inside that root). `packages/tcip-web/src/tcip_web/routes/review.py:72`, `routes/inference.py:428` and `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:1285` open on the derived root instead of composing a state dir each.
Phase 3 verdict: single.

## S17. Canonical per-image annotation JSON schema

Must agree: every writer produces, and every reader accepts, the same name-based record shape.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:409` (`def annotation_from_payload(`, the one conversion from a client payload to a record, with `_annotations_of` at line 227 the one parse back and `write_annotations` at line 336 the one writer).
Side B: `packages/tcip-web/src/tcip_web/routes/annotate.py:191`, `packages/tcip-web/src/tcip_web/routes/review.py:667` and `packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:179` (every save door converts through it rather than assembling records of its own).
Phase 3 verdict: single.

## S18. Bounding-box coordinate convention across the HTTP boundary

Must agree: the browser and the route use corner coordinates while the file uses xywh, with the conversion happening once.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:41` (`bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel`, the wire form).
Side B: `packages/tcip-annotation/src/tcip_annotation/json_io.py:622` (`def xywh(`, the one corner-to-xywh conversion and the 2-decimal grid the stored document lives on, applied on write and, via the import at `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py:42`, to every box scored against a stored label so both sides of a match sit on one grid; a wire box becomes a `BBox` in `annotation_from_payload`, line 175, and `_annotations_of`, line 227, is the inverse read).
Phase 3 verdict: single.

## S19. Annotation format detection scope (json, coco)

Must agree: every reader refuses rather than guesses when a file's format is undetermined.
Side A: `packages/tcip-annotation/src/tcip_annotation/format_io.py:68` (`def detect_format(path: str) -> AnnotFormat:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:114` (`file_fmt = fmt or detect_format(str(gt_path))`).
Phase 3 verdict: single.

## S20. classes.json class registry

Must agree: the GUI editor, the path resolver, and the training loader read one registry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/class_registry.py:4` (`The on-disk registry (`` `<dataset_root>/classes.json` ``) is self-describing and name-based::`).
Side B: `packages/tcip-web/src/tcip_web/routes/classes.py:176` (`from tcip_mcp.dataset_layout import classes_path`).
Phase 3 verdict: single.

## S21. Training name-to-id assignment versus inference decode map

Must agree: a prediction's integer label decodes to the class name the run trained it as.
Side A: `packages/tcip-mcp/src/tcip_mcp/class_registry.py:445` (`def assign_class_ids(`, the one assignment, reached by the loader through `pipelines/data/datasets.py:120`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:125` (`def resolve_decode_id_map(`, the one resolution every door that decodes predictions or reads GT by id calls: `run_inference` at line 822, the raster export at line 1011, the GUI worker at `packages/tcip-web/src/tcip_web/routes/inference.py:281`, and block calibration at `pipelines/block_calibration.py:274`, which hands over the run's own scope rather than restating the prefer-recorded-else-derive rule).
Phase 3 verdict: single.

## S22. image_status.json confirmed-negative store

Must agree: a negative is empty labels plus an explicit human Complete, every consumer applies the same bucket keying and status vocabulary, and each stored status carries the actor who set it and when, so a person's Complete and a status a harvest wrote stay distinguishable.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:790` (`def derive_status(`, with `IMAGE_STATUSES` at line 610 as the one vocabulary, `status_of` at line 542 as the one predicate for what the store holds, and `record_image_statuses`/`replace_image_status_store` as the two declared writers, both through the registered store).
Side B: `packages/tcip-web/src/tcip_web/routes/classes.py` and `routes/review.py` call `derive_status`; the browser imports one `ImageStatus` type from `api/classes.ts`, pinned against the Python vocabulary by a gate test.
Phase 3 verdict: single.

## S23. image_status_digest.json attribute-schema stamp

Must agree: writer and reader compute the digest the same way for a stale stamp to be detectable.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:920` (`def stamp_image_status_digests(`, the one transactional read-merge writer, called by the web route, the materializer, the split tools and the schema-change sweep `class_registry._sweep_schema_change`, which passes `only_unstamped` so a confirmation-time stamp is never re-dated).
Side B: `packages/tcip-mcp/src/tcip_mcp/class_registry.py:162` (`attribute_schema_digest`, the one digest computation).
Phase 3 verdict: single.

## S24. view_coverage.json advisory coverage record

Must agree: backend store shape and the browser's coverage payload match, keyed by status_bucket.
Side A: `packages/tcip-web/src/tcip_web/routes/_coverage_models.py:19` (`class GridGeometry`) through `:101` (`class CoverageRecord`), the one declaration `routes/coverage.py`'s `post_coverage`/`get_coverage` validate against.
Side B: `scripts/generate_frontend_types.py`, rendering `packages/tcip-web/frontend/src/api/types.generated.ts` from Side A; the browser imports the generated types (`lib/coverageTracker.ts`) rather than hand-declaring its own, held current by `tests/test_generated_frontend_types.py` and round-tripped through the real route by `tests/test_coverage_routes.py`'s `test_post_from_a_plain_rgb_view_round_trips`/`test_post_from_a_composite_view_round_trips`.
Phase 3 verdict: single.

## S25. region_completeness.json attestation store

Must agree: an attestation written by the GUI is readable, and staleness-checkable, by the calibration path that relies on it.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:565` (`def region_completeness_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/region_completeness.py:123` (`def stale_cells(`).
Phase 3 verdict: restated-in-test.
Differs from phase0 record: phase0 cited a line inside the function's body rather than its header; the function itself is defined at `region_completeness.py:123` (`def stale_cells(`).

## S26. dataset.json identity and fingerprint

Must agree: the stored fingerprint and the recomputed one cover the same inputs.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:340` (`def dataset_identity_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:814` (`def dataset_fingerprint(dataset_root: str | Path) -> str | None:`).
Phase 3 verdict: single.

## S27. Trained-model registry .tcip/models/registry.json

Must agree: the MCP registrar and the GUI model pickers read one registry entry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/model_registry.py:65` (`def read_registry_index(`, the read path for everything outside the module; `register_model`, line 200, replaces one entry by name inside one `tcip_store.transaction` on the key `registry_index_key`, line 36, mints).
Side B: `packages/tcip-web/src/tcip_web/routes/results.py:1082` (`@router.get("/models/registered")`, serving `model_tools.list_registered_models`) and the browser's one entry declaration, `packages/tcip-web/frontend/src/api/inference.ts:16` (`export interface RegisteredModel {`), held field by field against an entry the real registrar wrote by `tests/test_registry_entry_shape_agreement.py`.
Phase 3 verdict: single.

## S28. operating_point.json prediction-bucket sidecar

Must agree: every writer stamps, and every consumer finds, the same provenance keys next to a bucket's predictions.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:1028` (`def operating_point_stamp(`, the one stamp constructor, every field required of every producer; the stamp's whole key set is declared beside it as `STAMP_KEYS`, the constructor's own, plus `STAMP_EXTENSION_KEYS`, the producer-local additions each named for the producer that writes it; `write_sidecar`, line 790, and `update_sidecar`, line 805, are the only writers, both through `sidecar_key`, line 715, both refusing an unearned validation claim, and for this document refusing a top-level key outside that declared union: a fresh mint on its whole body, an update on the keys it introduces). A validated claim is earned in two phases beside the resolvers it selects among: `open_validation`, line 1159, runs the document's own resolver over the evidence and refuses a result that cleared no accepted reference, and `seal_validation`, line 1265, takes the covered buckets' content identity from the files as they landed, files the row through the experiment record's validations member, and returns the stamp body with its pointer merged in.
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:1138` (`read_operating_point_sidecar`, the one reader, with `verify_stamp_binding`, line 1395, deciding inside every reconciler whether a claiming stamp is answered for). The producers are `tools/inference_tools.py:1044` and `tools/inference_tools.py:1380` and `packages/tcip-web/src/tcip_web/routes/inference.py:289`; the review promotion merges into the stored stamp under its lock at `routes/review.py:884`.
Phase 3 verdict: single.

## S29. Prediction-bucket immutability

Must agree: no writer overwrites a bucket whose predictions already carry human review verdicts.
Side A: `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:191` (`def resolve_writable_bucket(`, the one guard; `bucket_stems`, `prediction_buckets.py:22`, excludes every provenance stamp through `tcip_annotation.json_io.prediction_documents` rather than naming one filename).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:875`, `tools/vision_tools.py:1029`, `tools/inference_tools.py:1268` and `packages/tcip-web/src/tcip_web/routes/inference.py:430` (every writer door resolves through it, against the verdict store `review_state_dir_of` names).
Phase 3 verdict: single.

## S30. split.json train/val manifest

Must agree: the calibration holdout is disjoint from the split the run actually trained on, and,
when a split manifest is in play, from the checkpoint's own selection (val) side too.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:1669` (`def read_split_manifest(`, the one path and parse beside the member's key constructor; the writer persists through the same key).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py` (precheck and resolver share one spatial-strip predicate over that reader) and `pipelines/operating_point.py` (`_train_disjointness` and `_selection_disjointness` both read through it and share `_resolve_group_stem_disjointness`, the one group/stem-overlap implementation).
Phase 3 verdict: single.

## S31. Checkpoint payload structural markers

Must agree: a checkpoint written by the training envelope is kind-routable and rebuildable by the predictor that later loads it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:30` (`MODEL_SOURCE_KEY` and `STATE_DICT_KEY`, the one key vocabulary).
Side B: the three checkpoint writers in `pipelines/training/generic_trainer.py` and the readers (`generic_predictor.py`, `inference/predictor.py`, `training/evaluation.py`) all bind through the constants.
Phase 3 verdict: single.

## S32. Single operating-point resolution for all consumers

Must agree: the same model and images yield the same conf/NMS/max_dets/tile whichever entry door asks for them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py:789` (`def resolve_operating_point(`, the calibrated regime; a caller-supplied `max_dets` earns a derivation label only by naming where it came from, and otherwise records itself as a caller override).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:372` and `:414` (`raw_operating_point` and `block_calibrated_export_operating_point`, the two uncalibrated regimes). Every door takes its bundle from one of the three: `tools/inference_tools.py:304,797,982,1001`, `packages/tcip-web/src/tcip_web/routes/inference.py:234`, `pipelines/training/envelope.py:213`.
Phase 3 verdict: single.

## S33. Shared inference defaults DEFAULT_CONF / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS

Must agree: the MCP door and the GUI door start from the same unresolved defaults, and both read a caller's unstated parameter off the `None` sentinel rather than off equality with the default, so a caller who states the default value is honored as an override instead of being resolved as if they had stated nothing.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:161` (`DEFAULT_CONF = 0.5`, with `DEFAULT_NMS_IOU`, `DEFAULT_OVERLAP` and `DEFAULT_MAX_DETS` declared beside it).
Side B: `packages/tcip-web/src/tcip_web/routes/inference.py:32` (import; the sentinel form is `resolved_iou = DEFAULT_NMS_IOU if payload.iou is None else payload.iou`, line 444) and `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py:44` (the delivery-grade evaluation binds the same constants, so the point a run is selected at starts where the point it ships at does). `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:582` is the same form for `run_inference`, `export_predictions` and `tabulate_counts`, whose cap parameters default to `None`: the shared constant supplies the pass, and the unstated parameter travels to the resolver as unstated so it can derive one from the data.
Phase 3 verdict: single. One value is still spelled as a literal rather than bound: `predict_tiled`'s overlap default, `packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py:422`.

## S34. check_delivery_gate behind every delivery path

Must agree: no delivered result ships an unvalidated parameter without an explicit acknowledgement.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2727` (`def check_delivery_gate(`, judging each dimension against `_DIMENSION_REFERENCES`, line 2165, so a reference clears only the dimension whose kind earned it, with `DeliveryGateResult.column_stamp`, line 2146, as the one derivation of what a deliverable's validity column carries).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2048` (`def delivered_tail(`, the one composition every delivered tail's validity columns route through, deriving `own_column` from which of `_DIMENSION_TO_COLUMN`'s columns a door's own column list carries, never a second list stating so). `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py:323` and `pipelines/postprocessing/aggregation.py:533` (`delivered_tail(provenance, measurement_recon["bindings"], gate,`, each delivery door composing its tail through it rather than stamping the column itself), `tools/inference_tools.py:2294` (`gate.column_stamp("operating_point")`, `tabulate_counts`'s own read of the gate it also hands to `export_detection_csv`), and `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:608` (`delivered_tail(`, inside `_write_phenology_delivery`, the one writer both phenology delivery doors call through, `tools/phenology_tools.py`'s `compute_phenology` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv`, rather than stamping the column themselves). The aggregated per-plant door also floors a claim-scope dimension read from each bucket's sidecar (`resolution.reconcile_claim_scope_validity`, whose accepted values, `CLAIM_SCOPE_REFERENCES`, are narrower than `VALIDATED_SHIPPABLE`, so an annotation reference cannot clear a raster-scope claim): a bucket that records no claim scope never acquires the dimension.
Phase 3 verdict: single.

## S35. ResolvedParam validation firewall

Must agree: a parameter needing validation is un-consumable as a bare number unless checked against the right kind of reference.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:182` (`class ResolvedParam:`; `.value` raises at line 216, and `unvalidated_value`, line 224, is how a door reads an unvalidated number).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:107` (`_ACCEPTED_REFERENCES`, whose geometry entry is built from `_GEOMETRY_REFERENCE_BY_SOURCE`, line 64, the one statement of which tile-size source earns which reference; `tile_size_source_of`, line 84, reads it back the other way).
Phase 3 verdict: single. One read of the raw value survives outside the class, in `scripts/calibrate_operating_point.py:160`'s console line.

## S36. Count-objective vocabulary versus registered pickers

Must agree: every named count objective has a registered picker function.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:75` (`COUNT_OBJECTIVES`, over the three names declared at lines 50-52).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py:66` (`COUNT_OBJECTIVE_PICKERS`, with the reconciling `assert set(COUNT_OBJECTIVE_PICKERS) == COUNT_OBJECTIVES,` at line 70). A picker's provenance label, and the review-verdict variant it earns through `REVIEW_VERDICT_LABEL_SUFFIX`, line 74, are read off that registry by `pipelines/derivations.py:640`, so registering a picker registers its labels.
Phase 3 verdict: single.

## S37. traits.py trait specs against crops.yml controlled vocabulary

Must agree: a registered trait's delivered phenotypes and units exist in the crops.yml vocabulary.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:229` (`def crops_yml_path(`, the one placement of `.github/skills/crops/crops.yml`, loaded once for every reader of it by `_crops_traits`, line 228).
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:300` (`_spec_from_config` cross-checks each spec against `_crops_vocab`, line 241) and `scripts/verify_skill_traits.py:46` (`load_vocab` checks a skill's trait tokens through that same read, and refuses an empty vocabulary rather than reporting a clean skill).
Phase 3 verdict: single.

## S38. Per-project trait spec records .tcip/state/trait_specs/*.json

Must agree: the MCP writer, the loader, and the GUI trait list agree on the spec fields and the reason a spec was skipped.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:347` (`def trait_specs_dir(`, the one placement, with `TRAIT_SPECS_STORE` at line 351 and `trait_spec_key` at line 365 addressing one spec).
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:422` (`load_trait_specs_with_errors`, the one scan and the one skip-reason list) and `:479` (`write_trait_spec_fields`, the one write, reading and merging compare-and-set against the version it read). `packages/tcip-web/src/tcip_web/routes/results.py:443` and `scripts/doctor.py:211` name the project and let the placement resolve here.
Phase 3 verdict: single.

## S39. Phenology CSV column vocabulary

Must agree: the delivered CSV's column names derive from the trait spec on every path that writes them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:81` (`def majority_provisional_column(spec) -> str | None:`, the one owner; `phenology_csv_columns` builds the schema through it).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:618` (`majority_provisional_column(spec)`, called from `_write_phenology_delivery`, the one writer both `tools/phenology_tools.py`'s `compute_phenology` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv` call through instead of assembling the name themselves).
Phase 3 verdict: single.

## S40. Per-band normalization stats for a non-3-channel detector

Must agree: the values passed as image_mean/image_std are per-band stats of the same length as in_chans.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py:457` (`def band_normalization_stats(`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py:65` (`def _normalization(adapter: Any, in_chans: int | None, image_mean, image_std,`).
Phase 3 verdict: single.

## S41. model_source bespoke build seam  <!-- queued: P5-320 unify -->

Must agree: the dict an agent writes into the config carries the keys the builder, the snapshotter, and the predictor all read.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:30` (`MODEL_SOURCE_KEY = "model_source"`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py:108` (`self.model_source = ckpt.get("model_source")`).
Phase 3 verdict: duplicated.

## S42. training_source bespoke train(ctx) seam  <!-- queued: P5-321 unify -->

Must agree: a bespoke train(ctx) callable is importable and accepts the TrainContext the envelope hands it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:424` (`training_source = run.config.get("training_source")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:120` (`training_source = config.get("training_source")`).
Phase 3 verdict: duplicated.

## S43. dataset_source bespoke dataset seam

Must agree: the builder the reader resolves off `data.dataset_source` returns a Dataset the trainer's loaders accept.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:313` (`dataset_source = (config.get("data") or {}).get(DATASET_SOURCE_KEY)`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:1631` (`def build_dataset(task: str, dataset_source: dict | None = None, **kwargs) -> Dataset:`).
Phase 3 verdict: duplicated.

## S44. Model-contract smoke batch versus the trainer's real batch

Must agree: the smoke batch has the same shape the trainer actually feeds model.forward for the task.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py:50` (`def _synth_batch(`, which synthesizes per-sample `(image, target)` items shaped like a dataset's `__getitem__` and hands them to the trainer's own collate).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/generic_trainer.py:491` (`def task_collate(task: str):`, the collate the DataLoader assembles the training batch with).
Phase 3 verdict: single.
Differs from phase0 record: phase0 cited a line inside the function's body rather than its header; the function is defined at `model_contract.py:50` (`def _synth_batch(`).

## S45. Review verdicts promoted into a calibration reference

Must agree: a breeder-confirmed sample reaches the operating-point sweep in the same record shape GT annotations do, and passes the same gate.
Side A: `packages/tcip-annotation/src/tcip_annotation/review_engine.py:665` (`def record_detection_action(`, the one writer of a stored verdict entry).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/verdicts.py:74` (`decode_verdict`, the one read of that entry, over the affirming actions declared at line 18), called by `pipelines/feedback/review_calibration.py:282` for the calibration reference and `pipelines/feedback/materialize.py:85` for the curated dataset. What each consumer then emits from the affirmed box (COCO xywh scaled by the image, pixel corners for a label file) stays its own.
Phase 3 verdict: single.

## S46. Frontend api/ layer against backend route paths

Must agree: every URL the browser builds matches a registered FastAPI route path and method.
Side A: `packages/tcip-web/frontend/src/api/routes.ts` (generated: the browser's only copy of the paths, each named for its method).
Side B: `packages/tcip-web/src/tcip_web/routes/__init__.py` (`register_all` mounts 16 routers with fixed prefixes).
Phase 3 verdict: single. The api/ helpers keep their hand-written signatures and reference a generated name; `scripts/generate_frontend_routes.py` projects the registered routes into that module, and `tests/test_frontend_route_paths.py` fails when the projection is stale or a call site writes a path of its own.

## S47. GuiState shape between state.py and store/types.ts  <!-- queued: P5-287 unify -->

Must agree: the snapshot the backend serializes deserializes into the store's typed shape.
Side A: `packages/tcip-web/src/tcip_web/state.py:98` (`class GuiState(BaseModel):`).
Side B: `packages/tcip-web/frontend/src/store/types.ts:55` (`export interface GuiState {`).
Phase 3 verdict: duplicated.

## S48. State WebSocket snapshot protocol  <!-- queued: P5-288 unify -->

Must agree: the browser knows which slices of a broadcast snapshot are backend-authoritative and orders them by version.
Side A: `packages/tcip-web/src/tcip_web/app.py:220` (`@app.websocket("/ws/state")`).
Side B: `packages/tcip-web/src/tcip_web/state.py:175` (`def version(self) -> int:`, "Monotonic version, bumped on every state change.").
Phase 3 verdict: duplicated.

## S49. Terminal PTY WebSocket protocol  <!-- queued: P5-289 unify -->

Must agree: control-message type names and field names match, and output frames are treated as raw text rather than JSON.
Side A: `packages/tcip-web/src/tcip_web/routes/terminal.py:341` (`@router.websocket("/ws/{session_id}")`).
Side B: `packages/tcip-web/frontend/src/components/TerminalRail.tsx:322` (`send({ type: "input", data });`).
Phase 3 verdict: duplicated.

## S50. Inference job stream WebSocket  <!-- queued: P5-304 unify -->

Must agree: the browser recognizes the terminal frame and the status vocabulary the backend uses.
Side A: `packages/tcip-web/src/tcip_web/routes/inference.py:583` (`@router.websocket("/jobs/{job_id}/stream")`).
Side B: `packages/tcip-web/src/tcip_web/jobstore.py:97` (`TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})`).
Phase 3 verdict: duplicated.

## S51. Training run stream WebSocket  <!-- queued: P5-297 unify -->

Must agree: the status payload the MCP tool returns is renderable by the browser's training view.
Side A: `packages/tcip-web/src/tcip_web/routes/training.py:244` (`@router.websocket("/runs/{run_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py` (`check_training_status` supplies the status payload).
Phase 3 verdict: duplicated.

## S52. Image-serve response headers  <!-- queued: P5-301 unify -->

Must agree: header names and value encodings match.
Side A: `packages/tcip-web/src/tcip_web/routes/images.py:692` (`"X-TCIP-Stats-Source": json.dumps(stats_source.model_dump(), allow_nan=False),`).
Side B: `packages/tcip-web/frontend/src/lib/imageLoader.ts:55` (`const statsSourceRaw = headers.get("X-TCIP-Stats-Source");`).
Phase 3 verdict: duplicated.

## S53. Optimistic-concurrency token for label saves

Must agree: the token the browser echoes is the same token the backend minted for that label file.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:190` (`"base_mtime": token,`, the token the load route mints; the save route compares the echoed one at `routes/annotate.py:200`).
Side B: `packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx:576` (`base_mtime: paths.mtime,`).
Phase 3 verdict: single.

## S54. Built frontend bundle location  <!-- queued: P5-305 unify -->

Must agree: the directory Vite writes is one of the directories the backend looks in.
Side A: `packages/tcip-web/frontend/vite.config.ts:25` (`outDir: "../static",`).
Side B: `packages/tcip-web/src/tcip_web/app.py:279` (`def _find_static_dir() -> Path:`).
Phase 3 verdict: duplicated.

## S55. Vite dev-server proxy prefixes  <!-- queued: P5-306 unify -->

Must agree: every backend path the browser calls in dev falls under a proxied prefix.
Side A: `packages/tcip-web/frontend/vite.config.ts:20` (`proxy: {`).
Side B: `packages/tcip-web/src/tcip_web/app.py:220` (`@app.websocket("/ws/state")`, one of the endpoints not under the `/api` prefix).
Phase 3 verdict: duplicated. The prefix literals still stand on their own, but `tests/test_frontend_route_paths.py` now fails when a path the frontend references falls outside them, sockets under the API prefix included.

## S56. Tab-name vocabulary  <!-- queued: P5-290 unify -->

Must agree: the tab a panel event targets, the tab the browser can restore, and the tab the backend persists are the same set of names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:160` (`ActiveTab = Literal["annotate", "review", "training", "tuning", "inference", "results", "meta"]`, with `TAB_NAMES = get_args(ActiveTab)` beside it, `tcip_web.state` importing both).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts:14` (`export const TAB_NAMES = [`, generated from the same declaration).
Phase 3 verdict: duplicated.

## S57. Review match computation and its response shape

Must agree: the TP/FP/FN classification the browser draws is the one the matching library computed.
Side A: `packages/tcip-annotation/src/tcip_annotation/matching.py` (`compute_matches` / `compute_classified_trait_matches`).
Side B: `packages/tcip-web/src/tcip_web/routes/review.py:378` (`class MatchesResponse(BaseModel):`).
Phase 3 verdict: restated-in-test.

## S58. Reference-grid geometry

Must agree: the cell name the agent points at and the cell the GUI highlights are the same rectangle.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py:57` (`def reference_cells(`, which builds the cells, with `grid_geometry`, line 151, the geometry handed over beside them).
Side B: `packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py:329` (`def grid_to_rect(`, the one cell-name lookup, with `grid_to_pixel`, line 360, built on it) and `packages/tcip-web/src/tcip_web/routes/coverage.py:91` (`@router.get("/grid")`, `get_grid`, whose cell list the browser consumes verbatim).
Phase 3 verdict: single.

## S59. Path confinement (the derived allow-set)

Must agree: every route that accepts a client-supplied path confines it to the same allowed roots.
Side A: `packages/tcip-web/src/tcip_web/paths.py:46` (`def allowed_roots() -> list[Path]:`).
Side B: `packages/tcip-web/src/tcip_web/routes/annotate.py:80` (`p = assert_path_allowed(path)`).
Phase 3 verdict: single.

## S60. WebSocket origin check

Must agree: every WebSocket endpoint applies the same origin policy before accept().
Side A: `packages/tcip-web/src/tcip_web/trust_boundary.py:256` (`def origin_allowed(origin: str | None, scope: Mapping[str, Any]) -> bool:`).
Side B: `packages/tcip-web/src/tcip_web/app.py:223` (`if not origin_allowed(websocket.headers.get("origin"), websocket.scope):`).
Phase 3 verdict: single.

## S61. Bash guard and PowerShell guard protected-path sets

Must agree: the two shells fence the same platform paths.
Side A: `packages/tcip-web/src/tcip_web/agent_bash_guard.py:338` (`kind = fence_rules.classify(target, root=root, mode=mode)`).
Side B: `packages/tcip-web/src/tcip_web/agent_powershell_guard.py:149` (`kind = fence_rules.classify(target, root=root, mode=mode)`).
Phase 3 verdict: single.

## S62. Guard hooks against the fence settings deny list

Must agree: what the tool-level deny list blocks and what the shell guards block cover the same paths.
Side A: `packages/tcip-web/src/tcip_web/agent_terminal.settings.json` (`permissions.deny` lists `Edit(packages/**)` and similar).
Side B: `packages/tcip-web/src/tcip_web/agent_fence_rules.py:165` (`def _declared_targets() -> "tuple[list[str], list[str], list[str]]":`, which splits those deny rules into the sets `classify` fences).
Phase 3 verdict: single.

## S63. Fence settings materialization for the spawned terminal  <!-- queued: P5-328 unify -->

Must agree: the hook command strings in the committed profile resolve to real guard files from whatever cwd the terminal starts in.
Side A: `packages/tcip-web/src/tcip_web/agent_terminal.settings.json` (hook commands are repo-relative).
Side B: `packages/tcip-web/src/tcip_web/terminal.py:78` (`def _materialize_fence_settings() -> Optional[Path]:`).
Phase 3 verdict: duplicated.

## S64. MCP tool registry against documented tool names  <!-- queued: P5-303 unify -->

Must agree: any document naming a tool names one the server actually registers.
Side A: `packages/tcip-mcp/src/tcip_mcp/server.py:37` (`def list_registered_tools() -> list[str]:`).
Side B: `scripts/list_tools.py:15` (`from tcip_mcp.server import list_registered_tools`).
Phase 3 verdict: duplicated.

## S65. MCP client launch configuration  <!-- queued: P5-307 unify -->

Must agree: the environment name in the client config, the docs, and the environment file match.
Side A: `.mcp.json` (launches `conda run -n tcip-agent python -m tcip_mcp`).
Side B: `environment.yml:14` (`name: tcip-agent`).
Phase 3 verdict: duplicated.

## S66. Skill and docstring examples against real signatures

Must agree: a documented call binds against the real function signature.
Side A: `.github/skills/` (python fenced examples in SKILL.md files).
Side B: `scripts/verify_doc_examples.py:33` (`_MD_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)`).
Phase 3 verdict: single.

## S67. Local gate commands against the CI gate  <!-- queued: P5-308 unify -->

Must agree: the checks a contributor runs locally are the checks CI runs.
Side A: `CLAUDE.md` (documents `pytest -n 4`, `ruff`, `mypy`, and the frontend command chain).
Side B: `.github/workflows/ci.yml` (mypy job, python job with `pytest -n auto` and `TCIP_MIN_TESTS`, typescript job with format:check/lint/typecheck/test/build).
Phase 3 verdict: duplicated.

## Totals

67 of 67 seams from the Phase 0 inventory carry a Phase 3 `single_implementation` verdict.
By verdict:

- `duplicated`: 22 seams.
- `single`: 43 seams (S01, S02, S03, S06, S07, S08, S12, S13, S14, S15, S16, S17, S18, S19, S20,
  S21, S22, S23, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S44,
  S45, S46, S53, S58, S59, S60, S61, S62, S66).
- `restated-in-test`: 2 seams (S25, S57).

43 of 67 seams (64%) hold their agreement in a single implementation.
