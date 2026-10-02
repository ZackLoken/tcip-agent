# TCIP architecture and API

This document states what exists at the current commit: what each module owns, the real
dependency graph, the public surface an adopter may rely on, the on-disk formats, and the
cross-layer seam inventory. Its factual sentences are written to be mechanically
checkable: each states one fact a script can verify (a path exists, a symbol is defined
in a file, a count equals N, X imports Y, a route is registered by a file).

Status: draft. A CI check over this document's factual sentences is planned and not
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

Source: the module inventory `tools/build_module_inventory.py` produces, run at HEAD b9100eca.
Every count in this section is read from that regenerated inventory, not from any earlier
snapshot; `tools/check_architecture_doc.py --inventory-json <path>` re-runs the same generator
and cross-checks its counts against this document's tables, this table's own module and line
totals included.

HEAD b9100eca has 386 modules across the six scanned roots (88696 total lines):

| Package (root) | Modules | Lines |
|---|---|---|
| tcip-mcp | 127 | 35982 |
| tcip-annotation | 10 | 2391 |
| tcip-web | 30 | 6093 |
| tcip-store | 13 | 4642 |
| tcip-web-frontend | 187 | 34220 |
| tools | 19 | 5368 |

`tcip-mcp`, `tcip-annotation`, `tcip-web`, and `tcip-store` are the four Python packages under
`packages/`; `tools` is `tools/` at the repo root (not an installed package);
`tcip-web-frontend` is the TypeScript/TSX tree under `packages/tcip-web/frontend/src`. These
counts are exactly the `counts.python_by_root` and `counts.typescript_total` fields of the
generator's JSON output, and this table's own line totals are each root's modules summed the
same way the generator counts them; there is no second definition of a module or a line count
for this section to drift against.

differs from the phase0 record: the Phase 0 inventory (gitignored dev tooling, not shipped)
recorded python_total 149 (tcip-mcp 85, tcip-annotation 11, tcip-web 30, scripts 23) and
typescript_total 123, over the four roots it scanned; `tcip-store` was not one of them. The
regenerated inventory at HEAD 11e29635 had python_total 218 (tcip-mcp 107, tcip-annotation 12,
tcip-web 36, tcip-store 13, scripts 50) and typescript_total 182; the tables below name every
module of every scanned root, and `tools/check_architecture_doc.py` refuses a source file
under a covered root that no row names.

## tcip-mcp

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-mcp/src/tcip_mcp/__init__.py | TCIP MCP Server: domain tools for the phenotyping platform. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/__main__.py | Entry point: ``python -m tcip_mcp``. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/agent_identity.py | Which agent harness this MCP server process is serving, and the session it minted for it. | 0 | 7 |
| packages/tcip-mcp/src/tcip_mcp/assessment.py | An assessment: a run measuring how well one checkpoint, under one execution record, reproduces a held-out reference, at one delivered kind of one trait revision. | 25 | 5 |
| packages/tcip-mcp/src/tcip_mcp/audit.py | The audit log: one append-only store under a dataset root or a project root (:func:`audit_log_key`). | 4 | 33 |
| packages/tcip-mcp/src/tcip_mcp/buckets.py | A prediction bucket: a directory of per-image prediction documents and one ``bucket.json``. | 11 | 20 |
| packages/tcip-mcp/src/tcip_mcp/cli/__init__.py | Operator command sub-package: each module is one ``tcip`` subcommand's implementation, exposing ``main(argv)`` and returning the exit code. | 2 | 11 |
| packages/tcip-mcp/src/tcip_mcp/cli/adopt_store.py | Move a root's existing record and log files into a store database. | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/archive_project.py | Export an annotation project as a portable bundle: a ZIP archive, or, with --output-dir, the identical bundle written as a directory tree. | 3 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/check_dataset_identity.py | Check a dataset's on-disk content against its recorded identity: detect changed / moved data. | 5 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/doctor.py | Data-state doctor: scan a live project for state inconsistencies code audits can't see. | 17 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/export_store.py | Write a root's database-held records and logs back out as files, for a root or for a whole project's roots at once: | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/import_project.py | Import an annotation project from a bundle ``tcip archive-project`` wrote: a ZIP archive, or a directory tree written by its ``--output-dir`` mode. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/inspect_compute_resources.py | Report the host's current compute headroom: CPU, memory, GPU free bytes, and how many training runs the project already has active. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/overlay_reference_grid.py | Render an image with a labeled reference-grid overlay for spatial referencing, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/plant_aware_group_splits.py | Plant-aware group-key derivation for ``draw_splits``, over per-stem georeferenced rasters. | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/preflight_config.py | Validate a training configuration before launching, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/render_failure_cases.py | Find and render the worst predictions for failure analysis. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/scan_dataset.py | Scan a folder for images, labels, and predictions. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/score_predictions.py | Score a published bucket's predictions against on-disk ground truth (COCOeval), from the command line. | 4 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/shp_to_plant_csv.py | Convert a plant-locations shapefile into ``read_plant_csvs``' CSV schema. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/triage_predictions.py | Sort a checkpoint's own predictions by confidence into needs-review and unscoreable queues, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/visualize.py | Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/write_project_site.py | Correct one project's authored site, the deliberate overwrite for a site typed wrong once. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/dataset_layout.py | Canonical dataset-layout resolver: where an image's ground-truth labels and model predictions live on disk. | 13 | 32 |
| packages/tcip-mcp/src/tcip_mcp/delivery.py | Delivering a measurement: the one gate every delivery function calls, the result a breeder's recorded acknowledgment binds to, and the one writer each delivers its CSV and delivery event through. | 12 | 9 |
| packages/tcip-mcp/src/tcip_mcp/experiments.py | A run as a directory: ``<project>/.tcip/experiments/<experiment_id>/``. | 7 | 23 |
| packages/tcip-mcp/src/tcip_mcp/identity.py | The platform's recorded-actor convention, in one place. | 1 | 4 |
| packages/tcip-mcp/src/tcip_mcp/knowledge/__init__.py | The one canonical domain-knowledge directory and its one reader. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/model_registry.py | Model registry: the checkpoints a project can load, and what is known of each. | 6 | 16 |
| packages/tcip-mcp/src/tcip_mcp/operationalization.py | A trait's latest confirmed revision, the one every measurement and delivery reads, and the check of whether its operationalization binds what a delivery door is about to write. | 2 | 12 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/__init__.py | Pipeline sub-package: data, models, training, evaluation, inference, postprocessing. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/__init__.py | Active learning pipeline: scorer and selector modules. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/helpers.py | Active-learning helpers: scorer lookup by method name. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/scorer.py | Active learning scorers: rank unlabeled images by informativeness. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/selector.py | Active learning selector: partition a checkpoint's own predictions. | 0 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/band_stats.py | Display band statistics, the 8-bit stretch every band render goes through, and the RGB composite it stacks into. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py | The reserved-region reference of a within-image split: the band geometry, completeness and feasibility an assessment reads a mosaic's own reserved calibration and test regions through (see ``split_construction.spatial_single_source_split``'s four-way split, ``reserve_calibration_fraction``), for a raster training source too large or too singular to hold whole images out from. | 4 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/__init__.py | Components sub-package: composable ML primitives. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/backbones.py | ``BackboneWrapper``: the interface a backbone must expose to the necks and detectors here. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py | 2D object-detector builders: plain torchvision detector factories. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/heads.py | Task-specific heads: each knows its loss, metric, and output format. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/losses.py | Loss functions for bespoke models: plain importable classes + a name->class map. | 1 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/necks.py | Neck modules: adapt backbone features for downstream heads. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/__init__.py | Data pipeline: dataset loading, augmentation, tiling, splitting. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/augmentations.py | Data augmentation transforms for all task types. | 2 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/band_groups.py | Sensor-agnostic band-group correlation: sibling single-band raster files that are really one logical multi-band capture (some multispectral drone sensors write one file per band instead of one multi-band file per image), and the ``.bandgroup`` manifest that records a found group. | 4 | 13 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/coco_import.py | An external dataset-level COCO document, converted into the dataset's per-image label documents. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py | Whole-dataset content identity: the ``dataset_fingerprint`` formula (labels + image files + registry), recompute-on-read authority for the cached value a dataset's own ``dataset.json`` carries. | 6 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py | Multi-task datasets with standardized interfaces. | 13 | 12 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py | The producer: where a directory of ground truth or a ground-truth table becomes the samples a run trains, evaluates or calibrates over. | 7 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/samplers.py | Task-aware data samplers: class-imbalance handling plus read-locality ordering. | 2 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/selection.py | A selection: which samples train, which validate, which are held back to calibrate on. | 4 | 20 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/split_construction.py | Constructing training splits from a data config, beside ``splits.py``. | 12 | 7 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/splits.py | Group-aware, annotation-stratified train/val/calibration splitting: group-coherent (sibling tiles of one source never straddle two splits), annotation-balanced, deterministic in seed. | 5 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/delivery_events_schema.py | The declared shapes of a delivery event record and of the acknowledgment it may ship under. | 1 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py | Tier-A derivations: compute a parameter (channels, num_classes, anchor ratios) from the artifact in hand. | 7 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/display_bounds.py | Pixel bounds for what the platform serves to a screen or writes as an agent-facing artifact. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py | The execution record a pass runs under, and the pass prepared from a checkpoint and that record. | 7 | 15 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/image_utils.py | Shared image utilities for the composable ML pipeline (channel-aware). | 3 | 30 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/__init__.py | Inference pipeline: model loading and batch prediction. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py | Generic predictor for any bespoke ``model_source`` checkpoint: task read from the saved ``model_source``, prediction over one image, a batch, or a sliced source. | 9 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/predictor.py | The tile geometry a pass runs at, resolved from what a caller states and what a checkpoint records of its own training geometry. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/__init__.py | Measurement primitives: morphology on a validated mask (area / perimeter / centroid / PCA axis extents). | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/mask_geometry.py | Mask-geometry: dimensional measurements on a validated binary/instance mask. | 1 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py | ``build_model``: a run config's ``model_source`` to an ``nn.Module``, by importing the dotted builder it names and calling it. | 3 | 14 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py | The one model-side contract: the measurement boundary, as a behavioral check, not a mold. | 5 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py | The measurement criteria an assessment computes, each reading its tolerances, floors and objective from the trait entry it is handed, and the detector knobs an execution record sets. | 4 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/overviews.py | External overview pyramids (.ovr sidecars) for large rasters. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/pixel_size.py | The resolver from a raster's georeferencing tags to a real-world pixel size in meters (:func:`resolve_pixel_size` and its two wrappers, :func:`raster_pixel_size` and :func:`raster_pixel_size_reason`). | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/__init__.py | Postprocessing pipeline: temporal aggregation and CSV export. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/aggregation.py | Per-plant aggregation, temporal/spatial aggregation of per-image results. | 5 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py | Encoding a pass's predictions as per-image documents, and delivering a bucket's per-image counts. | 9 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/orthomosaic_mapping.py | Georeferencing for a whole-mosaic GeoTIFF: a drone orthomosaic covers many plants in one raster, so mapping a detection to a real-world plant reads the GeoTIFF's own georeferencing tags to turn a pixel location into a real-world coordinate. | 1 | 6 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py | Canonical phenology measurement: a trait's positive-fraction milestones, for whichever registered trait it's computed for. | 10 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/plant_mapping.py | Plant-ID mapping across capture dates, by capture sequence plus GPS: | 14 | 13 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/segment_attribution.py | Per-plant attribution by canopy segment: a detection attributed to a plant by containment in a canopy boundary a person accepted, the segment itself tied to a registry plant by containment of the plant's own projected position. | 5 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/proposal.py | Annotation-proposal engines: a method-neutral seam for auto-labeling. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/raster_source.py | Raster reading: one open-and-read surface for every image source this platform decodes. | 4 | 19 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py | Named reference grid over a raster's native pixel frame: cells recomputed from the serializable geometry dict (:func:`grid_geometry`) by :func:`reference_cells`, each named spreadsheet-style, a bijective base-26 column letter plus a 1-based row number ("B3", ``tcip_annotation.grid``'s ``column_label``). | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/schemas.py | Pydantic v2 config schemas for structural/type validation; the runtime trainer reads the raw config dict. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/slicing.py | Tiled inference over the ``sahi`` library: the slice lattice, a platform checkpoint wrapped as a SAHI detection model, and the one cross-tile merge every tiled path runs. | 4 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/__init__.py | Training pipeline: trainer, progressive unfreezing, HPO. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/collation.py | Collate functions for a task's ``DataLoader``: batches a list of per-sample ``(image, target)`` pairs into the shape ``train()`` and ``evaluate()`` both expect. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py | The training envelope around any training body, the default trainer or an agent's custom ``train(ctx)``, and ``TrainContext``. | 17 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/eval_runners.py | Orchestrates a checkpoint evaluation run (tile-level or delivery-grade full-frame) and returns its scored result; ``evaluation.py`` keeps the metrics computation itself. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py | Task-aware evaluation metrics + composite selection objective: * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point TP/FP/FN), the canonical COCO mAP definition; * in-house scalar metrics for classification / ordinal / regression; * the composite selection objective (lower = better); * a task-agnostic two-pass ``evaluate()``. | 7 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/generic_trainer.py | Task-agnostic training loop for a bespoke ``model_source`` model. | 17 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/hpo.py | HPO, hyperparameter optimization on Ray Tune. | 7 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/optimizer_factory.py | Optimizer factory with differential learning rate support. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/run_registry.py | ``TrainRun``, the state of one run's body in the process running it. | 1 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py | The process entry point of one run's body: ``python -m tcip_mcp.pipelines.training.subprocess_worker --run-dir <dir>``, everything else it reads being in the directory's ``run.json``. | 9 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_guardian.py | Keep one child tied to the life of the process that launched it, on platforms with no job-object equivalent (Linux, macOS). | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_manager.py | TensorBoard process management for training and HPO runs. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/project_paths.py | Paths under a project the caller names, and the repository root this package sits in. | 0 | 14 |
| packages/tcip-mcp/src/tcip_mcp/project_record.py | The project record: the one document every project carries, holding its identity and its site. | 3 | 11 |
| packages/tcip-mcp/src/tcip_mcp/project_status.py | Per-project status pointer: a small, persisted summary of recent activity. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/registry_paths.py | How a record stores a path and where a stored path resolves, the one rule every record under a project writes and reads paths through. | 0 | 16 |
| packages/tcip-mcp/src/tcip_mcp/server.py | MCP server entry point: every domain tool, served on stdio for the project named at start (``--project <path>``). | 24 | 22 |
| packages/tcip-mcp/src/tcip_mcp/store_catalog.py | The whole store catalog in one import: every module that registers a store. | 27 | 6 |
| packages/tcip-mcp/src/tcip_mcp/stray_state.py | What a stray file under a project's ``.tcip/state`` root is, and whether one path may be deleted. | 4 | 2 |
| packages/tcip-mcp/src/tcip_mcp/subject_registry.py | The dataset's subject registry, subjects, their attributes, and the deterministic name→id assignment a training run uses (and records, so predictions stay decodable). | 3 | 14 |
| packages/tcip-mcp/src/tcip_mcp/tools/__init__.py | Tool sub-package: each module registers tools with the MCP server. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py | Annotation tools: load, save and score name-based annotations. | 14 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/bundle.py | What a project bundle holds: every file of a project tree classified (:func:`account_for`). | 11 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/calibration_tools.py | Assessment tools: assess a checkpoint against a held-out reference, against a mosaic's own reserved regions, and a physical scale against a breeder's reference measurements. | 7 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py | Data management tools: census a dataset, split data. | 12 | 5 |
| packages/tcip-mcp/src/tcip_mcp/tools/delivery_tools.py | The general per-plant CSV door, not owned by one trait or delivery kind. | 6 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/experiment_tools.py | Experiment MCP tools: read one run's directory, and list the project's runs. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/feedback_tools.py | Review-queue MCP tools: ``prioritize_review_queue`` and ``triage_predictions``, each skipping the images whose label document marks the named subject finished. | 12 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py | GUI-driving tools: push data to a panel, or drive the live Annotate tab to a frame. | 10 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py | Inference MCP tools: running a checkpoint and publishing its predictions as a bucket, and delivering a bucket's per-image counts. | 14 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/ingest_tools.py | Image ingestion: turn a raw folder of photos into a structured TCIP project. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/knowledge_tools.py | The ``serve_domain_knowledge`` MCP tool: the route to the platform's domain knowledge documents for a client with no skill or instruction-file mechanism of its own. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py | Meta-loop tools for self-improvement. | 5 | 4 |
| packages/tcip-mcp/src/tcip_mcp/tools/model_tools.py | Model management tools, registry, listing, comparison. | 4 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/orthomosaic_tools.py | Orthomosaic MCP tools: per-plant delivery from a published whole-raster prediction bucket plus a registered plant registry. | 13 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/phenology_tools.py | Phenology MCP tools, the agent-facing surface for the per-plant phenology pipeline. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py | Project management tools. | 18 | 10 |
| packages/tcip-mcp/src/tcip_mcp/tools/proposal_tools.py | Proposal-workflow tools: turn a chosen auto-labeling engine's output into predictions for canvas review. | 19 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py | Training MCP tools, config validation, launch training, HPO, status. | 24 | 6 |
| packages/tcip-mcp/src/tcip_mcp/tools/trait_tools.py | The agent-facing door for proposing a trait's entry; the breeder confirms it in the Setup tab. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/vision_tools.py | Vision tools: render annotations and predictions for visual analysis. | 22 | 5 |
| packages/tcip-mcp/src/tcip_mcp/traits.py | A trait: one entry holding its spec fields and the operationalization text for each delivery kind it delivers, kept per project as an appended list of revisions: ``propose_trait`` appends one, ``confirm_revision`` confirms or withdraws one by its number and content hash. | 9 | 26 |
| packages/tcip-mcp/src/tcip_mcp/utils/__init__.py | Shared low-level utilities for tcip-mcp. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/web_client.py | HTTP client for MCP tools to push state to the tcip-web backend (``post_panel_event``), and the declarations of the stores, the GUI state shape and the tab vocabulary (``ActiveTab``) the web package owns. | 8 | 12 |
| packages/tcip-mcp/src/tcip_mcp/workspace.py | The workspace: the folder whose child directories are the projects the GUI lists. | 6 | 10 |

## tcip-annotation

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-annotation/src/tcip_annotation/__init__.py | Headless annotation library: canonical name-based per-image JSON labels, and a COCO reader. | 6 | 4 |
| packages/tcip-annotation/src/tcip_annotation/format_io.py | The reader of an external dataset-level COCO document (an ``"images"`` / ``"annotations"`` / ``"categories"`` key), on its way into per-image documents. | 3 | 2 |
| packages/tcip-annotation/src/tcip_annotation/grid.py | Spreadsheet-style names for reference-grid cells, and the lookup of a named cell's rect. | 0 | 4 |
| packages/tcip-annotation/src/tcip_annotation/json_io.py | Per-image JSON: the on-disk label format of ground truth and predictions alike, one document per image holding every subject's annotations by name. | 3 | 37 |
| packages/tcip-annotation/src/tcip_annotation/mask_contours.py | Mask -> polygon rings: the contour extractor behind every mask-derived shape. | 1 | 3 |
| packages/tcip-annotation/src/tcip_annotation/matching.py | The platform's one matcher of detections to ground truth, and geometry helpers over :class:`~tcip_annotation.state.Annotation` geometries: box IoU as a matrix and polygon containment. | 2 | 8 |
| packages/tcip-annotation/src/tcip_annotation/state.py | The annotation data model: :class:`Annotation` and the geometries it carries. | 0 | 18 |
| packages/tcip-annotation/src/tcip_annotation/utils.py | Shared utilities: image orientation, geometry helpers. | 0 | 3 |
| packages/tcip-annotation/src/tcip_annotation/verdicts.py | The verdict shard: the log of a person's decisions on a prediction bucket's proposals for one image, each entry read through :func:`decode_verdict` alone. | 2 | 5 |
| packages/tcip-annotation/src/tcip_annotation/viz.py | Visualization rendering: draws annotations and predictions on images. | 2 | 2 |

## tcip-store

Counts in this table are import edges inside `packages/tcip-store/src`, counted the same way as every other table here and cross-checked against the regenerated module inventory the same way.

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-store/src/tcip_store/__init__.py | TCIP's storage seam: one interface for the platform's mutable records, logs, and blobs. | 6 | 50 |
| packages/tcip-store/src/tcip_store/adoption.py | Moving a root's existing record and log files into a database, atomically or not at all. | 6 | 3 |
| packages/tcip-store/src/tcip_store/binding.py | Which backend a process binds, decided once at its entry point. | 3 | 14 |
| packages/tcip-store/src/tcip_store/errors.py | Every refusal the storage seam raises. | 1 | 15 |
| packages/tcip-store/src/tcip_store/export.py | Writing one root's database back out as the file layout, and saying when it is stale. | 4 | 2 |
| packages/tcip-store/src/tcip_store/file_backend.py | The filesystem backend: identity to path, atomic replace, file locks, logs, and blobs. | 5 | 27 |
| packages/tcip-store/src/tcip_store/layout_claims.py | Which store could own which path under a root. | 3 | 9 |
| packages/tcip-store/src/tcip_store/model.py | Identity and value types the storage seam speaks, identical on every backend. | 0 | 6 |
| packages/tcip-store/src/tcip_store/registry.py | The store catalog: what each store is, how its values encode, and how it may be written. | 4 | 8 |
| packages/tcip-store/src/tcip_store/schema_version.py | The version-field accept rule every frozen store's reader applies. | 2 | 3 |
| packages/tcip-store/src/tcip_store/sqlite_backend.py | The SQLite backend: one WAL database per root, with blobs left as files. | 6 | 4 |
| packages/tcip-store/src/tcip_store/store.py | The storage seam's public surface: module functions bound to one backend per process. | 3 | 8 |
| packages/tcip-store/src/tcip_store/values.py | What a value must be before a store will carry it, and how a producer says it is not. | 0 | 3 |

## tcip-web

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/src/tcip_web/__init__.py | TCIP Web: FastAPI server for the ML pipeline. | 0 | 0 |
| packages/tcip-web/src/tcip_web/__main__.py | Entry point: ``python -m tcip_web``. | 7 | 0 |
| packages/tcip-web/src/tcip_web/app.py | FastAPI application: the GUI state snapshot and its WebSocket, the panel-event hub, the built frontend and the health probe; every domain route is mounted from ``tcip_web.routes``. | 11 | 4 |
| packages/tcip-web/src/tcip_web/cli/__init__.py | ``tcip``: the operator console command, dispatching to one subcommand per operator command. | 0 | 2 |
| packages/tcip-web/src/tcip_web/cli/__main__.py | Entry point for ``python -m tcip_web.cli``. | 1 | 0 |
| packages/tcip-web/src/tcip_web/cli/distill_learnings.py | Distill worksheet: one project's friction reports and retrospectives, or every workspace project's, printed as Markdown with their recurring themes; writes nothing. | 4 | 0 |
| packages/tcip-web/src/tcip_web/jobstore.py | The web's in-memory async job registry and its memory cap. | 1 | 4 |
| packages/tcip-web/src/tcip_web/label_annotations_cache.py | The content-digest-keyed label-document parse memo shared by every scan of per-image label files. | 1 | 2 |
| packages/tcip-web/src/tcip_web/paths.py | Path confinement for client-supplied paths. | 6 | 10 |
| packages/tcip-web/src/tcip_web/routes/__init__.py | Route modules for the tcip-web FastAPI backend. | 14 | 1 |
| packages/tcip-web/src/tcip_web/routes/_body_common.py | The body model for a state-changing route that carries no fields of its own. | 0 | 4 |
| packages/tcip-web/src/tcip_web/routes/_metrics_common.py | The response shape the tuning trial-metrics route serves. | 0 | 1 |
| packages/tcip-web/src/tcip_web/routes/annotate.py | The Annotate tab's routes: an image's label document read and saved, the proposals a prediction bucket offers for it, and the review queue. | 13 | 2 |
| packages/tcip-web/src/tcip_web/routes/audit_gap.py | The shared shape a GUI route answers with when a mutation it already committed could not be recorded to the audit log. | 1 | 6 |
| packages/tcip-web/src/tcip_web/routes/canvas.py | Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back. | 4 | 1 |
| packages/tcip-web/src/tcip_web/routes/dataset.py | Dataset routes: what a dataset root holds (its dates and subjects, through :mod:`tcip_mcp.dataset_layout`, and its published buckets, through :mod:`tcip_mcp.buckets`), the ``GuiState.dataset`` selection, and the current image position within it. | 8 | 2 |
| packages/tcip-web/src/tcip_web/routes/fs.py | Local-filesystem directory browsing for the frontend's folder picker: any directory the server's user can read, directories only, never files. | 2 | 1 |
| packages/tcip-web/src/tcip_web/routes/images.py | Image serving: the one path pixels reach the browser through. | 10 | 2 |
| packages/tcip-web/src/tcip_web/routes/inference.py | Inference routes: async runs + live progress WebSocket. | 10 | 2 |
| packages/tcip-web/src/tcip_web/routes/meta.py | Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives, enumerated, ordered and decoded by the module that owns their stores. | 2 | 1 |
| packages/tcip-web/src/tcip_web/routes/projects.py | The workspace's projects: listing them, opening one, removing one and renaming one. | 11 | 3 |
| packages/tcip-web/src/tcip_web/routes/results.py | Results routes: plant-mapping, per-plant phenology curves, CSV export, and the traits with the breeder's confirmation of a trait revision. | 18 | 2 |
| packages/tcip-web/src/tcip_web/routes/sessions.py | Session-tracking routes: annotation_stats.json equivalent. | 7 | 1 |
| packages/tcip-web/src/tcip_web/routes/subjects.py | Subject registry routes. | 7 | 1 |
| packages/tcip-web/src/tcip_web/routes/terminal.py | Agent terminal routes: provider status, session launch, restart and submitted requests over HTTP, and per session a WebSocket carrying raw PTY output as text frames out and ``TerminalInputFrame``/``TerminalResizeFrame`` JSON messages in. | 4 | 4 |
| packages/tcip-web/src/tcip_web/routes/training.py | Training routes: launchable configs, launch/relaunch, list runs, live metrics stream. | 11 | 2 |
| packages/tcip-web/src/tcip_web/routes/tuning.py | HPO / Tuning routes: relaunch, cancel, list and per-trial visibility, each read off the sweep's own directory under ``.tcip/hpo`` (``training_tools.sweep_record`` and ``read_sweep``). | 12 | 1 |
| packages/tcip-web/src/tcip_web/state.py | The web backend's own state: the workspace it serves, the project it has open, that project's live :class:`~tcip_mcp.web_client.GuiState` (persisted to the project's ``.tcip/state/gui.json`` on every change) and the panel events it retains for a browser that connects late. | 2 | 17 |
| packages/tcip-web/src/tcip_web/terminal.py | Embedded agent terminal: an agent harness from :data:`PROVIDERS` spawned directly in a pseudo-terminal (ConPTY via ``pywinpty`` on Windows, the stdlib ``pty`` on POSIX), its raw bytes streamed out and keystrokes streamed in. | 4 | 3 |
| packages/tcip-web/src/tcip_web/trust_boundary.py | The network trust boundary: the backend serves connections that arrived through this machine, and answers only to loopback names. | 0 | 1 |

## tcip-web-frontend

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/frontend/src/App.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/App.tsx | (none found) | 28 | 2 |
| packages/tcip-web/frontend/src/api/client.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/api/client.ts | Typed REST client for the tcip-web backend. | 7 | 32 |
| packages/tcip-web/frontend/src/api/devProxy.generated.ts | Dev-server proxy prefixes, generated by tools/generate_frontend_routes.py from the routes the FastAPI app registers. | 0 | 0 |
| packages/tcip-web/frontend/src/api/http.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/http.ts | Shared fetch helpers. | 0 | 28 |
| packages/tcip-web/frontend/src/api/inference.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/inference.ts | Inference + Results API helpers for the Inference and Results tabs. | 4 | 12 |
| packages/tcip-web/frontend/src/api/meta.ts | Meta-loop API helpers: the agent's friction reports and retrospectives. | 2 | 2 |
| packages/tcip-web/frontend/src/api/routes.ts | Every backend path the browser calls, named for its method and its route. | 0 | 10 |
| packages/tcip-web/frontend/src/api/sessions.ts | Session-tracking API helpers (annotation_stats.json on disk). | 2 | 5 |
| packages/tcip-web/frontend/src/api/streams.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/subjects.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/subjects.ts | Dataset subject-registry API helpers. | 3 | 15 |
| packages/tcip-web/frontend/src/api/terminal.ts | REST client for the embedded agent terminal's provider status, launches and requests. | 3 | 2 |
| packages/tcip-web/frontend/src/api/training.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/training.ts | Training-tab specific REST + WebSocket helpers. | 4 | 12 |
| packages/tcip-web/frontend/src/api/tuning.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/tuning.ts | Tuning (HPO) API helpers for the Tuning tab. | 3 | 3 |
| packages/tcip-web/frontend/src/api/types.generated.ts | Types generated by tools/generate_frontend_types.py from the pydantic models that declare them (routes/images.py, routes/training.py, routes/terminal.py, routes/projects.py, routes/results.py, tcip_mcp.traits, tcip_mcp.web_client.GuiState), plus a handful of runtime constants (routes/images.py, tcip_mcp.web_client, tcip_mcp.experiments, tcip_web.jobstore). | 0 | 25 |
| packages/tcip-web/frontend/src/api/ws.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/ws.ts | WebSocket client that subscribes to GuiState snapshots + panel events. | 5 | 4 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.test.tsx | (none found) | 7 | 0 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.tsx | Annotate-tab context toolbar. | 13 | 2 |
| packages/tcip-web/frontend/src/components/BandPicker.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/BandPicker.tsx | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.tsx | The Annotate canvas's Konva Stage wrapper, its pan + zoom state managed in the store. | 6 | 3 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.ts | Discrete zoom levels (5% .. | 0 | 3 |
| packages/tcip-web/frontend/src/components/CollapsibleSection.tsx | The app's collapsible-section primitive: one chevron glyph and one trigger+content unit. | 1 | 6 |
| packages/tcip-web/frontend/src/components/ColorPickerModal.tsx | Dark color picker: SI palette + basic palette + hex input, resolving to a hex string. | 0 | 1 |
| packages/tcip-web/frontend/src/components/ConfirmDialog.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/ConfirmDialog.tsx | A modal dialog shell for a destructive or otherwise consequential confirmation: a real focus trap (Tab/Shift+Tab stay inside), focus moved to the first control on open and returned to the button that opened it on close, Escape closes without confirming, and no backdrop click dismisses it (a destructive dialog must not close on a stray click). | 0 | 2 |
| packages/tcip-web/frontend/src/components/DeliveryEventsPanel.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/DeliveryEventsPanel.tsx | What has shipped from this project: one row per completed delivery, read-only. | 1 | 2 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.tsx | A titled chrome bar over an iframe, for the tools the platform runs beside the app (TensorBoard, Ray's dashboard). | 0 | 3 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.tsx | (none found) | 0 | 2 |
| packages/tcip-web/frontend/src/components/HaloLabel.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/HaloLabel.tsx | (none found) | 0 | 4 |
| packages/tcip-web/frontend/src/components/HelpOverlay.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/HelpOverlay.tsx | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/components/LaunchPicker.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/LaunchPicker.tsx | The config-picker launch surface, shared by the Training and Tuning tabs' headers: a list of rows read from records (never typed by the breeder) plus the agent request composer that remains reachable from both. | 1 | 3 |
| packages/tcip-web/frontend/src/components/ProjectBreadcrumb.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/ProjectBreadcrumb.tsx | Status-bar project breadcrumb: three fast-tracks in the lower-right corner: project name → a dropdown of recent projects (jump straight in), date → a dropdown of this project's dates (switch without the workspace), Switch Project → the full workspace (all projects). | 5 | 2 |
| packages/tcip-web/frontend/src/components/ProjectPicker.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/ProjectPicker.tsx | The front door. | 7 | 2 |
| packages/tcip-web/frontend/src/components/RunComparison.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/RunComparison.tsx | The Training tab's side-by-side run comparison: RunComparison and its cell helpers. | 5 | 2 |
| packages/tcip-web/frontend/src/components/SeasonRail.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/SeasonRail.tsx | Season rail: the app's signature. | 0 | 2 |
| packages/tcip-web/frontend/src/components/StatusBar.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/StatusBar.tsx | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/components/TabBanner.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/TabBanner.tsx | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/components/TabHeading.tsx | (none found) | 2 | 7 |
| packages/tcip-web/frontend/src/components/TerminalRail.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/TerminalRail.tsx | The agent rail: a real agent harness from the backend's provider table, embedded. | 4 | 2 |
| packages/tcip-web/frontend/src/components/Toasts.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/Toasts.tsx | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/components/TopBar.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/TopBar.tsx | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/components/TraitRevisionPanel.tsx | The breeder's confirmation surface for trait revisions: every trait's revision list, the entry of the revision shown, and confirm or withdraw on that revision. | 6 | 1 |
| packages/tcip-web/frontend/src/components/annotate/AnnotateLegend.tsx | (none found) | 4 | 1 |
| packages/tcip-web/frontend/src/components/annotate/AnnotationShapes.tsx | (none found) | 9 | 1 |
| packages/tcip-web/frontend/src/components/annotate/AttributeEditors.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/annotate/AttributeEditors.tsx | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/components/annotate/AttributePanel.test.tsx | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/components/annotate/AttributePanel.tsx | (none found) | 5 | 2 |
| packages/tcip-web/frontend/src/components/annotate/BoxOverlay.tsx | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/components/annotate/InProgressPolygon.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/components/annotate/PointOverlay.tsx | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/components/annotate/PolygonOverlay.tsx | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/components/annotate/ProposalShapes.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/annotate/ProposalShapes.tsx | (none found) | 5 | 2 |
| packages/tcip-web/frontend/src/components/annotate/SnapIndicator.tsx | (none found) | 1 | 1 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.ts | Mirror the active tab into the backend GUI state so view_gui_state reports the tab the human actually sees. | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.ts | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useDisclosure.ts | Open/closed state for a collapsible region, optionally remembered across sessions. | 0 | 3 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.ts | A staged agent request that follows the dataset selection until the breeder edits it. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.ts | Retries an embedded-tool launch attempt on a timer until it settles, the one polling shape the Training tab's run TensorBoard panel and the Tuning tab's sweep TensorBoard panel both need instead of each keeping its own copy of the same loop. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useImageBands.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageBands.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/hooks/useImageNav.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageNav.ts | Single source of truth for image navigation: arrow keys and the TopBar Prev/Next + jump counter all step through the dataset's image list in its own order through this hook. | 2 | 3 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.ts | (none found) | 0 | 3 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/hooks/usePrefetchAdjacentImages.ts | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/hooks/useRegionServes.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useRegionServes.ts | The cell-aligned region serves the current viewport needs when the user zooms past the base bitmap's resolution on a large raster. | 7 | 2 |
| packages/tcip-web/frontend/src/hooks/useServingGrid.ts | The region-serving grid over the open raster, fetched once per image from the route that serves the raster; cells always come from the route, nothing is derived client-side. | 2 | 1 |
| packages/tcip-web/frontend/src/index.css.test.ts | Compiles index.css through PostCSS + Tailwind (same pipeline as the real build) and asserts the keyboard focus-visible ring rules exist on the shared component classes: a typo'd token or dropped @apply utility fails here instead of silently shipping invisible focus. | 1 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.ts | Drive the Annotate tab to a specific (subject, date, image, mode, proposal bucket) in response to the agent's `annotate_focus` event. | 4 | 2 |
| packages/tcip-web/frontend/src/lib/annotateKeys.ts | The Annotate tab's key bindings, declared once: the tab binds each by its `keys` and the help overlay lists each by its `label` and `desc`. | 0 | 3 |
| packages/tcip-web/frontend/src/lib/authorshipSymbology.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/authorshipSymbology.ts | The Annotate canvas' dash symbology, one table shared by the box, polygon and point overlays so a "derived" pattern (a polygon's own read-only bounding box) and a "tool" pattern (a shape a tool drew that no person has accepted) never drift apart between them. | 0 | 6 |
| packages/tcip-web/frontend/src/lib/bandSelection.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/bandSelection.ts | (none found) | 2 | 8 |
| packages/tcip-web/frontend/src/lib/canvasSync.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/canvasSync.ts | Live canvas-state sync: lets the agent see exactly what the canvas shows. | 4 | 7 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.ts | Stop the browser's own ctrl+wheel page zoom over the app, so the canvas' zoom is the only zoom. | 0 | 2 |
| packages/tcip-web/frontend/src/lib/datasetUiState.ts | Per-(project, dataset, date, subject/bucket) UI state in sessionStorage, so switching and returning within a session lands where you were. | 2 | 3 |
| packages/tcip-web/frontend/src/lib/editGeometry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/editGeometry.ts | Pure geometry for in-place box/polygon editing. | 1 | 2 |
| packages/tcip-web/frontend/src/lib/glyphs.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/glyphs.ts | The one glyph a select or a value render shows for "no value chosen" or "the record carries none": a colon, never an em dash or a hyphen. | 0 | 9 |
| packages/tcip-web/frontend/src/lib/imageLoader.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/imageLoader.ts | Shared image loader for /api/images serves. | 1 | 6 |
| packages/tcip-web/frontend/src/lib/joinRunSeries.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/joinRunSeries.ts | Overlay-chart helper for the Training tab's run comparison (kept out of the .tsx so it's unit-testable). | 2 | 2 |
| packages/tcip-web/frontend/src/lib/labelProblemToast.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/lib/labelSerde.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/labelSerde.ts | The single mapping between the unified name-based label file (one Annotation list per image) and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). | 1 | 6 |
| packages/tcip-web/frontend/src/lib/openProject.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/openProject.ts | Opening a workspace project: making it the backend's open project, then pointing the GUI at a dataset inside it (the project's own tree) via /dataset/select. | 5 | 3 |
| packages/tcip-web/frontend/src/lib/paths.ts | The paths the browser reads off the dataset selection. | 1 | 3 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.ts | Pure hit-testing and structural-edit helpers for the annotate canvas' polygon geometry. | 1 | 7 |
| packages/tcip-web/frontend/src/lib/recentProjects.ts | The ids of the last few projects the user opened, most recent first, for the status-bar fast-track; each is named through the current project listing. | 0 | 3 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.ts | A reconnecting WebSocket: capped exponential backoff on an unexpected close, at most one attempt open or connecting, a replaced socket's late events ignored, and a restartable start/stop pair. | 0 | 5 |
| packages/tcip-web/frontend/src/lib/registrySave.ts | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/lib/runStatus.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/runStatus.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/lib/servingGrid.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/servingGrid.ts | Pure helpers over the region-serving grid a raster's route serves (GET /api/images/serving_grid); nothing here re-derives cells from the geometry. | 2 | 2 |
| packages/tcip-web/frontend/src/lib/subjectColors.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/subjectColors.ts | (none found) | 0 | 6 |
| packages/tcip-web/frontend/src/lib/tabLabels.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/lib/toolMode.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/toolMode.ts | Cycling the Annotate toolbar's drawing tool (its `m` shortcut and any other stepper). | 1 | 2 |
| packages/tcip-web/frontend/src/lib/viewGeometry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/viewGeometry.ts | Shared view math for the canvas: fit the view to a pixel rect and clamp pan offsets. | 2 | 4 |
| packages/tcip-web/frontend/src/main.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/appState.ts | (none found) | 10 | 11 |
| packages/tcip-web/frontend/src/store/exportSurface.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/guiStateShape.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/store/index.ts | (none found) | 12 | 58 |
| packages/tcip-web/frontend/src/store/mergeSnapshot.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/store/slices/agentActivity.ts | (none found) | 1 | 4 |
| packages/tcip-web/frontend/src/store/slices/bandSelection.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/store/slices/banners.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/canvas.ts | (none found) | 3 | 3 |
| packages/tcip-web/frontend/src/store/slices/gui.ts | (none found) | 4 | 9 |
| packages/tcip-web/frontend/src/store/slices/pendingTerminalMessage.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/registryStatus.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/store/slices/terminalOpen.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/toasts.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/user.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/store.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/store/tabRestore.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/store/terminalOpenPolicy.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/types.ts | The GUI state's shape is generated from the backend's own model (types.generated.ts). | 1 | 39 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.test.tsx | (none found) | 7 | 0 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx | (none found) | 32 | 2 |
| packages/tcip-web/frontend/src/tabs/InferenceTab.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/tabs/InferenceTab.tsx | (none found) | 5 | 2 |
| packages/tcip-web/frontend/src/tabs/MetaTab.test.tsx | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/tabs/MetaTab.tsx | (none found) | 7 | 2 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.tsx | (none found) | 10 | 2 |
| packages/tcip-web/frontend/src/tabs/RunMonitorLayout.tsx | The shell the Training and Tuning tabs share: a fixed-width scrolling sidebar of runs beside a detail region. | 0 | 2 |
| packages/tcip-web/frontend/src/tabs/SetupTab.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/tabs/SetupTab.tsx | (none found) | 6 | 2 |
| packages/tcip-web/frontend/src/tabs/TrainingTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/TrainingTab.tsx | (none found) | 19 | 2 |
| packages/tcip-web/frontend/src/tabs/TuningTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/TuningTab.tsx | (none found) | 14 | 2 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.ts | Plain-language requests the run tabs stage for the agent, editable before they're sent. | 0 | 3 |
| packages/tcip-web/frontend/src/tabs/chartTheme.ts | Recharts takes literal color strings (not Tailwind classes), so the field-station tokens are mirrored here as hex. | 0 | 3 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.ts | Metric-stream helpers for the Training tab (kept out of the .tsx so they're unit-testable). | 1 | 7 |
| packages/tcip-web/frontend/src/test/setup.ts | Extends Vitest's `expect` with jest-dom matchers (toBeInTheDocument, etc.) and registers automatic cleanup after each test. | 0 | 0 |
| packages/tcip-web/frontend/src/test/traitRecords.ts | (none found) | 1 | 2 |

## tools

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| tools/build_module_inventory.py | Builds a module inventory and real import graph for the repo's Python and TypeScript source trees. | 0 | 0 |
| tools/check_architecture_citations.py | Verify ARCHITECTURE.md's file:line citations against the code they quote, for CI. | 0 | 0 |
| tools/check_architecture_doc.py | Verify ARCHITECTURE.md's module-ownership tables against the tree, for CI. | 0 | 0 |
| tools/cross_family_ask.py | Pose one identical question to several agent harnesses and record comparable answers. | 0 | 0 |
| tools/gate_baseline.py | Run the quality gate CI actually declares, so a local pass means CI would pass too. | 0 | 0 |
| tools/generate_frontend_routes.py | Generate the browser's route-path module and the dev server's proxy from the backend's registered routes. | 1 | 0 |
| tools/generate_frontend_types.py | Generate the browser's backend-declared types from the pydantic models that declare them. | 14 | 0 |
| tools/generate_frozen_manifest.py | Generate frozen-formats.json, the shipped freeze commitment, from the store registry. | 2 | 0 |
| tools/generate_harness_discovery.py | Generate the files that project the domain-knowledge documents (`tcip_mcp.knowledge.list_documents()`): a `SKILL.md` per document under `.claude/skills/` and `.agents/skills/`, the generated block of `AGENTS.md`, and the `WebFetch(domain:...)` grants of Claude Code's row settings file from the hosts the `cv-research` document's source list names. | 2 | 0 |
| tools/generate_trait_fixture.py | Write the trait listings the frontend tests read, produced by the platform itself. | 6 | 0 |
| tools/line_delta.py | Sum a change's insertions and deletions by area: package code, tests, everything else. | 0 | 0 |
| tools/list_store_consumers.py | List every registered store's writers and readers, from the import graph plus a symbol scan. | 2 | 0 |
| tools/list_tools.py | Print the live MCP tool registry (count + names). | 1 | 0 |
| tools/prove_test_fails_before.py | Prove a test actually fails against the code it was written to catch. | 0 | 0 |
| tools/serve_capture_app.py | Start or stop the served web app under a scratch environment, for a GUI capture harness. | 0 | 0 |
| tools/smoke_terminal_e2e.py | Live smoke of one provider row's real harness through the embedded agent terminal. | 4 | 0 |
| tools/verify_skill_tools.py | Guardrail: hold every tool name in agent-facing prose to the registry. | 3 | 0 |
| tools/verify_skill_traits.py | Guardrail: flag every trait-like token in a crop/domain knowledge document that is not in crops.yml. | 1 | 0 |
| tools/worktree_gate.py | Run ruff, mypy and pytest inside a worktree, its own editable-install resolution proven first. | 0 | 0 |

## Package-level dependency rules holding at HEAD 10d9eae4

The following sentences are checked against every in-repo Python import edge in the regenerated module inventory (an edge is counted only when both the importing file and the imported file resolve to a file inside this repo; stdlib and third-party imports are excluded by `build_module_inventory.py`, see docstring at `tools/build_module_inventory.py:9-20`).

- No module under `packages/tcip-annotation` imports from `tcip-mcp`.
- No module under `packages/tcip-annotation` imports from `tcip-web`.
- No module under `packages/tcip-annotation` imports from `tools`.

Non-zero cross-package edge counts at HEAD:

- `tools` -> `tcip-mcp`: 20 import edges.
- `tools` -> `tcip-web`: 18 import edges.
- `tcip-mcp` -> `tcip-annotation`: 76 import edges.
- `tcip-web` -> `tcip-annotation`: 13 import edges.
- `tcip-web` -> `tcip-mcp`: 119 import edges, one of them `routes/projects.py`'s own
  `_job_conflict` importing `tcip_mcp.experiments`, the edge the job-registry walk carries.
- `tcip-annotation` -> `tcip-store`: 6, `tcip-mcp` -> `tcip-store`: 136, `tcip-web` -> `tcip-store`: 25,
  `tools` -> `tcip-store`: 6 import edges, the seam every package reads its records through.

Every other ordered pair of roots holds zero import edges at HEAD; two of them are stated, since
each was once non-zero. `tools` -> `tcip-annotation` is zero. `tcip-mcp` ->
`tcip-web` is zero: the inventory walks the whole AST, so a function-body import counts;
`workspace.remove_project` takes the requesting identity as a string and the route checks the
job registries before calling it, so the package holds no edge into the layer above it.

`packages/tcip-web/frontend/src` (`tcip-web-frontend`) has zero in-repo import edges to any Python module in any of the five Python roots: `build_module_inventory.py` resolves a TypeScript specifier only against a relative path or the `@/` alias into `packages/tcip-web/frontend/src` itself (`tools/build_module_inventory.py:307-327`), so no specifier in the frontend source tree can resolve to a file outside that tree.

## Modules with zero importers (134)

A module counts as zero-importer when no other module in its own scanned tree resolves an in-repo import to it (`imported_by_count == 0` in the regenerated inventory). This includes package entry points (`__init__.py`, `__main__.py`), CLI scripts under `tools/` invoked as processes, package `cli/` command modules invoked by name through the `tcip` dispatcher rather than imported, and every TypeScript `*.test.ts`/`*.test.tsx` file, none of which are expected to have an in-repo importer.

| Root | Module path |
|---|---|
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__main__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/adopt_store.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/archive_project.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/check_dataset_identity.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/doctor.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/export_store.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/import_project.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/inspect_compute_resources.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/overlay_reference_grid.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/plant_aware_group_splits.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/preflight_config.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/render_failure_cases.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/scan_dataset.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/score_predictions.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/shp_to_plant_csv.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/triage_predictions.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/visualize.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/write_project_site.py |
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
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_guardian.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/tools/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/utils/__init__.py |
| tcip-web | packages/tcip-web/src/tcip_web/__init__.py |
| tcip-web | packages/tcip-web/src/tcip_web/__main__.py |
| tcip-web | packages/tcip-web/src/tcip_web/cli/__main__.py |
| tcip-web | packages/tcip-web/src/tcip_web/cli/distill_learnings.py |
| tcip-web-frontend | packages/tcip-web/frontend/src/App.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/client.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/devProxy.generated.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/http.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/inference.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/streams.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/subjects.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/training.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/tuning.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/api/ws.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/AnnotateToolbar.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/BandPicker.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/CanvasStage.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/zoom.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ConfirmDialog.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/DeliveryEventsPanel.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/EmbeddedTool.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ErrorBoundary.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/HaloLabel.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/HelpOverlay.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/LaunchPicker.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ProjectBreadcrumb.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/ProjectPicker.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/RunComparison.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/SeasonRail.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/StatusBar.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/TabBanner.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/TerminalRail.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Toasts.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/TopBar.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/annotate/AttributeEditors.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/annotate/AttributePanel.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/annotate/ProposalShapes.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageBands.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageNav.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useRegionServes.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/index.css.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/annotateFocus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/authorshipSymbology.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/bandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/canvasSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/editGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/glyphs.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/imageLoader.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/joinRunSeries.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/labelSerde.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/openProject.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/runStatus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/servingGrid.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/subjectColors.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/toolMode.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/viewGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/main.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/exportSurface.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/guiStateShape.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/mergeSnapshot.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/store.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/tabRestore.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/store/terminalOpenPolicy.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/AnnotateTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/InferenceTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/MetaTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/ResultsTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/SetupTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/TrainingTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/TuningTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/test/setup.ts |
| tools | tools/build_module_inventory.py |
| tools | tools/check_architecture_citations.py |
| tools | tools/check_architecture_doc.py |
| tools | tools/cross_family_ask.py |
| tools | tools/gate_baseline.py |
| tools | tools/generate_frontend_routes.py |
| tools | tools/generate_frontend_types.py |
| tools | tools/generate_frozen_manifest.py |
| tools | tools/generate_harness_discovery.py |
| tools | tools/generate_trait_fixture.py |
| tools | tools/line_delta.py |
| tools | tools/list_store_consumers.py |
| tools | tools/list_tools.py |
| tools | tools/prove_test_fails_before.py |
| tools | tools/serve_capture_app.py |
| tools | tools/smoke_terminal_e2e.py |
| tools | tools/verify_skill_tools.py |
| tools | tools/verify_skill_traits.py |
| tools | tools/worktree_gate.py |


## Public surface

This section is maintained incrementally rather than regenerated wholesale, and held to the
tree by `tools/check_architecture_doc.py` and `tools/check_architecture_citations.py`.

## 1. MCP tools

`packages/tcip-mcp/src/tcip_mcp/server.py` builds the server (`build_server`) for the project it
was started for. `python tools/list_tools.py` is the authority for the registered tool list and
count; a count written here drifts, so none is. Every registered tool's name matches a `@tool()`
decorator site in `packages/tcip-mcp/src/tcip_mcp/tools/*.py`. A mutating tool's site is
followed by an `@audited` decorator (some spelled `@audited(scope_arg=...)`) unless its library
records each of its acts itself, and a tool that only reads need carry none. The "audited" column
says which: `yes` for the decorator, `library` for a line its library operation writes.

Tables below group by defining module. Column "line" is the `def`/`async def` line.
Docstring is the function's docstring first line, verbatim.

### annotation_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `save_annotations` | `annotation_tools.py:69` | library | Write an image's annotations to its single per-image label file (all subjects, one file). |
| `write_subject_registry` | `annotation_tools.py:289` | library | Author the dataset's nested subject registry, a thin wrapper over ``subject_registry``. |

### data_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `freeze_selection` | `data_tools.py:18` | yes | Freeze a finished run's own drawn train/val partition into a selection, so a later run can |
| `draw_splits` | `data_tools.py:228` | yes | Compute a leakage-free, annotation-stratified train/val/calibration split. |

### experiment_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `get_experiment` | `experiment_tools.py:11` | no | Read one run's directory. |
| `list_experiments` | `experiment_tools.py:47` | no | Enumerate every run directory of the project: a training run and a calibration run of a |

### feedback_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `prioritize_review_queue` | `feedback_tools.py:97` | no | Rank unfinished images by active-learning informativeness for the next review batch. |

### gui_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `push_panel_event` | `gui_tools.py:32` | yes | Push structured data to a TCIP GUI panel via the tcip-web backend. |
| `focus_human_attention` | `gui_tools.py:64` | yes | Drive the live Annotate tab to a (subject, date) frame in the right mode, showing the |

### inference_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `run_inference` | `inference_tools.py:157` | no, its library event | Run a trained model over images or a raster, and publish the predictions as a bucket. |
| `deliver_per_image_counts` | `inference_tools.py:277` | no, its library event | Deliver a published bucket's per-image detection counts as a CSV. |

### calibration_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `assess_checkpoint` | `calibration_tools.py:40` | no, its library event | Assess a checkpoint against the calibration and holdout sides of a selection, for one |
| `assess_reserved_regions` | `calibration_tools.py:84` | no, its library event | Assess a checkpoint trained on one mosaic against that mosaic's own reserved calibration and |
| `calibrate_physical_scale` | `calibration_tools.py:123` | no, its library event | Derive a per-pixel physical scale on a selection's calibration side and check it against |

### ingest_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `ingest_images` | `ingest_tools.py:208` | yes | Copy raw images into a structured project, bucketed by the capture date each file states. |
| `import_coco` | `ingest_tools.py:366` | no, its library event | Convert an external dataset-level COCO document into the dataset's per-image label documents. |

### meta_tools.py (5 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `report_friction` | `meta_tools.py:196` | yes | Log structured friction when you get stuck, confused, or surprised. |
| `load_project_memory` | `meta_tools.py:251` | yes | Read one project-memory corpus into context so context isn't lost between sessions. |
| `read_audit_log` | `meta_tools.py:314` | yes | Read one audit log's own entries: which door touched a dataset or project, when, with |
| `write_retrospective` | `meta_tools.py:476` | yes | Write an end-of-project retrospective to markdown. |
| `record_distillation_pass` | `meta_tools.py:573` | yes | Record that this project's friction reports and retrospectives were reviewed, resetting |

### knowledge_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `serve_domain_knowledge` | `knowledge_tools.py:34` | yes | Read the platform's domain knowledge: trait semantics, workflow patterns, and per-crop |

### model_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_model` | `model_tools.py:12` | no | Register a trained model in the project model registry. |
| `rank_registered_models` | `model_tools.py:71` | yes | List the project's registered models, or rank them by an explicit metric. |

### trait_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `propose_trait` | `trait_tools.py:12` | no, its library event | Propose a trait's complete entry, its spec fields and what its delivered number means per |

### orthomosaic_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `deliver_orthomosaic_plant_counts` | `orthomosaic_tools.py:221` | no, its library event | Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s). |

### delivery_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `deliver_per_plant_csv` | `delivery_tools.py:11` | no, its library event | The general per-plant CSV door: ``aggregate_per_plant``'s own output delivered over the |

### phenology_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_plant_registry` | `phenology_tools.py:26` | yes | Register a plant-locations CSV set under a name, so `build_plant_mapping` and |
| `build_plant_mapping` | `phenology_tools.py:78` | no, its library event | Assign each geolocated image to a plant, then persist the mapping under this project. |
| `deliver_phenology_milestones` | `phenology_tools.py:137` | no, its library event | Per-plant phenology milestones from classified prediction buckets and a plant mapping. |  <!-- queued: P5-43 unify -->

### project_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_dataset` | `project_tools.py:97` | yes | Record a dataset's identity so a delivered number can be traced to the exact data behind it. |
| `initialize_project` | `project_tools.py:173` | yes | Create a TCIP project: ``.tcip/`` with its artifacts and models directories, and its record |
| `view_gui_state` | `project_tools.py:205` | yes | The live GUI session the human is looking at in this project: dataset, date, trait, tab, |
| `inspect_project` | `project_tools.py:220` | yes | Get an overview of the project. |

`tools/bundle.py` (not a tool module: no `@tool()` sites) is the one membership accounting
`archive_project` and `import_project` both compose from, `account_for(tree)`. It derives every
root a project tree is or holds (the fixed `.tcip` structure, plus every `selection.json`
anchor under placement constraints that raise `AnchorMisplaced` when one
sits at the tree root, under `.tcip`, under a blob home, or under/above another derived root),
then classifies every file by precedence: bookkeeping, a record or log claimed by exactly one
derived root's own layout (`tcip_store.adoption.plan_root`; two roots claiming one file is a
`collisions` entry on the result, never a specificity tie), a recognized blob home, or
unaccounted. `archive_project` bundles the record/log and blob classes; `import_project` refuses
on any bookkeeping, collided, undecodable or unaccounted member before adopting or moving
anything.

### proposal_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `propose_annotations` | `proposal_tools.py:164` | yes | Propose candidate annotations on an image for review, using a chosen auto-labeling engine. |
| `stage_proposals` | `proposal_tools.py:611` | yes | Stage model-/agent-proposed shapes as predictions for canvas review, the "show on canvas |

### training_tools.py (6 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `launch_training` | `training_tools.py:333` | library | Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder. |
| `monitor_training` | `training_tools.py:478` | yes | Check the status of a training run, or of a hyperparameter sweep. |
| `cancel_training` | `training_tools.py:729` | yes | Request graceful cancellation of a running training run. |
| `run_hyperparameter_search` | `training_tools.py:918` | library | Run hyperparameter optimization on Ray Tune, training each trial for real. |
| `cancel_hyperparameter_search` | `training_tools.py:1258` | yes | Request cooperative cancellation of a running HPO sweep. |
| `evaluate_model` | `training_tools.py:1711` | no | Evaluate a trained checkpoint on a (held-out) dataset and return the result. |

### vision_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `capture_live_canvas` | `vision_tools.py:712` | yes | Render exactly what the human's GUI canvas shows right now: image, shapes, viewport. |

## 2. HTTP routes and WebSocket endpoints

`packages/tcip-web/src/tcip_web/app.py` builds the FastAPI app, registers 5 HTTP routes
and 2 WebSocket routes directly, then calls `register_all(app)` from
`packages/tcip-web/src/tcip_web/routes/__init__.py`, which `include_router`s 14 route
modules under `routes/`, each with a fixed prefix. Verified: `routes/__init__.py`,
`routes/_metrics_common.py`, and `routes/_body_common.py` define no routes of their own (0
`@router.*` decorator sites in any of the three); `_metrics_common.py` holds `metrics_response`,
the response shape `tuning.py`'s trial-metrics route answers in, and `_body_common.py` holds
`EmptyBodyPayload`, the empty body model six path-parameter-only routes now declare so the
browser must send a preflighted request rather than reaching the handler as a simple one.

Total HTTP routes at HEAD: 69 (5 on `app.py` plus 64 across the 14 route modules, both counts
obtained by grepping `@app.get/post(` and `@router.get/post(` and summing);
websocket routes are counted separately, below, and excluded from this total. Each per-router
heading's own route count (and their sum, 67) includes any websocket route it lists, since
`routes/inference.py`, `routes/terminal.py` and `routes/training.py` each carry one; net of
those three, the 14 modules hold the 64 HTTP routes counted here.

Total WebSocket routes at HEAD: 5 (`/ws/state`, `/ws/panel/{panel}` on `app.py`;
`/api/terminal/ws/{session_id}` on `routes/terminal.py`; `/api/inference/jobs/{job_id}/stream`  <!-- queued: P5-124 unify -->
on `routes/inference.py`; `/api/training/runs/{experiment_id}/stream` on `routes/training.py`).

Tables below group by defining module. Column "line" is the handler's `def`/`async def` line,
the same convention the tool tables use; the `@router.*`/`@app.*` decorator carrying the method
and path sits directly above it. Every method, path, and handler name below is the one
registered at HEAD.

### app.py (not under a router prefix)

| method | path | handler | line |
|---|---|---|---|
| GET | `/api/state` | `get_state` | `app.py:148` |
| POST | `/api/state/tab` | `set_active_tab` | `app.py:157` |
| WS | `/ws/state` | `state_ws` | `app.py:166` |
| GET | `/health` | `health` | `app.py:256` |
| GET | `/` | `index` | `app.py:264` |
| POST | `/api/events/{panel}` | `post_panel_event` | `app.py:287` |
| WS | `/ws/panel/{panel}` | `panel_ws` | `app.py:324` |

### routes/annotate.py, prefix `/api/annotate` (5 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/labels` | `load_labels` | `routes/annotate.py:103` |
| POST | `/labels` | `save_labels` | `routes/annotate.py:119` |
| GET | `/proposals` | `load_proposals` | `routes/annotate.py:164` |
| POST | `/queue/launch` | `launch_priority_queue` | `routes/annotate.py:243` |
| GET | `/queue/{job_id}` | `get_priority_queue_job` | `routes/annotate.py:263` |

### routes/canvas.py, prefix `/api/canvas` (1 route)

| method | path | handler | line |
|---|---|---|---|
| POST | `/state` | `push_canvas_state` | `routes/canvas.py:56` |

### routes/dataset.py, prefix `/api/dataset` (3 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/tree` | `get_dataset_tree` | `routes/dataset.py:109` |  <!-- queued: P5-83 unify -->
| POST | `/select` | `select_dataset` | `routes/dataset.py:159` |
| POST | `/nav` | `set_current_image` | `routes/dataset.py:230` |

### routes/fs.py, prefix `/api/fs` (1 route)

| method | path | handler | line |
|---|---|---|---|
| GET | `/list` | `list_dir` | `routes/fs.py:88` |

### routes/images.py, prefix `/api/images` (5 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/serving_grid` | `get_serving_grid` | `routes/images.py:254` |
| GET | `` (root) | `serve_image` | `routes/images.py:492` |
| GET | `/bands` | `get_bands` | `routes/images.py:688` |
| POST | `/overviews` | `build_image_overviews` | `routes/images.py:804` |
| GET | `/overviews/status` | `get_overview_job` | `routes/images.py:822` |

### routes/inference.py, prefix `/api/inference` (3 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| POST | `/launch` | `launch_inference` | `routes/inference.py:148` |
| GET | `/jobs` | `list_jobs` | `routes/inference.py:191` |
| POST | `/jobs/{job_id}/cancel` | `cancel_job` | `routes/inference.py:196` |
| WS | `/jobs/{job_id}/stream` | `stream_job` | `routes/inference.py:206` |

### routes/meta.py, prefix `/api/meta` (2 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/reports` | `get_reports` | `routes/meta.py:20` |
| GET | `/retrospectives` | `get_retrospectives` | `routes/meta.py:42` |

### routes/projects.py, prefix `/api/projects` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `` (root) | `list_projects` | `routes/projects.py:83` |
| POST | `/open` | `open_project` | `routes/projects.py:138` |
| POST | `/remove` | `remove_project` | `routes/projects.py:180` |
| POST | `/rename` | `rename_project_route` | `routes/projects.py:220` |

### routes/results.py, prefix `/api/results` (10 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/plant_mapping/build` | `build_plant_mapping` | `routes/results.py:92` |
| POST | `/plant_mapping/load` | `load_plant_mapping` | `routes/results.py:126` |
| GET | `/plant_mapping/list` | `list_plant_mappings` | `routes/results.py:144` |
| POST | `/phenology_measurement` | `phenology_measurement` | `routes/results.py:201` |
| POST | `/export_csv` | `export_csv` | `routes/results.py:300` |
| POST | `/export_count_csv` | `export_count_csv` | `routes/results.py:378` |
| GET | `/traits` | `list_traits` | `routes/results.py:501` |
| POST | `/traits/confirm` | `confirm_trait_revision` | `routes/results.py:545` |
| GET | `/delivery-events` | `list_delivery_events` | `routes/results.py:445` |
| GET | `/models/registered` | `registered_models` | `routes/results.py:578` |

### routes/sessions.py, prefix `/api/sessions` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/image_event` | `image_event` | `routes/sessions.py:73` |
| POST | `/start` | `start_session` | `routes/sessions.py:132` |
| POST | `/end` | `end_session` | `routes/sessions.py:148` |
| GET | `/load` | `load_sessions` | `routes/sessions.py:164` |

### routes/subjects.py, prefix `/api/subjects` (2 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/load` | `load_subjects` | `routes/subjects.py:52` |
| POST | `/save` | `save_subjects` | `routes/subjects.py:105` |

### routes/terminal.py, prefix `/api/terminal` (3 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| GET | `/status` | `get_status` | `routes/terminal.py:357` |
| POST | `/sessions` | `create_session` | `routes/terminal.py:404` |
| POST | `/sessions/{session_id}/restart` | `restart_session` | `routes/terminal.py:433` |
| POST | `/sessions/{session_id}/submit` | `submit_to_session` | `routes/terminal.py:458` |
| WS | `/ws/{session_id}` (full path `/api/terminal/ws/{session_id}`) | `terminal_ws` | `routes/terminal.py:477` |

### routes/training.py, prefix `/api/training` (10 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| GET | `/configs` | `list_configs_route` | `routes/training.py:24` |
| GET | `/configs/{experiment_id}/splits` | `list_split_choices_route` | `routes/training.py:32` |
| POST | `/runs` | `relaunch_config_route` | `routes/training.py:50` |
| GET | `/runs` | `list_runs_route` | `routes/training.py:84` |
| GET | `/runs/{experiment_id}` | `get_run` | `routes/training.py:92` |
| POST | `/runs/{experiment_id}/tensorboard` | `launch_run_tensorboard` | `routes/training.py:99` |
| POST | `/runs/{experiment_id}/cancel` | `cancel_run_route` | `routes/training.py:129` |
| POST | `/compare` | `compare_runs_route` | `routes/training.py:149` |
| POST | `/compare/best` | `compare_best_route` | `routes/training.py:163` |
| GET | `/metric-directions` | `metric_directions_route` | `routes/training.py:199` |
| WS | `/runs/{experiment_id}/stream` (full path `/api/training/runs/{experiment_id}/stream`) | `training_stream_ws` | `routes/training.py:271` |

### routes/tuning.py, prefix `/api/tuning` (10 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/sweeps` | `relaunch_sweep` | `routes/tuning.py:105` |
| POST | `/sweeps/{sweep_id}/cancel` | `cancel_sweep_route` | `routes/tuning.py:130` |
| GET | `/sweeps` | `list_sweeps` | `routes/tuning.py:142` |
| GET | `/sweeps/{sweep_id}` | `get_sweep` | `routes/tuning.py:155` |
| GET | `/sweeps/{sweep_id}/trials` | `list_trials` | `routes/tuning.py:169` |
| GET | `/sweeps/{sweep_id}/trials/{trial_id}/metrics` | `get_trial_metrics` | `routes/tuning.py:177` |
| GET | `/ray-dashboard` | `get_ray_dashboard` | `routes/tuning.py:195` |
| POST | `/sweeps/{sweep_id}/tensorboard` | `launch_sweep_tensorboard` | `routes/tuning.py:256` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard` | `launch_trial_tensorboard` | `routes/tuning.py:267` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop` | `stop_trial_tensorboard` | `routes/tuning.py:281` |

### 5 routes with no located frontend caller

Per phase0's `web-surface.md`, these 5 registered backend routes had no caller found
under `packages/tcip-web/frontend/src/` by literal-path grep. Not re-derived this
session; restated from phase0 as the brief instructs, with each route's line verified
above against HEAD.

| method | path | note (from phase0) |
|---|---|---|
| GET | `/health` | not fetched from `frontend/src`; a liveness endpoint |
| GET | `/` | loaded by the browser's own navigation, not via fetch/XHR from app code |
| POST | `/api/events/{panel}` | posted by MCP tools (`tcip_mcp.web_client`), not by the browser |
| GET | `/api/state` | no caller found |
| POST | `/api/training/compare` | no caller found; `trainingApi` has no `compare` function |  <!-- queued: P5-111 keep-as-tool -->

`/api/training/compare` is kept regardless: it is the backend for the comparison view, which
wires it directly rather than through `trainingApi`.

## 3. tcip-annotation importable public symbols

`packages/tcip-annotation/src/tcip_annotation/__init__.py:21-36` defines `__all__`: 11 entries.

| name | re-exported from | `__init__.py` line |
|---|---|---|
| `Annotation` | `state` | 3 |
| `BBox` | `state` | 3 |
| `Point` | `state` | 3 |
| `Polygon` | `state` | 3 |
| `bbox_of` | `state` | 3 |
| `read_annotations` | `json_io` | 11 |
| `write_annotations` | `json_io` | 11 |
| `parse_coco_annotations` | `format_io` | 15 |
| `point_in_polygon` | `matching` | 16 |
| `mask_to_polygon_rings` | `mask_contours` | 18 |
| `cell_fields` | `grid` | 19 |

Package no-dependency claim, restated from phase0 and not re-derived this session: no
`import tcip_mcp` / `import tcip_web` statement exists anywhere in
`packages/tcip-annotation/src/tcip_annotation/`; `packages/tcip-annotation/CLAUDE.md`
states the same rule and is loaded automatically when reading files in the package.

Names not re-exported in `__all__` but importable directly from their defining submodule
(restated from phase0, not re-derived this session): `json_io.ANNOTATIONS_KEY`,
`json_io.UNLABELED`, `json_io.target_class_id`,
`json_io.LABEL_SUFFIX`, `json_io.LabelDocument`, `json_io.read_label_document`,
`format_io.coco_categories`, `mask_contours.DEFAULT_EPSILON_FRAC`,
`verdicts.Verdict`, `verdicts.verdict_key`, `verdicts.read_verdicts`,
`matching.pair_detections`, `matching.pair_proposals`, `grid.grid_to_rect`,
`grid.column_label`, `grid.column_index`, `utils.auto_orient_image`,
`utils.get_image_dimensions`, and every public name in `viz.py`
(`COLOR_PALETTE`, `render_detections`, `render_segmentations`, `render_comparison`,
`render_grid`, `render_candidates`, `render_grid_overlay`, `render_canvas_state`).

## 4. Entry points

`python -m tcip_mcp`: `packages/tcip-mcp/src/tcip_mcp/__main__.py:1-5` imports `main`
from `tcip_mcp.server` and calls it: `packages/tcip-mcp/src/tcip_mcp/server.py:104`
(`def main(argv: list[str] | None = None) -> None:`), which takes `--project <path>`, the one
project the server acts on, and runs the server `build_server` (`server.py:86`) builds for it:
every function a `@tool()` decorator (`server.py:31`) in
`packages/tcip-mcp/src/tcip_mcp/tools/*.py` declared, each bound to that project
(`python tools/list_tools.py` lists them; the count is never written down, since it drifts).

`python -m tcip_web`: `packages/tcip-web/src/tcip_web/__main__.py` defines `main()`
(`packages/tcip-web/src/tcip_web/__main__.py:34`), which reads the workspace once
(`tcip_mcp.workspace.workspace_from_environment`, `workspace.py:32`, refusing an unset
`TCIP_WORKSPACE`), reads `TCIP_WEB_PORT` (default `8765`) and binds the loopback address
`BACKEND_HOST` (`packages/tcip-mcp/src/tcip_mcp/web_client.py:20`, `"127.0.0.1"`, the one host the
MCP tools reach too), writes the bound port under that
workspace (`replace(backend_port_key(workspace), str(port))`,
`packages/tcip-web/src/tcip_web/__main__.py:44`), configures the app's `StateStore` with the
same workspace (`store.configure(workspace, image_roots_from_environment())`,
`packages/tcip-web/src/tcip_web/__main__.py:52`), and serves the app via
`uvicorn.run(app, host=BACKEND_HOST, port=port)` (`packages/tcip-web/src/tcip_web/__main__.py:53`).
The app's lifespan (`app.py:42`) opens the project the workspace's last-opened pointer names
(`open_last_opened`, `packages/tcip-web/src/tcip_web/routes/projects.py:120`); with no pointer
the backend starts with no project open and every project-scoped route answers 409 until one is
opened from the picker. Every process reads the workspace once at its entry and hands it on as a
value; a test configures its own `StateStore`. Locality is a property
of the accepted connection rather than the bind host: every request is checked by
`tcip_web.trust_boundary.TrustBoundaryMiddleware` (`trust_boundary.py:193`), which refuses an
arrival that is not through this machine (`local_arrival`, `trust_boundary.py:123`). The same
middleware applies one Origin policy (`origin_allowed`, `trust_boundary.py:180`) to every
WebSocket scope and to every HTTP scope whose method is state-changing
(`STATE_CHANGING_METHODS`, `trust_boundary.py:30`), rather than leaving each handler to call it
for itself.

`.mcp.json` (repo root): declares one MCP server, `tcip`, which launches
`conda run -n tcip-agent --no-capture-output python -m tcip_mcp`. A semantic code-search server
(claude-context, backed by an embedding model and a vector store) is optional developer tooling
some maintainers configure locally; it is not part of the platform and is not declared in this
tracked file, so its presence or configuration varies per machine.

`tools/` (repo root): a non-API surface, not imported by `tcip_mcp`, `tcip_web`, or
`tcip_annotation` package code; each file is a standalone CI/development script invoked
directly (`python tools/<name>.py`). `tools/README.md` names every tracked script here, held
to the tree by `tests/test_tools_readme_index.py` rather than a hand-written count or list. An
operator command instead lives as a `cli/` module inside the package whose imports it needs,
registered under a name in `tcip_web.cli.COMMANDS` and run as `tcip <name>` (or, without the
console script installed, `python -m tcip_web.cli <name>`).


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
`packages/tcip-annotation/src/tcip_annotation/json_io.py:894`;
`tcip_mcp.pipelines.data.coco_import.import_coco_document`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/coco_import.py:23`;
`tcip_mcp.dataset_layout.save_label_document`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:474`, the one label save every door calls;
`tcip_mcp.buckets.publish`, `packages/tcip-mcp/src/tcip_mcp/buckets.py:189`;
`tcip_mcp.tools.proposal_tools._stage_document`,
`packages/tcip-mcp/src/tcip_mcp/tools/proposal_tools.py:451`.

Readers: `tcip_annotation.json_io.read_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:605`;
`tcip_mcp.dataset_layout.subjects_on_date`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:584`.

A prediction record's `created_by` is one spelling, `buckets.prediction_producer`,
`packages/tcip-mcp/src/tcip_mcp/buckets.py:263`, so every checkpoint-backed writer
stamps the same `model:<checkpoint-stem>@<sha256-prefix>` producer identity rather than each door
composing its own string.

Seam S17 ("Canonical per-image annotation JSON schema"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: every writer and the shared `json_io.read_annotations`
reader are exercised in round-trip tests
(`tests/test_tcip_web_routes.py`, `tests/test_label_document_gestures.py`,
`tests/test_review_channel.py`, `tests/test_name_based_annotation_schema.py`). Gap
recorded by S17: no test cross-writes through one production writer and cross-reads through a
different consumer path in the same test.

## 2. External COCO dataset JSON, import only

Path: caller-supplied, single dataset-level `.json` file, not per-image. The platform writes none.

Reader: `tcip_annotation.format_io.parse_coco_annotations`, `format_io.py:68`, which names each
record's subject and decodes it through format 1's own decoder, called only by
`tcip_mcp.pipelines.data.coco_import.import_coco_document`, `coco_import.py:23`, which writes the
per-image documents of format 1 from it. Format 1's reader refuses this shape wherever it sits.

## 3. `subjects.json`, subject registry

Path: `<dataset_root>/subjects.json`. A dataset root carrying the retired `classes.json` is
refused by every registry writer until the file is renamed to `subjects.json` by hand; no
platform door conforms it. See S20 below.

Writer: `tcip_mcp.subject_registry.replace_registry`,
`packages/tcip-mcp/src/tcip_mcp/subject_registry.py:212`, the one write both registry doors call
(the GUI's `save_subjects` and the tool's `write_subject_registry`,
`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:289`).

Readers: `tcip_mcp.subject_registry.read_registry`, `subject_registry.py:187`;
`tcip_mcp.dataset_layout.list_subjects` (delegates to `subject_registry`),
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:744`.

`assign_class_ids`, `subject_registry.py:319`, derives the training-time name-to-id map from this
file's declared attribute order; no integer id is stored in the file itself.

Seam S20 ("subjects.json subject registry"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_name_based_annotation_schema.py:84` writes the
registry through the real `replace_registry` and reads `num_classes` back through the real training
loader's call to `subject_registry.assign_class_ids`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:55`
(`return registry, subject_registry.assign_class_ids(registry, subject, attribute)`).
Gap: no test drives the actual `/api/subjects/save` HTTP route in the same test as the
training-side read.

## 4. `dataset.json`, dataset identity

Path: `<dataset_root>/dataset.json`.

Writer: `tcip_mcp.tools.project_tools.register_dataset`,
`packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:97`.

Reader: `tcip_mcp.pipelines.data.dataset_fingerprint.dataset_fingerprint` (recompute-on-read is
the stated authority; the stored value is a cache),
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py:109`
(`def dataset_fingerprint(dataset_root: str | Path) -> str | None:`).

Seam S26 ("dataset.json identity and fingerprint"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_dataset_identity_recording.py:78` calls the
real `register_dataset` writer, then the real `split_construction.dataset_identity` reader, both of
which call the identical `dataset_fingerprint` function, and asserts they agree. Gap: the seam's
named third consumer, `tcip check-dataset-identity`, is never executed by any test.

## 5. Completion marks, inside format 1's label document

Path: format 1's own document, under its `complete` key: per subject, a list of marks, each
`{rect: [x, y, w, h], by, at, digest, proposals_hidden}`.

Writer: `tcip_mcp.dataset_layout.save_label_document`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:474`, through the one encoder
`json_io.encode_annotations`, which keeps a subject's marks only while their digest
(`json_io.subject_digest`, over `json_io.canonical_digest`) still names that subject's
annotations, so an edit and the loss of its subject's marks are one write of one document.

Reader: `tcip_annotation.json_io.completion_marks`, decoded by `label_document`, and
`LabelDocument.state(subject)`, the one reading of `complete`, `negative`, `partial` and
`unannotated`, which the training admission (`label_queries.admitted_records`), the review queue
(`feedback_tools._prepare_queue_sources`), the editor's listing (`routes/annotate.py`'s
`_completion`), the session telemetry (`routes/sessions.py`'s `_marked_negative`) and the doctor
read; block calibration reads each mark's rect through `json_io.covers`
(`pipelines/block_calibration.py`'s `check_completeness`). A selection's sample names the
document its marks live in through its own `ground_truth` path.

Guards: `tests/test_label_document_gestures.py` (a mark on one subject survives another
subject's edit, a negative is read alike at the admission, the queue and the listing, a mark
carries `proposals_hidden`), `tests/test_block_calibration.py` (a mark's rect attests that region
alone).

## 9. `audit_log`, one append-only store under two kinds of root

Path: `.tcip/audit.jsonl` under the root an entry's scope names (`audit_log_key`,
`packages/tcip-mcp/src/tcip_mcp/audit.py:74`): a dataset root for an event that changed a record
traveling with the data; the project root otherwise. Every writer names its scope; there is no
process-wide default log. That path is what the file backend places the log at and what
`tcip export-store` writes back out; on the default database backend the rows live in that
root's `.tcip/store.db` until they are exported.

Writers: three write paths, `audit.py:186` (`audited`), `audit.py:113` (`record_event`) and
`audit.py:131` (`record_event_or_raise`), each appending at `audit_log_key(scope)`; a line records
no `scope` field, since the log it sits in is its scope. No line carries a `schema_version` field:
absence is the frozen version 1, `frozen-formats.json`'s ceiling for this store.

Every mutating door leaves exactly one line per act, the decorator's or the library's. The
decorator writes none for a call that returns its error dict and an `exception` line for a call
that raises; whatever a refusing call committed first is recorded by the library that committed
it. The library that mutates records the act, and the decorator stays only on a door whose own
act has no library owner: `audited` covers the platform's mutating doors (the MCP tools in
`tools/`, plus the script-invoked doors demoted from them) except a door whose libraries record
each of its acts (`import_coco`, `build_plant_mapping`, `deliver_per_plant_csv`,
`deliver_orthomosaic_plant_counts`, `deliver_phenology_milestones`, `register_model`,
`run_inference`, `deliver_per_image_counts`, `propose_trait`, `save_annotations`,
`write_subject_registry`, `run_hyperparameter_search`, and the three assessment doors
`assess_checkpoint`, `assess_reserved_regions` and `calibrate_physical_scale`, whose assessment
leaves its library's line): bare, the project's own log;
`@audited(scope_arg=...)` names the argument carrying
a dataset or project location, resolved via `dataset_scope_of` (`audit.py:154`) (through the tool's own
canonicalizer when the declaration passes one as `scope_via`). Four doors declare one: three
dataset-scoped (`register_dataset`, `tools/project_tools.py:97`; `propose_annotations`,
`tools/proposal_tools.py:164`; `stage_proposals`, `tools/proposal_tools.py:611`). A resolution that answers "no dataset" leaves
the entry in the project's own log; a resolver that raises refuses the call rather than filing it
there.

`record_event`/`record_event_or_raise` cover code that is neither an MCP tool nor a demoted door.
Project-scoped: the training envelope's open/close events
(`pipelines/training/envelope.py`), the model registry's write event (`model_registered`,
`model_registry.py`), an assessment's one line once its record is written
(`assessment_recorded`, `assessment.py`'s `_finish`),
and `routes/terminal.py`'s one line per agent-terminal launch (`agent_terminal_started`, `routes/terminal.py:99`). The `@audited(scope_arg=...)`
doors span every category by whatever root their declared argument resolves; this paragraph
names the explicit-emitter files, not a closed census of the decorator's doors.
Dataset-scoped: the label save every door calls (`save_label_document`,
`dataset_layout.py`), dataset-scoped when the label lies in a dataset and project-scoped
otherwise, the registry write both doors call (`replace_registry`, `subject_registry.py`),
`delivery.py`'s `delivery_event` line
(`delivery.py`'s `deliver_csv`, dataset-scoped when a
delivery's buckets share one dataset root, project-scoped otherwise),
the COCO import's `coco_document_imported` (`pipelines/data/coco_import.py`), written
once a document has committed, on success and on a later failed write alike, and a prediction
bucket's `prediction_bucket_published` (`buckets.py`'s `publish`),
written for an image bucket and a raster bucket alike, whichever door
published it (the GUI's inference worker included), under status
`failed` naming the documents written and the error when a pass raises after its first document
and before its `bucket.json`, filed under the bucket's dataset root or the project's log when the
bucket sits under none.
Project-scoped: `pipelines/postprocessing/plant_mapping.py`'s `persist_mapping`, reached from
both doors through the one build (`build_plant_mapping`, `pipelines/postprocessing/plant_mapping.py`),
whose receipt is the build's one line in the project's own log: the MCP tool passes the project
its server was started for and the web build route passes the backend's open project; and a
sweep's opening (`open_sweep`, `tools/training_tools.py`), which both the tool and the GUI's
relaunch reach.

What a failed append means is where the three write paths part: `record_event` warns and
returns, because its callers bracket work rather than follow a
mutation; `record_event_or_raise` raises `AuditEntryNotWritten`; the decorator raises
`MutationCommittedWithoutAuditLine`, because its append runs after the tool body and a warning
there invites a blind retry of a mutation already on disk.

A log's own root is its scope, so a line names no root and a moved or imported project's log
carries no machine path of its own; arguments a caller passed as absolute paths travel in an
archive unredacted, since a project archive is provenance-preserving, not path-sanitized.

Readers: one production parser, reading through the storage seam's `read_log` rather than
decoding lines by hand, and refusing (never scanning past) a page reporting corruption or an
unknown `schema_version`. `plant_mapping._scan_receipts`
(`pipelines/postprocessing/plant_mapping.py:1451`) and `_require_receipt`
(`pipelines/postprocessing/plant_mapping.py:1511`), the hard receipt gate `load_mapping` runs
before trusting a persisted mapping record:
every `plant_mapping_built` entry in the record's own project log is scanned for a receipt naming
the record's digest, and a page reporting `page.corrupt` or `page.version_refused` raises rather
than reading past it, since an entry could be hiding behind either kind of unreadable line unread.

Beyond these two, `archive_project` (`tools/project_tools.py`) bundles the log file as a claimed
ROOT-layout record through the shared membership accounting (`tcip_mcp.tools.bundle.account_for`)
without opening it, and every other reader goes through the storage seam's own `read_log` or a
test asserting named keys, so a new per-entry field (such as `schema_version` here) is additive
for every consumer beside the two parsers above.

## 10-15. `.tcip/experiments/<experiment_id>/`, a run directory

Path root: `.tcip/experiments/<experiment_id>/`, resolved via `experiments_dir`,
`packages/tcip-mcp/src/tcip_mcp/experiments.py:92`, under the project the caller names. A run
is a directory of plain files, not store records: each file is written by one process, the launch
record once by the parent and the final status once by the child, and nothing in it is rewritten.
Immutability is the files' own: `publish_once`, `experiments.py:189`, writes a file's whole bytes
to a staging name and publishes them under the final name without replacing anything, so a file
under its final name is always whole. A relaunch or a resume is a new directory naming its source
(`parent_experiment`, `resume_from`); `create_run_directory`, `experiments.py:154`, refuses a
directory that already exists (`RunDirectoryExists`). A sweep is the same shape one level up,
`.tcip/hpo/<study_name>/` (`sweeps_dir`, `experiments.py:97`): its `sweep.json` input written
once before its thread starts, a heartbeat, its final status, and one `trial_<id>/` run directory
per trial, from which its projection derives.

- `run.json` (`RUN_FILE`, `experiments.py:43`): written once by `open_run`,
  `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:301` (`def open_run(`), before the child
  starts: the config as launched with its seed drawn onto it, what the launcher resolved it to
  (`resolved`: the data section, the partition and the objective with its direction, from
  `split_construction.resolve_run`; `null` for an HPO trial whose point failed to resolve, whose
  final status is written `failed` naming why), the environment, the dataset identity, the
  run it relaunched and the checkpoint it resumes from, its wall clock, the model contract
  preflight proved, the source snapshot, and an HPO trial's sampled point. The child builds its
  loaders from `resolved` and resolves nothing again. Read through `observe`,
  `experiments.py:357`, and `run_resolution`, `experiments.py:414`, which `assessment.py` takes the
  partition from.
- `metrics.jsonl` (`METRICS_FILE`, `experiments.py:44`, append-only): created empty with the
  directory, appended through `append_row`, `experiments.py:230`, by the training envelope's one
  sink, `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:223`
  (`def _epoch_sink(`). Read by `read_rows`, `experiments.py:249`. A row landing after the final
  status is reported as late (`rows_after_end`), never read as reopening the run.
- `heartbeat` (`HEARTBEAT_FILE`, `experiments.py:45`): touched by `keep_heartbeat`,
  `experiments.py:275`, while the run's process lives; `last_alive`, `experiments.py:292`, falls
  back to the launch record before the first touch.
- `cancel_requested.json` (`CANCEL_FILE`, `experiments.py:46`): written once by `request_cancel`,
  `experiments.py:311`, which keeps the first request's time and refuses a directory whose final
  status is written.
- `final_status.json` (`FINAL_STATUS_FILE`, `experiments.py:47`): written once through
  `write_final_status`, `experiments.py:218`, by the envelope or by the child's pre-envelope
  failure path: the state, when it ended, its error, and for a completed run the checkpoint
  (path inside the run directory and sha256) the verified checkpoint reader admitted. `observe`
  answers that state once written, else `running` or `interrupted` by the heartbeat window; a
  run's summary, completed or live, is the best selection over its own metrics rows under its
  recorded objective (`run_summary`, `experiments.py:437`).
- The checkpoints: the files the run's body saves (`model_best.pt`, held in memory until the run
  ends, `model_final.pt`, `checkpoint_epoch_*.pt`, a bespoke loop's own tags), each written once
  under a name no other write takes. An evaluation of the run writes nothing here.

Readers over the whole directory: `get_experiment`, `experiments.py:474`; `compare_experiments`,
`experiments.py:522`; `get_experiment_lineage`, `experiments.py:586`.

The run directory's lifecycle, each case through the real launcher and a real child process
(launch to completion, cancel by id, resume into a new directory, a child killed before its
final status, a row logged after the final status, a launch into an existing directory, and an
HPO trial as a run directory reporting its sweep's one objective), is held by
`tests/test_run_directory_lifecycle.py`. The metrics log's writer and the training stream's reader
are held against each other by `tests/test_metrics_row_writer_reader_agreement.py`, which writes
rows through the real run body and reads them back through the real websocket route; the tuning
trial-metrics route reads the same file through `read_rows`. The partition's writer (the
launcher's `resolve_run`, recorded by `open_run`) and its readers (`assessment.py`'s disjointness
check and its reserved-regions assessment) are driven against the same
resolved record by `tests/test_spatial_region_containment.py`,
`tests/test_run_partition_membership_fidelity.py` and `tests/test_selection_binding.py`.

## 16. `.tcip/models/registry.json`, trained-model registry

Path: `<project_path>/.tcip/models/registry.json`.

The registry is one relation from sha256 to its one owner, `registered_entries`,
`packages/tcip-mcp/src/tcip_mcp/model_registry.py:121`: the completed run whose final status names
those bytes (`run_entry`, `model_registry.py:106`, named by the run's id and carrying it as
`experiment_id`; the earliest to complete when two do), else the index's foreign entry with
`experiment_id` null. A run's completion is its registration: nothing writes a run's entry into
the index. The index holds foreign registrations, one entry per sha256, written by
`ModelRegistry.register_model` (the `register_model` tool's door, `tools/model_tools.py:17`),
which digests the bytes the verified checkpoint reader admits (`admitted_digest`,
`model_registry.py:244`), replaces an earlier entry of the same sha256 and returns the owner the
relation answers, so a registration before or after the producing run completes leaves one entry;
a name is presentation only. What a ranking reads (`kind`, `metrics`, `metrics_source`) comes from
the checkpoint's payload through that reader (`entry_facts`, `model_registry.py:136`): a run's
metrics are the ones its payload carries, sourced `"trainer"` or `"training_source"`, a foreign
entry's the ones its registration stated, sourced `"caller"`.

The index document is `{entries: [...]}`, no `schema_version` field until this store's first bump
(absence is the frozen version 1), read and written through one pair,
`_read_registry_document`/`_write_registry_document`, that is the only code touching the raw
value: an absent key answers the empty document, a present bare array is the shape this store
carried before the family that wrapped it and refuses by name (`RegistryVersionRefused`,
deliberately not a `StoreError`) stating that no operator door rewraps a live project's registry
in place: the registry predates the entries-mapping shape the platform writes, and nothing
repairs it in place. `ModelRegistry.register_model` spells `checkpoint_path` through
`registry_paths.checkpoint_registry_path_for` against the registry's own scope root: relative
POSIX when the checkpoint resolves under it, absolute when it does not (the dataset registry's
own `entry_is_external`/`registry_path_for` share the same containment core and grammar-aware
`is_external_form` test, `registry_paths.py`). An entry's path is read as written; nothing
searches for a moved checkpoint by suffix, basename or digest.

Readers: `read_registry_index`, `model_registry.py:95`, the read path for anything outside the
module (`packages/tcip-mcp/src/tcip_mcp/cli/doctor.py:409`, `"metrics_source"`), and the accessors built on
it: `ModelRegistry.list_models`, `model_registry.py:429`, and the module's `best_model`,
`model_registry.py:425`, over its entries. `best_model` takes `metric_key` and `higher_is_better`
as required keywords, no default and no name heuristic, and by default ranks only entries whose
`metrics_source` is `"trainer"` (`include_unverified=True` also ranks the rest). The
`rank_registered_models` tool (`tools/model_tools.py:123`) resolves `higher_is_better` from
`evaluation.HIGHER_IS_BETTER_BY_METRIC` (`pipelines/training/evaluation.py:58` (`HIGHER_IS_BETTER_BY_METRIC: dict[str, bool] = {`)) when the caller
states none, the single declared-direction mapping `resolve_selection_metric`
(`pipelines/training/generic_trainer.py`) also reads for the trainer's own checkpoint selection.
Every one of these accessors, plus `register_model`'s own return, answers `checkpoint_path`
resolved to an absolute path (`registry_paths.resolved_registry_path`) on a copy, never the
registry's own internal relative-or-absolute storage spelling; bundle's checkpoint resolution and
`doctor.py`'s registry check resolve through the same function. `unresolved_registered_checkpoints`
and `import_project`'s own disclosure split on `is_external_form`: a designed-external entry is
`external_checkpoints` (its own existence stated per entry), never counted toward
`checkpoint_paths_unresolved` (an entry expected to resolve under the tree that does not).

Seam S27 ("Trained-model registry .tcip/models/registry.json"), verdict `one-side-only`,
`phase0_implementation: once, shared`: `tests/test_lifecycle_wiring.py:7`,
`tests/test_model_registry_metrics.py:8,40,54,70`, `tests/test_provenance_spine.py:70,84,94,111`,
`tests/test_tcip_web_results_routes.py:593,605`. Gap: no test registers a real model and then
calls `GET /api/results/models/registered`, or runs an inference launch end to end through the web
route's identity-stamp block, to confirm the GUI-visible entry matches the MCP-registered one; the
only two web-route tests check a 403-confinement case and an empty-registry case.

## 17. Prediction buckets, write-once prediction directories

A prediction bucket is not a score bin or a quota allocation: it is one directory holding a
single model run's per-image prediction documents and one `bucket.json` (format 18), its identity
the directory's own path (relative to the dataset root under one, its own resolved path otherwise,
`bucket_key_of`, `packages/tcip-mcp/src/tcip_mcp/buckets.py:268`). A bucket is published once:
`publish`, `buckets.py:189`, creates the directory through `create_run_directory` before it
consumes a document and refuses an existing one (`BucketExists`), so a new run names a new bucket
and nothing rewrites a published one. A directory holding documents and no `bucket.json` is not a
bucket, and every reader refuses it. A staged proposal is a bucket of its own, one per staged
image, its record naming the engine or agent that proposed it and no execution.

Path: caller-named, on every door alike; nothing composes it from a model name and a date.

Writer: `publish`, `buckets.py:189`, writes each per-image document through
`write_predictions_json` (format 1's schema) as the pass yields it, then `bucket.json` once.

Readers: `read_bucket`, `buckets.py:95`, the one decoder; `buckets_by_date`, `buckets.py:159`,
enumerates a dataset's buckets by reading each record; `Bucket.document`, the one document lookup,
answers only the documents the record names, so a file dropped beside them is read by no reader.
The reserved name `BUCKET_RECORD` is refused by the encoder itself
(`tcip_annotation.json_io.is_reserved_stem`).

A second run into a published bucket refuses before any pass and leaves it
(`tests/test_run_inference_bucket_handling.py:82`); publishing the same bucket twice refuses the
second publish (`tests/test_end_to_end_measurement_chain.py:178`).

## 18. `bucket.json`, a prediction bucket's record

Path: `<prediction_bucket_dir>/bucket.json`, sibling of the bucket's per-image prediction files,
the one name `BUCKET_RECORD` reserves and every bucket enumeration excludes (format 17).

Writer: `publish`, `packages/tcip-mcp/src/tcip_mcp/buckets.py:189`, once, through `write_once`,
after every document it states is on disk: the producer (the checkpoint's sha256 and producing
run), the scope (subject, attribute, id map), the execution record the pass ran under, the
capture's dataset id and date, the raster path with the raster's identity for a raster pass, the
documents by stem with the source file each was predicted from, the count of dropped zero-extent
boxes, and the id of the assessment it was published under, if any. Every door that predicts
publishes through it: `run_inference` for an image directory and a raster alike, and the GUI's
inference worker.

Reader: `read_bucket`, `buckets.py:95`, the one decoder, refusing a directory with no record or
one that does not decode (`ValueError`). `tools/orthomosaic_tools.py`'s
`deliver_orthomosaic_plant_counts` reads `raster_identity` back and refuses a delivery whose
supplied raster does not match it; `delivery.gate` reads `assessment_id`, the producer and the
date for its checks (format 27).

## 19. `.tcip/state/gui.json`, live GUI state snapshot

Path: `<project_root>/.tcip/state/gui.json`, addressed by `gui_snapshot_key`,
`packages/tcip-mcp/src/tcip_mcp/web_client.py:66`.

Writer: `write_gui_snapshot`, `tcip_mcp/web_client.py:266`, called by `StateStore.mutate`
(`tcip_web/state.py:170`) for the project open when the change is made, before the change is
held; a write that fails raises and the change is not held. The store is declared
`durable=False`: losing the last snapshot costs a re-selection, not history.

Reader: `read_gui_snapshot`, `tcip_mcp/web_client.py:280`, run by `StateStore.open_project`
(`tcip_web/state.py:101`) each time a project is opened and by the MCP `view_gui_state` tool; a
snapshot that does not decode as its whole shape raises.

`StateStore.mutate` validates the merged mutation through `GuiState` before holding it, raising
`GuiMutationInvalid` (`tcip_web/state.py:25`) on a field that does not validate or a key
`GuiState` does not declare; `app.py`'s `_gui_mutation_invalid_handler` answers a route that
raises it with 400 and the validation message rather than the 500 an unhandled `ValueError` would
produce.

Seam S10 ("Live GUI state .tcip/state/gui.json"): one writer and one reader, both in
`tcip_mcp/web_client.py`; `tests/test_active_context.py` persists through the backend's own
`StateStore` and reads the snapshot back through `view_gui_state` and a reopened store.

## 20. `.tcip/state/project_status.json`, per-project activity pointer

Path: `<project_path>/.tcip/state/project_status.json`, addressed by `project_status_key`,
`packages/tcip-mcp/src/tcip_mcp/project_status.py:47`, on the store `PROJECT_STATUS_STORE`,
`project_status.py:32`; `project_status_path`, `project_status.py:52`, is the same address as a
path for a caller that needs one.

Writers: `record_report`, `project_status.py:109`; `record_retrospective`,
`project_status.py:125`; `record_distillation`, `project_status.py:145`; all via the shared
locked read-modify-write `_update`, `project_status.py:78`.

Reader: `read_project_status`, `project_status.py:58`.

No seam id in `seam-coverage.json`'s 67-entry inventory names `project_status.json`.

## 21. `.tcip/state/review/*.jsonl`, verdict shard log

Path: `<state_dir>/review/<bucket>/<img_name>.jsonl`, the directory and suffix spelled once in
`tcip_store.layout_claims` (`REVIEW_SHARD_DIRNAME`, `REVIEW_SHARD_SUFFIX`). The store is keyed
`("bucket", "image")`, the bucket being the prediction bucket's path relative to its dataset root
as `bucket_key_of` spells it, each part folded by `verdicts.verdict_key` so the shard's own path
spells its key back. Real-world `state_dir` is `<dataset_root>/.tcip/state`, derived once by
`project_state_dir`, `packages/tcip-mcp/src/tcip_mcp/project_paths.py:9`.

Each entry is one decision, `{proposal, action, by, at}`: the proposal's index in the bucket's
document for the image, `accepted` or `rejected`, the person in the label record's own spelling,
and the time. Writer: `verdicts.record_verdicts`, called only by
`dataset_layout.save_label_document` when a save accepts or rejects a proposal, each entry
through `verdicts.encode_verdict`. Reader: `verdicts.read_verdicts`, every entry through
`verdicts.decode_verdict`, which refuses a malformed entry by name; the editor's proposals route
reads each proposal's last decision through it, and `tcip doctor`'s `check_state` reports a
shard that will not read.

Guard: `tests/test_label_document_gestures.py` (each decision recorded once, a malformed entry
refused); `tests/test_store_contract.py` (the shard's key reads back on both backends, a cleared
log enumerates as absent).

## 22. `.tcip/datasets.json`, project-level dataset identity registry

Path: `<project_root>/.tcip/datasets.json`.

Writer: `upsert_dataset`, `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:80`.

Reader: `read_datasets`, `project_tools.py:64`.

Shape: each entry's `path` is relative to `<project_root>` whenever the dataset sits under it by
filesystem identity (`registry_path_for`, `project_tools.py`), the project's own tree becoming
`"."`; absolute for a dataset outside it. Resolved on read through the one accessor,
`dataset_entry_path`, `project_tools.py`, which every consumer of an entry's path calls rather
than reading `path` off the entry directly. A project registered before this row was corrected
by a one-off operator script, since applied to every real project and retired; there is no
runtime migration.

No seam id in `seam-coverage.json`'s 67-entry inventory names `.tcip/datasets.json` (distinct from
`dataset.json`, format 4, which S26 covers).

## 23. Workspace last-opened pointer (`last_opened.txt`)

Path: `<workspace_root>/.tcip/state/last_opened.txt`, addressed by `last_opened_key`,
`packages/tcip-mcp/src/tcip_mcp/workspace.py:74`, on the store `workspace_last_opened`
(`workspace.py:58`): one project id and a newline.

Writer: `write_last_opened`, `workspace.py:86`, called by the web backend's `open_project_by_id`
(`packages/tcip-web/src/tcip_web/routes/projects.py:110`) on every open.

Readers: `read_last_opened`, `workspace.py:80`, read by the backend's lifespan through
`open_last_opened` (`routes/projects.py:120`), which opens the project whose record holds that id
(`project_by_id`, `workspace.py:99`); when none does it opens nothing, and the project list
names the missing id (`last_opened_problem`). The
pointer is a preference, never the MCP server's project: each server is started for one project
by `--project`.

## 24. Formats named but not exhaustively enumerated in phase0

`classifier_operating_point.json`, `ordinal_operating_point.json`, `regression_operating_point.json`
(sibling sidecars to `operating_point.json`, named in `pipelines/operating_point.py`, not opened
for this section); any format defined inside `pipelines/postprocessing/` (`selection.json` is
covered in format 26). No seam id covers this placeholder entry since
it names no single format.

## 25. `.tcip/project.json`, per-project record (id, display name, site)

Path: `<project_path>/.tcip/project.json`, addressed by `project_record_key`,
`packages/tcip-mcp/src/tcip_mcp/project_record.py:66`, on the store `PROJECT_RECORD_STORE`,
`project_record.py:29`; `project_record_path`, `project_record.py:71`, is the same address as a
path for a caller that needs one. It holds `id` (minted once at creation by `mint_id`,
`project_record.py:61`), `display_name` and `site`; the id is the project's identity, never its
directory name.

Writers: `create_record`, `project_record.py:148`, a create-only write: an absent record is
written, a present record with the same display name and site is left as is, and one differing
raises. `rename_project`, `project_record.py:201`, replaces the display name alone; `replace_site`,
`project_record.py:194`, behind `tcip write-project-site --replace`, the site alone. Each keeps the
id.

Readers: `read_record`, `project_record.py:105`; `record_fields`, `project_record.py:219`, is the
one reader every surface (the picker, `inspect_project`, the doctor, `project_by_id`) calls, and
never raises.

No seam id in `seam-coverage.json`'s 67-entry inventory names `project.json`: the record is new.

## 26. `selection.json`, a partition `draw_splits` drew

Path: `<output_path>/selection.json`, addressed by `selection_key`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/selection.py:314`, under whatever directory the
caller asked the partition to be written to; no dataset resolver owns this layout.

Writer: `draw_splits`, `data_tools.py:228`, when `output_path` is given. A finished run's own
drawn train/val partition freezes into the identical shape through `freeze_selection`
(`data_tools.py:18`), the second writer; both go through the one `write_selection`
(`selection.py:431`), which composes the document through `selection_document`
(`selection.py:345`) and refuses a crossing partition before anything lands, so the two writers
can never disagree on what a selection carries or write one a reader would reject. A frozen
selection also carries `origin` (`{"experiment_id", "frozen_at"}`), absent on a drawn one, the
field `read_selection` and its callers use to tell the two apart.

The record is its sample list. Each sample carries `source` (the image path, the `.bandgroup`
manifest standing in for a grouped capture, or the raster path when the sample is a region),
`ground_truth` (the path to whatever answers for it, never derived from `source`; for a label
document, also the document whose completion marks admitted it), `group` (the key that keeps
related samples together), `side` (one of `selection.SIDES`: `train`, `val`, `calibration`), and
optionally `rect` (a half-open pixel rect
for a within-image draw), `row_key` (the row inside a tabular ground truth) and
`ground_truth_digest` (that file's digest at draw time). `rect` and `row_key` are refused by
`selection.refuse_unreadable_samples` wherever a loader would otherwise read past them. Nothing in the record names a capture date or a directory scope:
every sample names its own paths, so one selection spans as many dates as the draw admitted, and
two dates holding a same-named image are two samples rather than one identity that has to be told
apart from itself. Beside the samples the record carries `subject`, `attribute` (`null` when
none), `id_map` (the `assign_class_ids` map the admission resolved), `seed`, `group_by` (the
resolved policy), `dataset_fingerprint`, `admission_counts` (the summed admission counts;
a frozen selection records `{}`, since freezing a training run's own drawn partition records no
admission draw), `realized_ratios` and `origin`.

A side whose ratio is zero is not drawn, written or not, and a negative ratio refuses. The draw
refuses, before any write, when the tree holds fewer foreground groups of `subject` (and
`attribute`, when scoped) than one per side it draws, counted through a subject-scoped `count_label_lines` on every draw,
independent of `stratify_foreground` (which only gates the balancing pass's own foreground
signal). `realized_ratios` records each side's delivered member share: at a floor-sized tree the
minimum pass can consume every foreground group before the balancing pass ever sees the caller's
fractions, so the delivered shares can diverge from the ratios asked for. The answer also carries
`calibration_foreground_groups`, how many of the calibration side's own groups carry foreground
at all, since the floor above is over the whole draw and the calibration slice can still land
short of two; the calibration door's own floor is where that absence bites. A stats-only call (no
`output_path`) is the same draw with the same answer, written nowhere.

Readers: `selection.read_selection` (`selection.py:446`), the one reader a training or tuning
run's `data.split.selection_dir` resolves through (`split_construction.auto_train_val`) and a
selection-restricted calibration resolves through
(`splits.selection_calibration_universe`), which refuses by name when the record is
absent, undecodable, not a mapping, lists no samples, holds a sample missing any of
`source`/`label`/`group`/`side`, names a side outside `selection.SIDES`, carries a malformed
`rect`, or holds a partition whose sides cross. That last check is `refuse_crossing_sides`
(`selection.py:265`), the same one the writer runs: one source identity on two sides is the same
pixels trained on and selected on, and one group key on two sides splits the crops of one parent
across sides. `tcip plant-aware-group-splits` reads no selection back, it only writes one through
`draw_splits`.

`selection.read_selection_checked` (`selection.py:460`) is the checked variant a listing calls in
place of the raising reader: absence answers `(None, None)`, a record that exists but will not
decode, fails a shape check, or is version-refused answers `(None, text)`, catching
`tcip_store.SchemaVersionRefused` beside the plain-shape `ValueError` for that purpose only,
since a version refusal must never read as an ordinary absence.
`training_tools.selection_compatibility` is every objection a launch binding one config to one
selection would raise, checked ahead of that launch: composed from the config-only conflict and
task checks (computed before any read, so an unreadable selection never suppresses them) and the
selection-dependent checks (a selection recording no subject, an empty train or val side, and a
config stating a scope of its own that disagrees with the selection's, which calls the bind's own
`_refuse_scope_disagreement` rather than restating it). A bound run does not restate its scope at
all in the ordinary case: it reads subject, attribute and class map off the selection. `preflight_config` calls both halves directly, in the same order, over a selection it
read itself; `training_tools.list_split_choices`, the relaunch data picker's own reader wrapped by
`GET /api/training/configs/{experiment_id}/splits`, calls the composed function per candidate
selection it read through the checked variant above, and builds each candidate's launch config
through `training_tools.candidate_config_with_selection`, the same function the relaunch route's
own launch build calls.

No seam id in `seam-coverage.json`'s inventory names this record: it is new, and
`tests/test_selection_binding.py` calls the real writer and the real consumer
(`auto_train_val`'s selection branch) against the same files.

## 27. `assessment.json`, `.tcip/assessments/<assessment_id>/`

Path: one directory per assessment under the project's `.tcip/assessments/`, holding
`assessment.json` and a copy of every reference ground-truth file it read under `reference/`
(`packages/tcip-mcp/src/tcip_mcp/assessment.py`).

Writer: `assessment.assess`, `assessment.assess_reserved_regions` and
`assessment.assess_physical_scale`, each finishing through `assessment._finish`, one write-once
record and one audit line; nothing rewrites an assessment. The record states the trait revision
(`TraitRevision.ref`), the delivery kind, the producer, the execution record, the reference (each
sample's source digest and ground-truth digest, the retained copies, the captures or raster it
covers), the disjointness evidence, the criterion's evidence, the scale (physical-scale only), the
failures and whether it passed.

Readers: `assessment.read_assessment`, the one decoder into a typed `Assessment`, by
`delivery.gate` for every delivered bucket naming one and for a delivery's scale assessment, by
`run_inference` restoring the execution record an assessment measured, and by the assessment
doors answering with it.

No seam id in `seam-coverage.json`'s inventory names this record.

## Formats with a general path-resolution seam but no per-format seam entry above

Seam S14 ("dataset_layout.py as the on-disk path resolver"), verdict `both-sides-restated`,
`phase0_implementation: once, shared`, and seam S15 ("Per-image label filename convention"),
verdict `one-side-only`, `phase0_implementation: written twice`, both bear on how formats 1-5's
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
on at HEAD `f943c12d`, which `tools/check_architecture_citations.py` holds to the tree. Where
a citation carried in the Phase 0 record no longer names the exact line of the symbol it
describes, the corrected HEAD line is given and the drift is noted.

The "Phase 3 verdict" column is `single_implementation.verdict` from the batch B01-B12
adjudication files, a semantic re-check of whether the two sides actually stay in
agreement, distinct from Phase 0's own structural "Implementation" label (`once,
shared` / `written twice` / `mixed`), which described only whether the code was
physically written once or twice, not whether that code enforces agreement in
practice. The two fields disagree for several seams (S14, S17, S19, S20, S26, S32,  <!-- queued: P5-275 unify -->
S33, S40, S59, S66): Phase 0 recorded a single shared implementation at the  <!-- queued: P5-294 unify -->
structural level, which the Phase 3 adjudication accepts only when no second,
unshared restatement of the same fact exists elsewhere, whatever one function is  <!-- queued: P5-293 unify -->
the primary implementation. Both fields are
reported as-is below; the Phase 3 verdict is the one used for the seam count at the
end.

## S03. Backend port discovery file .tcip/state/web_port.txt

Must agree: the MCP process finds the port the web backend actually bound.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:27` (`BACKEND_PORT_STORE`, declared beside the reader because the reader cannot import `tcip_web`; `backend_port_key`, `web_client.py:42`, is the one address, read at `web_client.py:342`).
Side B: `packages/tcip-web/src/tcip_web/__main__.py:44` (`replace(backend_port_key(workspace), str(port))`, publishing through that same key under the workspace `main` resolved once, and raising rather than swallowing a failure, since the fallback silently misses an OS-picked port).
Phase 3 verdict: single.

## S04. Panel-event panel vocabulary (VALID_PANELS)  <!-- queued: P5-324 unify -->

Must agree: sender and receiver accept the same set of panel names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:301` (`VALID_PANELS = frozenset(`).
Side B: `packages/tcip-web/src/tcip_web/app.py:25` (`VALID_PANELS,`).
Phase 3 verdict: duplicated.

## S05. Panel event_type vocabulary  <!-- queued: P5-272 unify -->

Must agree: the Python poster, the FastAPI hub, and the browser handler use the same event_type strings.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py:160` (`result = post_panel_event(project, workspace, "app", PANEL_EVENT_ANNOTATE_FOCUS, payload)`).
Side B: `packages/tcip-web/src/tcip_web/app.py:312` (`if event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:`).
Phase 3 verdict: single. The posted payload carries `subject` beside `active_subject` (`packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py:157`), the key both readers take (`packages/tcip-web/src/tcip_web/app.py:316`, `frontend/src/lib/annotateFocus.ts`), held by `tests/test_event_integration.py`'s producer-driven test, which posts the focus_human_attention tool's own event and asserts the advisory state's `active_subject`.

## S06. `audit_log`, one append-only store under two kinds of root

Must agree: mutations from any process land in the log the scope names, a dataset's own for a
record traveling with the data and the project's own otherwise, all with the same entry shape;
and the project's own receipt gate (`plant_mapping.load_mapping`) trusts only what that project's
own log actually recorded.
Side A: `packages/tcip-mcp/src/tcip_mcp/audit.py:186` (`def audited(`, taking a declared
`scope_arg` naming which tool argument carries the dataset a scoped tool mutates a record of) and
`record_event` (`audit.py:113`)/`record_event_or_raise` (`audit.py:131`), the two emitters for code
that is neither an MCP tool nor a script-invoked door demoted from one; all three append at the
one `audit_log_key`, `audit.py:74`, and differ only in what a failed append means: `record_event`
warns; `record_event_or_raise` raises `AuditEntryNotWritten`; the decorator refuses
(`MutationCommittedWithoutAuditLine`), since its append runs after the tool body.
Side B: the label save and the registry write record through their library
(`save_label_document`, `replace_registry`) whichever door calls them, a failed append raising
`AuditEntryNotWritten`, which the GUI route answers as `routes/audit_gap.py`'s 409; the GUI
inference worker writes no line of its own and publishes through the one publisher,
`routes/inference.py:139` (`bucket = publish(`),
whose receipts are the library's, catching `AuditEntryNotWritten` into the job's
`audit_warning`. Reader:
`pipelines/postprocessing/plant_mapping.py:1511` (`_require_receipt`)
trusts only a `plant_mapping_built` entry it finds in the log under the project its caller names
(the MCP server's started project; the web backend's open project), scanned by `_scan_receipts`
(`pipelines/postprocessing/plant_mapping.py:1451`), which refuses (never scans past) a page
reporting corruption or an unknown `schema_version`.
Phase 3 verdict: single. Each writer is exercised through a real append and checked for its own
tool name landing in the log its own scope names: `tests/test_tcip_web_routes.py`'s
`test_annotate_save_audits_into_the_log_of_the_dataset_it_wrote` (a dataset-scoped GUI write,
checked against the same dataset's log, never the platform's);
`tests/test_tcip_web_results_routes.py:680,701` (a dataset-scoped delivery-binding event beside a
project-scoped export audit line, from the one route); `tests/test_tcip_web_subjects_routes.py`'s
`test_a_registry_save_leaves_one_library_line_through_either_door` (a dataset-scoped registry
write, checked against the project log for the same request); and
`tests/test_audit_row_core_field_agreement.py:26`, which runs a real GUI route and a real
platform write against one log and holds the two rows to the same core fields. The receipt
gate's own agreement is held by `tests/test_plant_mapping.py:293,326` (a version-refused line
still blocks the scan; a real receipt still admits) and
`tests/test_plant_mapping_binding.py:661,709` (a receipt that cannot be written fails the build
and the web route alike).

## S07. Experiment record .tcip/experiments/<id>/

Must agree: the launching process and the run's own child agree on the run directory's layout, and each file is written by one of them once.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:122` (`def experiment_dir(` plus the file names beside it, the one declaration of the directory's path and members).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py:68` (`def run_directory(`, the child reading the launch record `open_run` wrote and writing the resolved record and final status beside it).
Phase 3 verdict: single.

## S08. metrics.jsonl row format

Must agree: the writer's row shape is what the reader and the stream consumer expect.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:223` (`def _epoch_sink(`, the one writer; the trainer and a bespoke loop hand rows to it).
Side B: `packages/tcip-web/src/tcip_web/routes/training.py:255` (`rows, cursor = await asyncio.to_thread(read_rows, observation.metrics_log, after=cursor)`, the training stream's incremental tail read off the event loop, pushed as a `TrainingMetricFrame` per row) and `routes/tuning.py:177` (`def get_trial_metrics(`, reading a trial's log through `read_rows`, answered in the shape `_metrics_common.metrics_response` builds).
Phase 3 verdict: single. An HPO trial is a run directory, so its log is the same file shape written by the same sink.

## S10. Live GUI state .tcip/state/gui.json  <!-- queued: P5-284 unify -->

Must agree: the MCP agent reading GUI context parses the snapshot the web backend wrote.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:266` (`def write_gui_snapshot(`, the one writer, through the one address `gui_snapshot_key`, which `StateStore.mutate` in `tcip_web/state.py` calls).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:213` (`state = read_gui_snapshot(project)`, the one reader, which the backend's own `open_project` also calls).
Phase 3 verdict: single.

## S11. Live canvas state files canvas_live.json / canvas_shapes.json, under the backend's open project

Must agree: the push route writes canvas_live.json/canvas_shapes.json under the project the backend has open, and capture_live_canvas reads them from the project its MCP server was started for, so a capture reads the GUI's live canvas only while the two are the same project. Both sides address the documents through one locator pair (`canvas_meta_key`, `packages/tcip-mcp/src/tcip_mcp/web_client.py:96`; `canvas_geometry_key`, `web_client.py:101`). The push carries the project id the browser drew for (`packages/tcip-web/src/tcip_web/routes/canvas.py:57`, `def push_canvas_state(`), compared with the open project's own id (`StateStore.project_id`, `packages/tcip-web/src/tcip_web/state.py:108`) before anything is written, and a panel event from the MCP side carries its project's id the same way (`web_client.py:286`), delivered only when that project is open.
Phase 3 verdict: single.

## S12. Friction reports and retrospectives under .tcip/

Must agree: the GUI reader finds, decodes and orders what the MCP writer produced.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py:151` (`def report_documents(`, the one enumeration, decode and ordering of the friction reports, with `retrospective_documents`, `meta_tools.py:184`, doing the same for the retrospectives). Both stores are records, enumerated through the seam and ordered by the timestamp each document states (a report's own `timestamp` field, a retrospective's own `## Retrospective:` section headers), never by when the bytes landed. A report is one whole JSON document, not a line of a stream.
Side B: `packages/tcip-web/src/tcip_web/routes/meta.py:20` (`get_reports`) and `routes/meta.py:42` (`get_retrospectives`), both routes importing those MCP-side enumerators directly and presenting the rows they return, rather than walking a directory of their own.
Phase 3 verdict: single.

## S13. Session telemetry's negative-confirmation time

Must agree: the time a session spent confirming a negative is classified by the same reading of
a negative the admission trains on.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:572` (`def state(self, subject: str) -> SubjectState:`, the one reading of a label document's completion).
Side B: `packages/tcip-web/src/tcip_web/routes/sessions.py:224` (`return read_label_document(label).state(subject) == "negative"`).
Phase 3 verdict: single.

## S14. dataset_layout.py as the on-disk path resolver

Must agree: agent writes and GUI reads resolve to the same files.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:141` (`def image_root(`, with `annotation_root`/`prediction_root` and the dated dir calls built on them).
Side B: `packages/tcip-web/src/tcip_web/routes/dataset.py` (`select_dataset` resolves every directory through the resolver; `tcip_mcp.cli.doctor`, `data_tools`, `project_tools` and `annotation_tools` no longer re-spell the tree).
Phase 3 verdict: single.

## S15. Per-image label filename convention

Must agree: the browser's label path and the Python resolver's label path name the same file.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:387` (`def label_filename(`, with `annotation_path`/`prediction_path` built on it).
Side B: `packages/tcip-web/frontend/src/lib/paths.ts:24` (`labelPath`, the browser's one join site over the directories the backend resolves; a gate test pins the record extension against the resolver).
Phase 3 verdict: single. The browser still joins directory plus filename client-side at that one site; handing fully resolved per-image paths across the API would add a backend round trip to image navigation, an open owner question in the batch report.

## S16. Verdict shard store directory

Must agree: the verdict writer and every verdict reader look at the same shard store (a bucket
here is format 17's prediction directory, not a score bin).
Side A: `packages/tcip-mcp/src/tcip_mcp/project_paths.py:9` (`def project_state_dir(`, the one derivation of the store root).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:432` (`def verdict_key_of(`, the one address the label save writes and the editor's proposals route reads, inside the layout `tcip_annotation.verdicts` owns).
Phase 3 verdict: single.

## S17. Canonical per-image annotation JSON schema

Must agree: every writer produces, and every reader accepts, the same name-based record shape.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:328` (`def annotation_from_payload(`, the one conversion from a client payload to a record, with `json_io.py:423` (`_annotations_of`) the one parse back and `json_io.py:894` (`write_annotations`) the one writer).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:519` (`contents.append(annotation_from_payload(payload))`, inside the one label save every door calls rather than assembling records of its own).
Phase 3 verdict: single.

## S18. Bounding-box coordinate convention across the HTTP boundary

Must agree: the browser and the route use corner coordinates while the file uses xywh, with the conversion happening once.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:50` (`bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel`, the wire form).
Side B: `packages/tcip-annotation/src/tcip_annotation/json_io.py:768` (`def xywh(`, the one corner-to-xywh conversion and the 2-decimal grid the stored document lives on, applied on write and, via the import at `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py:26` (`from tcip_annotation.json_io import xywh`), to every box scored against a stored label so both sides of a match sit on one grid; a wire box becomes a `BBox` in `annotation_from_payload` (`json_io.py:328`), and `_annotations_of` (`json_io.py:423`) is the inverse read).
Phase 3 verdict: single.

## S19. Annotation format detection scope (json, coco)

Must agree: a label document's shape is decided by its one reader, never guessed beside it.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:140` (`def parse_label_document(text: str, *, source: str) -> dict:`), which refuses a dataset-level COCO for every reader of a per-image document.
Side B: none; admission, the doctor and the read tool take that reader's answer.
Phase 3 verdict: single.

## S20. subjects.json subject registry

Must agree: the GUI editor, the path resolver, and the training loader read one registry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py:4` (`The on-disk registry (`` `<dataset_root>/subjects.json` ``) is self-describing and name-based::`).
Side B: `packages/tcip-web/src/tcip_web/routes/subjects.py:81` (`registry, version = read_versioned_registry(allowed_path(dataset_root))`).
Phase 3 verdict: single.

## S21. Training name-to-id assignment versus inference decode map

Must agree: a prediction's integer label decodes to the class name the run trained it as.
Side A: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py:319` (`def assign_class_ids(`, the one assignment, reached by the loader through `pipelines/data/label_queries.py:55` (`return registry, subject_registry.assign_class_ids(registry, subject, attribute)`)).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py:229` (`scope=ClassScope.of(checkpoint.data_config)`, the checkpoint's own recorded `scope`, map included, taken by the prepared pass every inference regime, the GUI worker and every assessment share; the reserved-regions assessment reads it off that pass at `assessment.py:603` (`scope = p.scope.admitted_for(DOCUMENT`); nothing re-derives a map from a live registry).
Phase 3 verdict: single.

## S22. Completion marks in the label document

Must agree: a negative is an empty subject plus a person's mark, every consumer reads the same marks the same way, and each mark carries who made it and when.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:505` (`def completion_marks(`, the one decoder of a document's marks, keeping only those whose digest still names their subject's annotations).
Side B: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:548` (`marks = {s: held for s, held in stored.marks.items() if gestures.complete.get(s, True)}`, the one writer's carry-over through the label save); the browser receives each subject's derived state from the load route and restates no predicate, its `SubjectState` type generated from the Python `Literal`.
Phase 3 verdict: single.

## S25. Region completeness on an orthomosaic

Must agree: a region a person marks complete is the region the calibration gate reads as attested.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:551` (`rect=gestures.rect or (0, 0, width, height), by=cast(str, author), at=now,`, the mark's rect as the save records it).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py:100` (`uncovered = sorted(name for name, rect in rects.items() if not covers(marks, rect))`).
Phase 3 verdict: single.

## S26. dataset.json identity and fingerprint

Must agree: the stored fingerprint and the recomputed one cover the same inputs.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:308` (`def dataset_identity_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py:109` (`def dataset_fingerprint(dataset_root: str | Path) -> str | None:`).
Phase 3 verdict: single.

## S27. Trained-model registry .tcip/models/registry.json

Must agree: the MCP registrar and the GUI model pickers read one registry entry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/model_registry.py:121` (`def registered_entries(`, the one read every consumer goes through: one owner per sha256, the completed run whose final status names the bytes, else the index's foreign entry; `_write_registry_entry`, `model_registry.py:295`, replaces one foreign entry by sha256 inside one `tcip_store.transaction` on the key `registry_index_key`, `model_registry.py:85`, mints).
Side B: `packages/tcip-web/src/tcip_web/routes/results.py:577` (`@router.get("/models/registered")`, serving `model_tools.rank_registered_models`'s listing view) and the browser's one entry declaration, `packages/tcip-web/frontend/src/api/inference.ts:29` (`export interface RegisteredModel {`), held field by field against an entry the real registrar wrote by `tests/test_registry_entry_shape_agreement.py`.
Phase 3 verdict: single.

## S28. bucket.json prediction-bucket record

Must agree: every publisher writes, and every consumer reads, the same record next to a bucket's predictions.
Side A: `packages/tcip-mcp/src/tcip_mcp/buckets.py:189` (`def publish(`, the one writer, called through `inference_tools.infer` by `run_inference` for an image directory and a raster alike and by the GUI's inference worker, `routes/inference.py:108` (`result = infer(`), and by `stage_proposals` for a staged proposal).
Side B: `packages/tcip-mcp/src/tcip_mcp/buckets.py:95` (`def read_bucket(`, the one decoder every reader calls).
Phase 3 verdict: single.

## S29. Prediction-bucket immutability

Must agree: no writer writes into a bucket that already exists.
Side A: `packages/tcip-mcp/src/tcip_mcp/buckets.py:189` (`def publish(`, refusing an existing directory through `create_run_directory` before it consumes a document).
Side B: every writer of a bucket, the inference operation and the proposal staging alike, publishes through it; a raster pass keeps its resume progress under the project, apart from the bucket it will publish.
Phase 3 verdict: single.

## S30. split.json train/val partition

Must agree: an assessment's reference is disjoint from the split the producing run actually
trained and selected on.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:414` (`def run_resolution(`, the one reader of the partition the launcher resolved and wrote into the run's `run.json`).
Side B: `assessment.py`'s `_disjointness`, a set intersection over group keys and source digests between the reference sides and the run's train and val sides read through it, and the reserved-regions assessment, which takes its regions from the same reader.
Phase 3 verdict: single.

## S31. Checkpoint payload structural markers

Must agree: a checkpoint written by the training envelope is kind-routable and rebuildable by the predictor that later loads it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:24` (`MODEL_SOURCE_KEY` and `STATE_DICT_KEY`, the one key vocabulary).
Side B: the three checkpoint writers in `pipelines/training/generic_trainer.py` and the readers (`generic_predictor.py`, `inference/predictor.py`, `training/eval_runners.py`) all bind through the constants.
Phase 3 verdict: single.

## S32. One execution record for every pass

Must agree: the same model and stated values yield the same conf, cap, tile edge, overlap and merge whichever entry point asks for them, and the record executed is the record stamped.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py:176` (`def prepare_pass(`, preparing a record from the one `Stated` mapping a caller hands it or restoring one exactly, refusing a stated value the restored record differs on).
Side B: every pass prepares through it: `tools/inference_tools.py:118`, `packages/tcip-web/src/tcip_web/routes/inference.py:111`, `assessment.py:207` (both assessment kinds, through `_prepared`), and `pipelines/training/eval_runners.py:119` (the full-frame regime); `bucket.json` and `assessment.json` both carry `Execution.record()`.
Phase 3 verdict: single.

## S33. Shared inference defaults DEFAULT_CONF / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS

Must agree: the MCP entry point and the GUI entry point start from the same unresolved defaults, and both read a caller's unstated parameter off the `None` sentinel rather than off equality with the default, so a caller who states the default value is honored as an override instead of being resolved as if they had stated nothing.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/execution.py:25` (`DEFAULT_CONF = 0.5`, with `DEFAULT_NMS_IOU`, `DEFAULT_OVERLAP` and `DEFAULT_MAX_DETS` declared beside it).
Side B: `packages/tcip-web/src/tcip_web/routes/inference.py:179` (`stated=payload.stated`: the GUI launch carries the one `Stated` mapping unresolved, `None` where omitted, and its worker resolves it through `prepare_pass`, which records each value's source, `explicit` or `default`, on the execution record) and `packages/tcip-mcp/src/tcip_mcp/pipelines/training/eval_runners.py:66` (the tile-level regime resolving its conf and cap through `untiled_execution`).
Phase 3 verdict: single.

## S34. One delivery gate behind every delivery path

Must agree: no delivered result ships unvalidated without an explicit acknowledgment, and every delivery's validated column, revision and event id come from one clearance.
Side A: `packages/tcip-mcp/src/tcip_mcp/delivery.py:210` (`def gate(`, clearing every bucket against the assessment it names: passed, of this delivery's kind, under this revision, its reference unmoved, produced by the bucket's own checkpoint and execution record, covering its capture; refusing no bucket, differing producers, and a detector delivery whose buckets do not count the measured subject).
Side B: the three delivery functions, each calling it once and writing its rows and its one event through `delivery.py:390` (`def deliver_csv(`): `pipelines/postprocessing/export.py:176` (`def deliver_per_image_counts_csv(`), `pipelines/postprocessing/aggregation.py:219` (`def deliver_per_plant_aggregate(`, which the orthomosaic plant-count door also delivers through), and `pipelines/postprocessing/phenology.py:462` (`def deliver_phenology(`).
Phase 3 verdict: single.

## S36. Count-objective vocabulary versus registered pickers

Must agree: every named count objective has a registered picker function.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:27` (`COUNT_UNBIASED = "count_unbiased"`, with `DETECTION_F1` and `PRESENCE` declared beside it).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py:27` (`COUNT_OBJECTIVE_PICKERS`, keyed by those three names; an objective with no picker refuses in the count criterion). A picker's provenance label is read off that registry by `pipelines/derivations.py:647`, so registering a picker registers its label.
Phase 3 verdict: single.

## S37. Trait entries against crops.yml controlled vocabulary

Must agree: a trait entry's delivered phenotypes exist in the crops.yml vocabulary.
Side A: `packages/tcip-mcp/src/tcip_mcp/knowledge/__init__.py:124` (`def crops_yml_path(`, the one placement of `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml`), reached through `packages/tcip-mcp/src/tcip_mcp/traits.py:100` (`def crops_yml_path(`, delegating), read whole or raising by `_crops_traits`, `traits.py:107`.
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:276` (`def check_proposed_entry(`, a proposal's one check of `delivers` against that read; a stored record decodes without it) and `tools/verify_skill_traits.py:26` (`load_vocab` checks a skill's trait tokens through that same read).
Phase 3 verdict: single.

## S38. Per-project trait records .tcip/state/traits/*.json

Must agree: the proposing tool, the confirmation door, the delivery doors and the GUI trait list read one record per trait and agree on which revision a delivery ships under.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:388` (`def trait_key(`, the one placement, with `TRAITS_STORE`, `traits.py:373`, the store every reader and writer addresses).
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:458` (`def propose_trait(`, the one append) and `packages/tcip-mcp/src/tcip_mcp/operationalization.py:46` (`def confirmed_revision(`, the one read of the latest confirmed revision every delivery door makes). `packages/tcip-web/src/tcip_web/routes/results.py:501` (`def list_traits(`) and `packages/tcip-mcp/src/tcip_mcp/cli/doctor.py:264` (`def check_traits(`) read the same record through `read_trait`.
Phase 3 verdict: single.

## S39. Phenology CSV column vocabulary

Must agree: the delivered CSV's column names derive from the trait spec on every path that writes them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:66` (`def phenology_csv_columns(spec) -> list[str]:`, the one owner).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:66` (`phenology_csv_columns(spec)`, inside `deliver_phenology`, the one writer both `tools/phenology_tools.py`'s `deliver_phenology_milestones` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv` call through instead of assembling the names themselves).
Phase 3 verdict: single.

## S40. Per-band normalization stats for a non-3-channel detector

Must agree: the values passed as image_mean/image_std are per-band stats of the same length as in_chans.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py:408` (`def band_normalization_stats(`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py:64` (`def _normalization(adapter: Any, in_chans: int | None, image_mean, image_std,`).
Phase 3 verdict: single.

## S41. model_source bespoke build seam  <!-- queued: P5-320 unify -->

Must agree: the dict an agent writes into the config carries the keys the builder, the snapshotter, and the predictor all read.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:24` (`MODEL_SOURCE_KEY = "model_source"`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py:75` (`self.model_source = ckpt.get("model_source")`).
Phase 3 verdict: duplicated.

## S42. training_source bespoke train(ctx) seam  <!-- queued: P5-321 unify -->

Must agree: a bespoke train(ctx) callable is importable and accepts the TrainContext the envelope hands it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:341` (`training_source = run.config.get("training_source")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:279` (`training_source = normalized.get(TRAINING_SOURCE_KEY)`).
Phase 3 verdict: duplicated.

## S43. dataset_source bespoke dataset seam

Must agree: the builder the reader resolves off `data.dataset_source` returns a Dataset the trainer's loaders accept.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:324` (`dataset_source = (config.get("data") or {}).get(DATASET_SOURCE_KEY)`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:983` (`def build_dataset(`).
Phase 3 verdict: duplicated.

## S44. Model-contract smoke batch versus the trainer's real batch

Must agree: the smoke batch has the same shape the trainer actually feeds model.forward for the task.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py:72` (`def _synth_batch(`, which synthesizes per-sample `(image, target)` items shaped like a dataset's `__getitem__` and hands them to the trainer's own collate).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/collation.py:35` (`def task_collate(task: str):`, the collate the DataLoader assembles the training batch with).
Phase 3 verdict: single.
Differs from phase0 record: phase0 cited a line inside the function's body rather than its header; the function is defined at `model_contract.py:72` (`def _synth_batch(`).

## S45. Verdict entry encoding

Must agree: the decision the label save records is the decision every reader decodes.
Side A: `packages/tcip-annotation/src/tcip_annotation/verdicts.py:34` (`def encode_verdict(verdict: Verdict) -> dict:`, the one encoder, called by `record_verdicts` inside the label save).
Side B: `packages/tcip-annotation/src/tcip_annotation/verdicts.py:39` (`def decode_verdict(entry: Mapping) -> Verdict:`, the one decoder, refusing a malformed entry by name), read by the editor's proposals route and the doctor.
Phase 3 verdict: single.

## S46. Frontend api/ layer against backend route paths

Must agree: every URL the browser builds matches a registered FastAPI route path and method.
Side A: `packages/tcip-web/frontend/src/api/routes.ts` (generated: the browser's only copy of the paths, each named for its method).
Side B: `packages/tcip-web/src/tcip_web/routes/__init__.py` (`register_all` mounts 14 routers with fixed prefixes).
Phase 3 verdict: single. The api/ helpers keep their hand-written signatures and reference a generated name; `tools/generate_frontend_routes.py` projects the registered routes into that module, and `tests/test_frontend_route_paths.py` fails when the projection is stale or a call site writes a path of its own.

## S47. GuiState shape between state.py and store/types.ts  <!-- queued: P5-287 unify -->

Must agree: the snapshot the backend serializes deserializes into the store's typed shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:183` (`class GuiState(_GuiFields):`).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts:362` (`export interface GuiState {`), generated from Side A by `tools/generate_frontend_types.py` and re-exported by `store/types.ts`.
Phase 3 verdict: single.

## S48. State WebSocket snapshot protocol  <!-- queued: P5-288 unify -->

Must agree: the browser knows which slices of a broadcast snapshot are backend-authoritative and orders them by version.
Side A: `packages/tcip-web/src/tcip_web/app.py:165` (`@app.websocket("/ws/state")`).
Side B: `packages/tcip-web/src/tcip_web/state.py:157` (`def version(self) -> int:`, "Monotonic version, bumped on every state change.").
Phase 3 verdict: duplicated.

## S49. Terminal PTY WebSocket protocol  <!-- queued: P5-289 unify -->

Must agree: control-message type names and field names match, and output frames are treated as raw text rather than JSON.
Side A: `packages/tcip-web/src/tcip_web/routes/terminal.py:476` (`@router.websocket("/ws/{session_id}")`).
Side B: `packages/tcip-web/frontend/src/components/TerminalRail.tsx:302` (`send({ type: "input", data });`).
Phase 3 verdict: duplicated.

## S50. Inference job stream WebSocket  <!-- queued: P5-304 unify -->

Must agree: the browser recognizes the terminal frame and the status vocabulary the backend uses.
Side A: `packages/tcip-web/src/tcip_web/routes/inference.py:205` (`@router.websocket("/jobs/{job_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/experiments.py:83` (`TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})`, the one declaration the job registry and the generated frontend vocabulary both read).
Phase 3 verdict: duplicated.

## S51. Training run stream WebSocket  <!-- queued: P5-297 unify -->

Must agree: the status payload the MCP tool returns is renderable by the browser's training view.
Side A: `packages/tcip-web/src/tcip_web/routes/training.py:270` (`@router.websocket("/runs/{experiment_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py` (`monitor_training` supplies the status payload).
Phase 3 verdict: duplicated.

## S52. Image-serve response headers  <!-- queued: P5-301 unify -->

Must agree: header names and value encodings match.
Side A: `packages/tcip-web/src/tcip_web/routes/images.py:671` (`extra = {"X-TCIP-Served-Size": f"{out_w}x{out_h}"}`, the one render header besides the cache's own).
Side B: `packages/tcip-web/frontend/src/lib/imageLoader.ts:35` (`servedSize: parseServedSize(headers.get("X-TCIP-Served-Size")),`).
Phase 3 verdict: duplicated.

## S53. Optimistic-concurrency token for label saves

Must agree: the token the browser echoes is the same token the backend minted for that label file.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:115` (`"base_mtime": token,`, the token the load route mints; the save door compares the echoed one at `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:523` (`if expect is not None and expect != read:`)).
Side B: `packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx:421` (`base_mtime: paths.mtime,`).
Phase 3 verdict: single.

## S54. Built frontend bundle location  <!-- queued: P5-305 unify -->

Must agree: the directory Vite writes is one of the directories the backend looks in.
Side A: `packages/tcip-web/frontend/vite.config.ts:26` (`outDir: "../static",`).
Side B: `packages/tcip-web/src/tcip_web/app.py:222` (`def _find_static_dir() -> Path:`).
Phase 3 verdict: duplicated.

## S55. Vite dev-server proxy prefixes  <!-- queued: P5-306 unify -->

Must agree: every backend path the browser calls in dev falls under a proxied prefix.
Side A: `packages/tcip-web/frontend/vite.config.ts:20` (`proxy: {`).
Side B: `packages/tcip-web/src/tcip_web/app.py:165` (`@app.websocket("/ws/state")`, one of the endpoints not under the `/api` prefix).
Phase 3 verdict: duplicated. The prefix literals still stand on their own, but `tests/test_frontend_route_paths.py` now fails when a path the frontend references falls outside them, sockets under the API prefix included.

## S56. Tab-name vocabulary  <!-- queued: P5-290 unify -->

Must agree: the tab a panel event targets, the tab the browser can restore, and the tab the backend persists are the same set of names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:128` (`ActiveTab = Literal["setup", "annotate", "training", "tuning", "inference", "results", "meta"]`, with `TAB_NAMES = get_args(ActiveTab)` beside it, `tcip_web.state` importing both).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts:17` (`export const TAB_NAMES = [`, generated from the same declaration).
Phase 3 verdict: duplicated.

## S57. One matcher for the assessment and the editor

Must agree: the pairs the editor shows a person and the pairs the assessment counts come from one matcher under one crowd rule.
Side A: `packages/tcip-annotation/src/tcip_annotation/matching.py:103` (`def pair_detections(gt: list[dict], dt: list[dict],`, the one matcher, COCO's center-in-crowd rule included).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py:379` (`m = pair_detections(gt, dt, criterion)`) and `packages/tcip-annotation/src/tcip_annotation/matching.py:141` (`m = pair_detections([record(annotations[i]) for i in gt],`, the editor's pairing through `dataset_layout.proposal_pairs` under the bucket's assessment's criterion, and the single-image scoring and its comparison render), held by `tests/test_label_document_gestures.py`'s agreement tests.
Phase 3 verdict: single.

## S58. Reference-grid geometry

Must agree: the cell name the agent points at and the cell the GUI highlights are the same rectangle.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py:36` (`def reference_cells(`, which builds the cells, with `grid_geometry`, `reference_grid.py:103`, the geometry handed over beside them).
Side B: `packages/tcip-annotation/src/tcip_annotation/grid.py:44` (`def grid_to_rect(cell: str, cells: "list[Any]") -> tuple[float, float, float, float]:`, the one cell-name lookup) and `packages/tcip-web/src/tcip_web/routes/images.py:253` (`@router.get("/serving_grid")`, whose cell list the browser's region serving consumes verbatim).
Phase 3 verdict: single.

## S59. Path confinement (the derived allow-set)

Must agree: every route that accepts a client-supplied path confines it to the same allowed roots.
Side A: `packages/tcip-web/src/tcip_web/paths.py:40`
(`def allowed_roots() -> list[Path]:`).
Side B: `packages/tcip-web/src/tcip_web/paths.py:174` (`p = allowed_path(path)`, the one adapter every route shares, `paths.py:156`).
Phase 3 verdict: single.

## S64. MCP tool registry against documented tool names  <!-- queued: P5-303 unify -->

Must agree: any document naming a tool names one the server actually registers.
Side A: `packages/tcip-mcp/src/tcip_mcp/server.py:98` (`def list_registered_tools() -> list[str]:`).
Side B: `tools/list_tools.py:15` (`from tcip_mcp.server import list_registered_tools`).
Phase 3 verdict: duplicated.

## S65. MCP client launch configuration  <!-- queued: P5-307 unify -->

Must agree: the environment name in the client config, the docs, and the environment file match.
Side A: `.mcp.json` (launches `conda run -n tcip-agent python -m tcip_mcp`).
Side B: `environment.yml:15` (`name: tcip-agent`).
Phase 3 verdict: duplicated.

## S66. Skill and docstring examples against real signatures

Must agree: a documented call binds against the real function signature.
Side A: `packages/tcip-mcp/src/tcip_mcp/knowledge/` (python fenced examples in the knowledge documents).
Side B: none. `verify_doc_examples.py` never ran against this tree (a documented-only script)
and was deleted rather than kept unrun; no other check verifies a knowledge document's fenced
example against the real signature it calls.
Phase 3 verdict: no live enforcement.

## S67. Local gate commands against the CI gate  <!-- queued: P5-308 unify -->

Must agree: the checks a contributor runs locally are the checks CI runs.
Side A: `CLAUDE.md` (documents `pytest -n 4`, `ruff`, `mypy`, and the frontend command chain; the docker job is CI-only by design, with no local counterpart).
Side B: `.github/workflows/ci.yml` (mypy job, python job with `pytest -n auto` and `TCIP_MIN_TESTS`, typescript job with format:check/lint/typecheck/test/build, docker job building `packages/tcip-web/Dockerfile` and polling the served GUI).
Phase 3 verdict: duplicated.

## Totals

61 seams are listed above. By verdict:

- `duplicated`: 21 seams.
- `single`: 38 seams (S03, S06, S07, S08, S12, S13, S14, S15, S16, S17, S18, S19, S20,
  S21, S22, S23, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S44,
  S45, S46, S53, S58, S59, S66).
- `restated-in-test`: 2 seams (S25, S57).

38 of 61 seams (62%) hold their agreement in a single implementation.
