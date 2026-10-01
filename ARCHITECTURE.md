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

Source: the module inventory `tools/build_module_inventory.py` produces, run at HEAD 452e37b2.
Every count in this section is read from that regenerated inventory, not from any earlier
snapshot; `tools/check_architecture_doc.py --inventory-json <path>` re-runs the same generator
and cross-checks its counts against this document's tables, this table's own module and line
totals included.

HEAD 452e37b2 has 437 modules across the six scanned roots (123385 total lines):

| Package (root) | Modules | Lines |
|---|---|---|
| tcip-mcp | 134 | 49205 |
| tcip-annotation | 12 | 3751 |
| tcip-web | 40 | 10459 |
| tcip-store | 13 | 4723 |
| tcip-web-frontend | 216 | 49220 |
| tools | 22 | 6027 |

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
| packages/tcip-mcp/src/tcip_mcp/audit.py | The audit log: one append-only store under a dataset root or a project root (:func:`audit_log_key`). | 4 | 34 |
| packages/tcip-mcp/src/tcip_mcp/cli/__init__.py | Operator command sub-package: each module is one ``tcip`` subcommand's implementation, exposing ``main(argv)`` and returning the exit code. | 2 | 12 |
| packages/tcip-mcp/src/tcip_mcp/cli/adopt_store.py | Move a root's existing record and log files into a store database. | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/archive_project.py | Export an annotation project as a portable bundle: a ZIP archive, or, with --output-dir, the identical bundle written as a directory tree. | 3 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/calibrate_operating_point.py | Calibrate + held-out validate a detection operating point over a labeled split. | 4 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/check_dataset_identity.py | Check a dataset's on-disk content against its recorded identity: detect changed / moved data. | 5 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/doctor.py | Data-state doctor: scan a live project for state inconsistencies code audits can't see. | 20 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/export_store.py | Write a root's database-held records and logs back out as files, for a root or for a whole project's roots at once: | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/import_project.py | Import an annotation project from a bundle ``tcip archive-project`` wrote: a ZIP archive, or a directory tree written by its ``--output-dir`` mode. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/inspect_compute_resources.py | Report the host's current compute headroom: CPU, memory, GPU free bytes, and how many training runs the project already has active. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/overlay_reference_grid.py | Render an image with a labeled reference-grid overlay for spatial referencing, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/plant_aware_group_splits.py | Plant-aware group-key derivation for ``draw_splits``, over per-stem georeferenced rasters. | 6 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/preflight_config.py | Validate a training configuration before launching, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/render_failure_cases.py | Find and render the worst predictions for failure analysis. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/scan_dataset.py | Scan a folder for images, labels, and predictions. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/score_predictions.py | Score on-disk predictions against on-disk ground truth (COCOeval), from the command line. | 4 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/shp_to_plant_csv.py | Convert a plant-locations shapefile into ``read_plant_csvs``' CSV schema. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/triage_predictions.py | Sort a checkpoint's own predictions by confidence into auto-accept, needs-review and unscoreable queues, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/visualize.py | Render annotations, predictions, a GT-vs-prediction comparison, or a sample grid, from the command line. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/cli/write_project_site.py | Correct one project's authored site, the deliberate overwrite for a site typed wrong once. | 2 | 0 |
| packages/tcip-mcp/src/tcip_mcp/dataset_layout.py | Canonical dataset-layout resolver: where an image's ground-truth labels and model predictions live on disk. | 6 | 50 |
| packages/tcip-mcp/src/tcip_mcp/experiments.py | A run as a directory: ``<project>/.tcip/experiments/<experiment_id>/``. | 6 | 25 |
| packages/tcip-mcp/src/tcip_mcp/identity.py | The platform's recorded-actor convention, in one place. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/knowledge/__init__.py | The one canonical domain-knowledge directory and its one reader. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/model_registry.py | Model registry: the checkpoints a project can load, and what is known of each. | 7 | 18 |
| packages/tcip-mcp/src/tcip_mcp/operationalization.py | A trait's latest confirmed revision, the one every measurement and delivery reads, and the check of whether its operationalization binds what a delivery door is about to write. | 2 | 13 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/__init__.py | Pipeline sub-package: data, models, training, evaluation, inference, postprocessing. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/__init__.py | Active learning pipeline: scorer and selector modules. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/helpers.py | Active-learning helpers: scorer lookup by method name and the composed-detector precondition. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/scorer.py | Active learning scorers: rank unlabeled images by informativeness. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/active_learning/selector.py | Active learning selector: partition a checkpoint's own predictions. | 0 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/band_stats.py | Display band statistics, the 8-bit stretch every band render goes through, and the RGB composite it stacks into. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py | Block-aware calibration/holdout: validate a detection operating point directly against a mosaic's own reserved calibration/test bands (see ``split_construction.spatial_single_source_split``'s four-way split, ``reserve_calibration_fraction``), for a raster training source too large or too singular to hold whole images out from. | 16 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/calibration.py | The calibrate/summarize pair every inference entry point shares: resolve a per-dataset operating point from a labeled split, and its compact, response-safe gate-evidence summary. | 10 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/__init__.py | Components sub-package: composable ML primitives. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/backbones.py | ``BackboneWrapper``: the interface a backbone must expose to the necks and detectors here. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py | 2D object-detector builders: plain torchvision detector factories. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/heads.py | Task-specific heads: each knows its loss, metric, and output format. | 1 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/losses.py | Loss functions for bespoke models: plain importable classes + a name->class map. | 1 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/components/necks.py | Neck modules: adapt backbone features for downstream heads. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/count_calibration.py | Resolve the count operating point over a locked, disjoint cal/holdout split of a labeled directory, through a prepared pass. | 11 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/__init__.py | Data pipeline: dataset loading, augmentation, tiling, splitting. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/augmentations.py | Data augmentation transforms for all task types. | 2 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/band_groups.py | Sensor-agnostic band-group correlation: sibling single-band raster files that are really one logical multi-band capture (some multispectral drone sensors write one file per band instead of one multi-band file per image), and the ``.bandgroup`` manifest that records a found group. | 5 | 18 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/coco_import.py | An external dataset-level COCO document, converted into the dataset's per-image label documents. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py | Whole-dataset content identity: the ``dataset_fingerprint`` formula (labels + image files + registry + confirmed negatives), recompute-on-read authority for the cached value a dataset's own ``dataset.json`` carries. | 6 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py | Multi-task datasets with standardized interfaces. | 11 | 13 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py | The producer: where a directory of ground truth or a ground-truth table becomes the samples a run trains, evaluates or calibrates over. | 8 | 13 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/samplers.py | Task-aware data samplers: class-imbalance handling plus read-locality ordering. | 2 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/selection.py | A selection: which samples train, which validate, which are held back to calibrate on. | 3 | 22 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/split_construction.py | Constructing training splits from a data config, beside ``splits.py``. | 14 | 6 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/data/splits.py | Group-aware, annotation-stratified train/val/calibration splitting: group-coherent (sibling tiles of one source never straddle two splits), annotation-balanced, deterministic in seed. | 11 | 21 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/delivery_events_schema.py | The ``delivery_events`` record's declared shape. | 0 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py | Tier-A derivations: compute a parameter (channels, num_classes, anchor ratios) from the artifact in hand. | 8 | 11 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/display_bounds.py | Pixel bounds for what the platform serves to a screen or writes as an agent-facing artifact. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/__init__.py | Review -> retrain feedback: materialize curated datasets and reconstruct a review-confirmed calibration reference from review verdicts. | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/materialize.py | Materialize a curated detection dataset from human review verdicts. | 11 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/feedback/review_calibration.py | Reconstruct a calibration reference from human review verdicts. | 7 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/image_utils.py | Shared image utilities for the composable ML pipeline (channel-aware). | 5 | 38 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/__init__.py | Inference pipeline: model loading and batch prediction. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py | Generic predictor for any bespoke ``model_source`` checkpoint: task read from the saved ``model_source``, prediction over one image, a batch, or a sliced source. | 10 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/inference/predictor.py | Model-kind contract + the predictor factory. | 6 | 10 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/__init__.py | Measurement primitives: morphology on a validated mask (area / perimeter / centroid / PCA axis extents). | 1 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/mask_geometry.py | Mask-geometry: dimensional measurements on a validated binary/instance mask. | 2 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/measurement/scale_calibration.py | Deriving and validating a physical per-pixel scale against real physical measurements. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py | ``build_model``: a run config's ``model_source`` to an ``nn.Module``, by importing the dotted builder it names and calling it. | 4 | 15 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py | The one model-side contract: the measurement boundary, as a behavioral check, not a mold. | 5 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py | Resolve the calibrated operating points (detection conf/NMS/max_dets/tile, and the classifier, ordinal and regression points) per dataset, at runtime. | 13 | 12 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/overviews.py | External overview pyramids (.ovr sidecars) for large rasters. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/pixel_size.py | The resolver from a raster's georeferencing tags to a real-world pixel size in meters (:func:`resolve_pixel_size` and its two wrappers, :func:`raster_pixel_size` and :func:`raster_pixel_size_reason`). | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/__init__.py | Postprocessing pipeline: temporal aggregation and CSV export. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/aggregation.py | Per-plant aggregation, temporal/spatial aggregation of per-image results. | 4 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py | CSV export for per-plant phenotyping results. | 7 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/orthomosaic_mapping.py | Georeferencing for a whole-mosaic GeoTIFF: a drone orthomosaic covers many plants in one raster, so mapping a detection to a real-world plant reads the GeoTIFF's own georeferencing tags to turn a pixel location into a real-world coordinate. | 1 | 6 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py | Canonical phenology measurement: a trait's positive-fraction milestones, for whichever registered trait it's computed for. | 7 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/plant_mapping.py | Plant-ID mapping across capture dates, by capture sequence plus GPS: | 13 | 14 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/segment_attribution.py | Per-plant attribution by canopy segment: a detection attributed to a plant by containment in a canopy boundary a person accepted, the segment itself tied to a registry plant by containment of the plant's own projected position. | 5 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/proposal.py | Annotation-proposal engines: a method-neutral seam for auto-labeling. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/raster_source.py | Raster reading: one open-and-read surface for every image source this platform decodes. | 4 | 20 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py | Named reference grid over a raster's native pixel frame: cells recomputed from the serializable geometry dict (:func:`grid_geometry`) by :func:`reference_cells`, each named spreadsheet-style, a bijective base-26 column letter plus a 1-based row number ("B3", ``tcip_annotation.sam_wrapper``'s ``column_label``). | 2 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/region_completeness.py | Per-cell content digest for the region-completeness store (:func:`tcip_mcp.dataset_layout.region_completeness_path`): detects an annotation edited or deleted inside an attested cell after attestation. | 6 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py | Runtime parameter resolution: a ``ResolvedParam`` carries a value, how it was derived and whether it is trustworthy. | 15 | 42 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/schemas.py | Pydantic v2 config schemas for structural/type validation; the runtime trainer reads the raw config dict. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/slicing.py | Tiled inference over the ``sahi`` library: the slice lattice, a platform checkpoint wrapped as a SAHI detection model, and the one cross-tile merge every tiled path runs. | 4 | 6 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/__init__.py | Training pipeline: trainer, progressive unfreezing, HPO. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/collation.py | Collate functions for a task's ``DataLoader``: batches a list of per-sample ``(image, target)`` pairs into the shape ``train()`` and ``evaluate()`` both expect. | 0 | 4 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py | The training envelope around any training body, the default trainer or an agent's custom ``train(ctx)``, and ``TrainContext``. | 18 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/eval_runners.py | Orchestrates a checkpoint evaluation run (tile-level or delivery-grade full-frame) and returns its scored result; ``evaluation.py`` keeps the metrics computation itself. | 11 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py | Task-aware evaluation metrics + composite selection objective: * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point TP/FP/FN), the canonical COCO mAP definition; * in-house scalar metrics for classification / ordinal / regression; * the composite selection objective (lower = better); * a task-agnostic two-pass ``evaluate()``. | 7 | 12 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/generic_trainer.py | Task-agnostic training loop for a bespoke ``model_source`` model. | 17 | 5 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/hpo.py | HPO, hyperparameter optimization on Ray Tune. | 7 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/optimizer_factory.py | Optimizer factory with differential learning rate support. | 0 | 2 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/run_registry.py | ``TrainRun``, the state of one run's body in the process running it. | 1 | 3 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py | The process entry point of one run's body: ``python -m tcip_mcp.pipelines.training.subprocess_worker --run-dir <dir>``, everything else it reads being in the directory's ``run.json``. | 9 | 1 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_guardian.py | Keep one child tied to the life of the process that launched it, on platforms with no job-object equivalent (Linux, macOS). | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/pipelines/training/tensorboard_manager.py | TensorBoard process management for training and HPO runs. | 0 | 3 |
| packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py | Prediction-bucket immutability: never silently overwrite predictions a human reviewed, and never silently publish a second run into a bucket a prior run already filled. | 8 | 8 |
| packages/tcip-mcp/src/tcip_mcp/project_paths.py | Paths under a project the caller names, and the repository root this package sits in. | 0 | 19 |
| packages/tcip-mcp/src/tcip_mcp/project_record.py | The project record: the one document every project carries, holding its identity and its site. | 3 | 11 |
| packages/tcip-mcp/src/tcip_mcp/project_status.py | Per-project status pointer: a small, persisted summary of recent activity. | 2 | 3 |
| packages/tcip-mcp/src/tcip_mcp/registry_paths.py | How a record stores a path and where a stored path resolves, the one rule every record under a project writes and reads paths through. | 0 | 15 |
| packages/tcip-mcp/src/tcip_mcp/server.py | MCP server entry point: every domain tool, served on stdio for the project named at start (``--project <path>``). | 25 | 23 |
| packages/tcip-mcp/src/tcip_mcp/store_catalog.py | The whole store catalog in one import: every module that registers a store. | 27 | 7 |
| packages/tcip-mcp/src/tcip_mcp/stray_state.py | What a stray file under a project's ``.tcip/state`` root is, and whether one path may be deleted. | 4 | 2 |
| packages/tcip-mcp/src/tcip_mcp/subject_registry.py | The dataset's subject registry, subjects, their attributes, and the deterministic name→id assignment a training run uses (and records, so predictions stay decodable). | 3 | 15 |
| packages/tcip-mcp/src/tcip_mcp/tools/__init__.py | Tool sub-package: each module registers tools with the MCP server. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py | Annotation tools: load, save and score name-based annotations. | 13 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/bundle.py | What a project bundle holds: every file of a project tree classified (:func:`account_for`). | 11 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/calibration_tools.py | Calibration-administration tools: redrawing a locked cal/holdout split, calibrating a scalar (ordinal-rank or continuous-value) trait against a disjoint held-out split, and earning a validated count operating point over an already-published prediction bucket. | 18 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/data_tools.py | Data management tools: census a dataset, split data. | 14 | 5 |
| packages/tcip-mcp/src/tcip_mcp/tools/delivery_tools.py | Delivery tools not owned by one trait or delivery kind: the general per-plant CSV door and delivery supersession. | 9 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/experiment_tools.py | Experiment MCP tools: read one run's directory, and list the project's runs. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/feedback_tools.py | Review -> retrain feedback MCP tools: ``materialize_review_dataset``, ``prioritize_review_queue`` and ``triage_predictions``, each reading the verdict store of the dataset root the review was recorded against, or the store the caller states instead. | 15 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py | GUI-driving tools: push data to a panel, or drive the live Annotate/Review tab to a frame. | 10 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py | Inference MCP tools: run_inference and deliver_per_image_counts. | 26 | 5 |
| packages/tcip-mcp/src/tcip_mcp/tools/ingest_tools.py | Image ingestion: turn a raw folder of photos into a structured TCIP project. | 8 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/knowledge_tools.py | The ``serve_domain_knowledge`` MCP tool: the route to the platform's domain knowledge documents for a client with no skill or instruction-file mechanism of its own. | 3 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py | Meta-loop tools for self-improvement. | 5 | 4 |
| packages/tcip-mcp/src/tcip_mcp/tools/model_tools.py | Model management tools, registry, listing, comparison. | 4 | 3 |
| packages/tcip-mcp/src/tcip_mcp/tools/orthomosaic_tools.py | Orthomosaic MCP tools: per-plant delivery from a persisted whole-raster prediction bucket plus a plant-locations CSV. | 12 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/phenology_tools.py | Phenology MCP tools, the agent-facing surface for the per-plant phenology pipeline. | 19 | 4 |
| packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py | Project management tools. | 18 | 11 |
| packages/tcip-mcp/src/tcip_mcp/tools/proposal_tools.py | Proposal-workflow tools: turn a chosen auto-labeling engine's output into predictions for canvas review. | 19 | 2 |
| packages/tcip-mcp/src/tcip_mcp/tools/scale_tools.py | Physical per-pixel scale calibration: the delivery-gating producer for ``resolve_scale.json``. | 12 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py | Training MCP tools, config validation, launch training, HPO, status. | 27 | 7 |
| packages/tcip-mcp/src/tcip_mcp/tools/trait_tools.py | The agent-facing door for proposing a trait's entry; the breeder confirms it in the Setup tab. | 2 | 1 |
| packages/tcip-mcp/src/tcip_mcp/tools/vision_tools.py | Vision tools: render annotations and predictions for visual analysis. | 21 | 5 |
| packages/tcip-mcp/src/tcip_mcp/traits.py | A trait: one entry holding its spec fields and the operationalization text for each delivery kind it delivers, kept per project as an appended list of revisions: ``propose_trait`` appends one, ``confirm_revision`` confirms or withdraws one by its number and content hash. | 10 | 28 |
| packages/tcip-mcp/src/tcip_mcp/utils/__init__.py | Shared low-level utilities for tcip-mcp. | 0 | 0 |
| packages/tcip-mcp/src/tcip_mcp/web_client.py | HTTP client for MCP tools to push state to the tcip-web backend (``post_panel_event``), and the declarations of the stores, the GUI state shape and the tab vocabulary (``ActiveTab``) the web package owns. | 7 | 14 |
| packages/tcip-mcp/src/tcip_mcp/workspace.py | The workspace: the folder whose child directories are the projects the GUI lists. | 6 | 13 |

## tcip-annotation

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-annotation/src/tcip_annotation/__init__.py | Headless annotation library: canonical name-based per-image JSON labels, and a COCO reader. | 8 | 5 |
| packages/tcip-annotation/src/tcip_annotation/annotation_engine.py | AnnotationEngine: Annotation CRUD, spatial index, undo/redo. | 2 | 1 |
| packages/tcip-annotation/src/tcip_annotation/format_io.py | The reader of an external dataset-level COCO document (an ``"images"`` / ``"annotations"`` / ``"categories"`` key), on its way into per-image documents. | 3 | 2 |
| packages/tcip-annotation/src/tcip_annotation/json_io.py | Per-image JSON: the canonical on-disk label format (ground truth + predictions). | 3 | 46 |
| packages/tcip-annotation/src/tcip_annotation/mask_contours.py | Mask -> polygon rings: the contour extractor behind every mask-derived shape. | 1 | 4 |
| packages/tcip-annotation/src/tcip_annotation/matching.py | Geometry helpers and GT-vs-prediction matching engine. | 2 | 4 |
| packages/tcip-annotation/src/tcip_annotation/review_engine.py | ReviewEngine: review logic, detection walk-through, accept/reject. | 5 | 7 |
| packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py | SAM2 wrapper for interactive segmentation. | 2 | 6 |
| packages/tcip-annotation/src/tcip_annotation/state.py | Annotation and review data model. | 0 | 24 |
| packages/tcip-annotation/src/tcip_annotation/utils.py | Shared utilities: image orientation, geometry helpers. | 0 | 4 |
| packages/tcip-annotation/src/tcip_annotation/verdicts.py | The review verdict: its action vocabulary, declared once, and the one reading of a stored entry. | 1 | 5 |
| packages/tcip-annotation/src/tcip_annotation/viz.py | Visualization rendering: draws annotations and predictions on images. | 2 | 2 |

## tcip-store

Counts in this table are import edges inside `packages/tcip-store/src`, counted the same way as every other table here and cross-checked against the regenerated module inventory the same way.

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-store/src/tcip_store/__init__.py | TCIP's storage seam: one interface for the platform's mutable records, logs, and blobs. | 6 | 62 |
| packages/tcip-store/src/tcip_store/adoption.py | Moving a root's existing record and log files into a database, atomically or not at all. | 6 | 3 |
| packages/tcip-store/src/tcip_store/binding.py | Which backend a process binds, decided once at its entry point. | 3 | 18 |
| packages/tcip-store/src/tcip_store/errors.py | Every refusal the storage seam raises. | 1 | 17 |
| packages/tcip-store/src/tcip_store/export.py | Writing one root's database back out as the file layout, and saying when it is stale. | 4 | 3 |
| packages/tcip-store/src/tcip_store/file_backend.py | The filesystem backend: identity to path, atomic replace, file locks, logs, and blobs. | 5 | 31 |
| packages/tcip-store/src/tcip_store/layout_claims.py | Which store could own which path under a root. | 3 | 11 |
| packages/tcip-store/src/tcip_store/model.py | Identity and value types the storage seam speaks, identical on every backend. | 0 | 6 |
| packages/tcip-store/src/tcip_store/registry.py | The store catalog: what each store is, how its values encode, and how it may be written. | 4 | 8 |
| packages/tcip-store/src/tcip_store/schema_version.py | The version-field accept rule every frozen store's reader applies. | 2 | 3 |
| packages/tcip-store/src/tcip_store/sqlite_backend.py | The SQLite backend: one WAL database per root, with blobs left as files. | 6 | 4 |
| packages/tcip-store/src/tcip_store/store.py | The storage seam's public surface: module functions bound to one backend per process. | 3 | 9 |
| packages/tcip-store/src/tcip_store/values.py | What a value must be before a store will carry it, and how a producer says it is not. | 0 | 3 |

## tcip-web

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/src/tcip_web/__init__.py | TCIP Web: FastAPI server for the ML pipeline. | 0 | 0 |
| packages/tcip-web/src/tcip_web/__main__.py | Entry point: ``python -m tcip_web``. | 7 | 0 |
| packages/tcip-web/src/tcip_web/app.py | FastAPI application: the GUI state snapshot and its WebSocket, the panel-event hub, the built frontend and the health probe; every domain route is mounted from ``tcip_web.routes``. | 10 | 6 |
| packages/tcip-web/src/tcip_web/cli/__init__.py | ``tcip``: the operator console command, dispatching to one subcommand per operator command. | 0 | 2 |
| packages/tcip-web/src/tcip_web/cli/__main__.py | Entry point for ``python -m tcip_web.cli``. | 1 | 0 |
| packages/tcip-web/src/tcip_web/cli/distill_learnings.py | Distill worksheet: gather one project's learning record in one place. | 6 | 0 |
| packages/tcip-web/src/tcip_web/identity.py | Current-user identity for provenance stamping (created_by / accepted_by). | 0 | 6 | <!-- queued: P5-329 unwired -->
| packages/tcip-web/src/tcip_web/jobstore.py | The web's in-memory async job registry and its memory cap. | 1 | 4 |
| packages/tcip-web/src/tcip_web/label_annotations_cache.py | The content-digest-keyed label-document parse memo shared by every scan of per-image label files. | 1 | 3 |
| packages/tcip-web/src/tcip_web/paths.py | Path confinement for client-supplied paths. | 6 | 13 |
| packages/tcip-web/src/tcip_web/routes/__init__.py | Route modules for the tcip-web FastAPI backend. | 17 | 1 |
| packages/tcip-web/src/tcip_web/routes/_body_common.py | The body model for a state-changing route that carries no fields of its own. | 0 | 4 |
| packages/tcip-web/src/tcip_web/routes/_coverage_models.py | The view-coverage record's viewing-context models. | 1 | 3 |
| packages/tcip-web/src/tcip_web/routes/_metrics_common.py | The response shape the tuning trial-metrics route serves. | 0 | 1 |
| packages/tcip-web/src/tcip_web/routes/annotate.py | Annotation label CRUD routes for the Annotate tab. | 10 | 2 |
| packages/tcip-web/src/tcip_web/routes/audit_gap.py | The shared shape a GUI route answers with when a mutation it already committed could not be recorded to the audit log. | 1 | 7 |
| packages/tcip-web/src/tcip_web/routes/canvas.py | Live canvas-state bridge: the GUI pushes what it is rendering; the agent reads it back. | 4 | 1 |
| packages/tcip-web/src/tcip_web/routes/coverage.py | View-coverage routes: the coverage lattice over a raster, the grid-zoom setting it is derived from, the per-image view-coverage record, and region-completeness attestations. | 13 | 2 |
| packages/tcip-web/src/tcip_web/routes/dataset.py | Dataset routes: what a dataset root holds (its dates, subjects and models, through :mod:`tcip_mcp.dataset_layout`), the ``GuiState.dataset`` selection, and the current image position within it. | 7 | 2 |
| packages/tcip-web/src/tcip_web/routes/fs.py | Local-filesystem directory browsing for the frontend's folder picker. | 2 | 1 |
| packages/tcip-web/src/tcip_web/routes/images.py | Image serving: the one path pixels reach the browser through. | 10 | 3 |
| packages/tcip-web/src/tcip_web/routes/inference.py | Inference routes: async tiled runs + live progress WebSocket. | 12 | 2 |
| packages/tcip-web/src/tcip_web/routes/meta.py | Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives, enumerated, ordered and decoded by the module that owns their stores. | 2 | 1 |
| packages/tcip-web/src/tcip_web/routes/projects.py | The workspace's projects: listing them, opening one, removing one and renaming one. | 10 | 3 |
| packages/tcip-web/src/tcip_web/routes/results.py | Results routes: plant-mapping, per-plant phenology curves, CSV export, and the traits with the breeder's confirmation of a trait revision. | 21 | 2 |
| packages/tcip-web/src/tcip_web/routes/review.py | Review routes: verdict/GT recording (compute matches, walk detections, record actions, save GT) plus the image-status group (mark_complete, backup_labels, image_statuses, generation_conf) and the priority queue. | 21 | 4 |
| packages/tcip-web/src/tcip_web/routes/sessions.py | Session-tracking routes: annotation_stats.json equivalent. | 6 | 1 |
| packages/tcip-web/src/tcip_web/routes/subjects.py | Subject registry routes. | 10 | 1 |
| packages/tcip-web/src/tcip_web/routes/terminal.py | Agent terminal routes: the HTTP/WS surface over :mod:`tcip_web.terminal`. | 4 | 5 |
| packages/tcip-web/src/tcip_web/routes/training.py | Training routes: launchable configs, launch/relaunch, list runs, live metrics stream. | 10 | 2 |
| packages/tcip-web/src/tcip_web/routes/tuning.py | HPO / Tuning routes: relaunch, cancel, list and per-trial visibility, each read off the sweep's own directory under ``.tcip/hpo`` (``training_tools.sweep_record`` and ``read_sweep``). | 10 | 1 |
| packages/tcip-web/src/tcip_web/routes/validation.py | Validation routes: promote a completed review into a validation reference. | 14 | 1 |
| packages/tcip-web/src/tcip_web/state.py | The web backend's own state: the workspace it serves, the project it has open, that project's live :class:`~tcip_mcp.web_client.GuiState` (persisted to the project's ``.tcip/state/gui.json`` on every change) and the panel events it retains for a browser that connects late. | 2 | 21 |
| packages/tcip-web/src/tcip_web/terminal.py | Embedded agent terminal: an agent harness from :data:`PROVIDERS` spawned directly in a pseudo-terminal (ConPTY via ``pywinpty`` on Windows, the stdlib ``pty`` on POSIX), its raw bytes streamed out and keystrokes streamed in. | 3 | 3 |
| packages/tcip-web/src/tcip_web/trust_boundary.py | The network trust boundary: which connections the backend serves and which names it answers to. | 0 | 2 |

## tcip-web-frontend

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| packages/tcip-web/frontend/src/App.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/App.tsx | (none found) | 31 | 2 |
| packages/tcip-web/frontend/src/api/client.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/api/client.ts | Typed REST client for the tcip-web backend. | 10 | 41 |
| packages/tcip-web/frontend/src/api/devProxy.generated.ts | Dev-server proxy prefixes, generated by tools/generate_frontend_routes.py from the routes the FastAPI app registers. | 0 | 0 |
| packages/tcip-web/frontend/src/api/http.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/http.ts | Shared fetch helpers. | 0 | 38 |
| packages/tcip-web/frontend/src/api/inference.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/inference.ts | Inference + Results API helpers for the Inference and Results tabs. | 4 | 14 |
| packages/tcip-web/frontend/src/api/meta.ts | Meta-loop API helpers: Claude's friction reports and retrospectives. | 2 | 2 |
| packages/tcip-web/frontend/src/api/routes.ts | Every backend path the browser calls, named for its method and its route. | 0 | 10 |
| packages/tcip-web/frontend/src/api/sessions.ts | Session-tracking API helpers (annotation_stats.json on disk). | 2 | 5 |
| packages/tcip-web/frontend/src/api/streams.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/subjects.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/subjects.ts | Dataset subject-registry + per-image-status API helpers. | 3 | 22 |
| packages/tcip-web/frontend/src/api/terminal.ts | REST client for the embedded agent terminal (a provider row's harness in a PTY). | 2 | 2 |
| packages/tcip-web/frontend/src/api/training.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/training.ts | Training-tab specific REST + WebSocket helpers. | 4 | 12 |
| packages/tcip-web/frontend/src/api/tuning.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/api/tuning.ts | Tuning (HPO) API helpers for the Tuning tab. | 3 | 3 |
| packages/tcip-web/frontend/src/api/types.generated.ts | Types generated by tools/generate_frontend_types.py from the pydantic models that declare them (routes/_coverage_models.py, routes/coverage.py, routes/review.py, routes/training.py, routes/terminal.py, routes/projects.py, routes/results.py, tcip_mcp.traits, tcip_mcp.web_client.GuiState), plus a handful of runtime constants (routes/images.py, tcip_mcp.web_client, tcip_mcp.experiments, tcip_web.jobstore). | 0 | 27 |
| packages/tcip-web/frontend/src/api/ws.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/api/ws.ts | WebSocket client that subscribes to GuiState snapshots + panel events. | 5 | 4 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.test.tsx | (none found) | 9 | 0 |
| packages/tcip-web/frontend/src/components/AnnotateToolbar.tsx | Annotate-tab context toolbar. | 17 | 2 |
| packages/tcip-web/frontend/src/components/BandPicker.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/BandPicker.tsx | (none found) | 2 | 3 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CanvasStage.tsx | Shared Konva Stage wrapper with pan + zoom state managed in the store. | 6 | 5 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageChrome.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageChrome.tsx | Coverage grid chrome for the open raster: the overlay toggle, a key naming what each overlay mark means, the grid's own derivation line, the attestation control for the cell under the viewport center (or the cell a Map click just opened, while it stays in view), and the errors and previous-lattice facts a breeder needs before trusting or acting on any of it. | 5 | 2 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageOverlay.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/CoverageOverlay.tsx | The coverage lattice drawn on the annotation canvas itself: a Konva layer content in image coordinates, culled to the viewport, listening={false} like every other canvas layer. | 2 | 2 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/Canvas/zoom.ts | Discrete zoom levels (5% .. | 0 | 3 |
| packages/tcip-web/frontend/src/components/CollapsibleSection.tsx | The app's collapsible-section primitive: one chevron glyph and one trigger+content unit. | 1 | 8 |
| packages/tcip-web/frontend/src/components/ColorPickerModal.tsx | Dark color picker: SI palette + basic palette + hex input, resolving to a hex string. | 0 | 2 |
| packages/tcip-web/frontend/src/components/ConfirmDialog.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/ConfirmDialog.tsx | A modal dialog shell for a destructive or otherwise consequential confirmation: a real focus trap (Tab/Shift+Tab stay inside), focus moved to the first control on open and returned to the button that opened it on close, Escape closes without confirming, and no backdrop click dismisses it (a destructive dialog must not close on a stray click). | 0 | 2 |
| packages/tcip-web/frontend/src/components/DeliveryEventsPanel.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/DeliveryEventsPanel.tsx | What has shipped from this project: one row per completed delivery, read-only. | 2 | 2 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/EmbeddedTool.tsx | A titled chrome bar over an iframe, for the tools the platform runs beside the app (TensorBoard, Ray's dashboard). | 0 | 3 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/ErrorBoundary.tsx | (none found) | 0 | 2 |
| packages/tcip-web/frontend/src/components/HaloLabel.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/HaloLabel.tsx | (none found) | 0 | 5 |
| packages/tcip-web/frontend/src/components/HelpOverlay.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/components/HelpOverlay.tsx | (none found) | 1 | 2 |
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
| packages/tcip-web/frontend/src/components/StatusBar.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/StatusBar.tsx | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/components/TabBanner.test.tsx | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/components/TabBanner.tsx | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/components/TabHeading.tsx | (none found) | 2 | 8 |
| packages/tcip-web/frontend/src/components/TerminalRail.test.tsx | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/components/TerminalRail.tsx | The agent rail: a real agent harness from the backend's provider table, embedded. | 5 | 2 |
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
| packages/tcip-web/frontend/src/components/annotate/AttributePanel.tsx | (none found) | 7 | 2 |
| packages/tcip-web/frontend/src/components/annotate/BoxOverlay.tsx | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/components/annotate/InProgressPolygon.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/components/annotate/PointOverlay.tsx | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/components/annotate/PolygonOverlay.tsx | (none found) | 3 | 1 |
| packages/tcip-web/frontend/src/components/annotate/SnapIndicator.tsx | (none found) | 1 | 1 |
| packages/tcip-web/frontend/src/components/review/EditShapeOverlay.tsx | (none found) | 2 | 1 |
| packages/tcip-web/frontend/src/components/review/FilterChip.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/components/review/LegendRow.tsx | A legend row whose color swatch is a button: click it to retune that symbology color. | 0 | 1 |
| packages/tcip-web/frontend/src/components/review/ReviewLegend.tsx | (none found) | 2 | 1 |
| packages/tcip-web/frontend/src/components/review/ReviewLine.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/components/review/ReviewOverlays.tsx | (none found) | 8 | 1 |
| packages/tcip-web/frontend/src/components/review/ReviewPoint.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/components/review/ReviewRect.tsx | (none found) | 0 | 1 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useActiveTabSync.ts | Mirror the active tab into the backend GUI state so view_gui_state reports the tab the human actually sees. | 3 | 2 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useBandSelection.ts | (none found) | 3 | 3 |
| packages/tcip-web/frontend/src/hooks/useCoverageGrid.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/hooks/useCoverageGrid.ts | The coverage lattice for the open raster at a subject's set grid zoom, plus the zoom-independent region-serving grid. | 3 | 3 |
| packages/tcip-web/frontend/src/hooks/useCoverageTracking.test.ts | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/hooks/useCoverageTracking.ts | Wires the CoverageTracker into the Annotate tab: resets on the (image, subject, date, dataset, grid) identity, hydrates from the stored record, feeds it viewport passes, the viewing context and the subject's working scale (the set grid zoom, never accumulated from an authoring commit), and exposes the seen/swept/pending sets for the coverage grid overlay plus the Complete warning facts. | 7 | 2 |
| packages/tcip-web/frontend/src/hooks/useDisclosure.ts | Open/closed state for a collapsible region, optionally remembered across sessions. | 0 | 6 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.ts | A staged agent request that follows the dataset selection until the breeder edits it. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.ts | Retries an embedded-tool launch attempt on a timer until it settles, the one polling shape the Training tab's run TensorBoard panel and the Tuning tab's sweep TensorBoard panel both need instead of each keeping its own copy of the same loop. | 0 | 4 |
| packages/tcip-web/frontend/src/hooks/useImageBands.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageBands.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/hooks/useImageNav.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageNav.ts | Single source of truth for image navigation. | 2 | 5 |
| packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.ts | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.ts | (none found) | 0 | 3 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useOverviewBuild.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/hooks/usePrefetchAdjacentImages.ts | (none found) | 4 | 2 |
| packages/tcip-web/frontend/src/hooks/useRegionCompleteness.test.ts | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/hooks/useRegionCompleteness.ts | Wires the region-completeness store into the Annotate tab: fetches every subject's attestation record and saved-annotation count for the open raster, exposes the active subject's own complete/stale cells separate from every other subject's, the active subject's working scale (the breeder's own set grid zoom, never derived from any annotation or echoed back from the browser), and posts an explicit attest/unattest/re-attest write. | 5 | 3 |
| packages/tcip-web/frontend/src/hooks/useRegionServes.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/hooks/useRegionServes.ts | The cell-aligned region serves the current viewport needs when the user zooms past the base bitmap's resolution on a large raster. | 6 | 3 |
| packages/tcip-web/frontend/src/index.css.test.ts | Compiles index.css through PostCSS + Tailwind (same pipeline as the real build) and asserts the keyboard focus-visible ring rules exist on the shared component classes: a typo'd token or dropped @apply utility fails here instead of silently shipping invisible focus. | 1 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/annotateFocus.ts | Drive the Annotate tab to a specific (subject, date, image, mode) in response to the agent's `annotate_focus` event. | 4 | 2 |
| packages/tcip-web/frontend/src/lib/authorshipSymbology.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/authorshipSymbology.ts | The Annotate canvas' dash symbology, one table shared by the box, polygon and point overlays so a "derived" pattern (a polygon's own read-only bounding box) and a "tool" pattern (a shape a tool drew that no person has accepted) never drift apart between them. | 0 | 6 |
| packages/tcip-web/frontend/src/lib/bandSelection.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/bandSelection.ts | (none found) | 2 | 9 |
| packages/tcip-web/frontend/src/lib/canvasSync.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/lib/canvasSync.ts | Live canvas-state sync: lets the agent see exactly what the canvas shows. | 5 | 12 |
| packages/tcip-web/frontend/src/lib/coverage.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/coverage.ts | Pure helpers over the coverage lattice a raster's grid route serves. | 2 | 18 |
| packages/tcip-web/frontend/src/lib/coverageTracker.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/lib/coverageTracker.ts | Session accumulator for the per-image view-coverage record. | 4 | 9 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/ctrlWheelGuard.ts | Stop the browser's own ctrl+wheel page zoom over the app, so the canvas' zoom is the only zoom. | 0 | 2 |
| packages/tcip-web/frontend/src/lib/datasetUiState.ts | Per-(project, dataset, date, subject/model) UI state in sessionStorage, so switching and returning within a session lands where you were: position and review filters in one blob, Review's GT/Pred visibility under its own key. | 3 | 4 |
| packages/tcip-web/frontend/src/lib/editGeometry.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/editGeometry.ts | Pure geometry for in-place box/polygon editing, shared by the Annotate and Review tabs' editors. | 2 | 4 |
| packages/tcip-web/frontend/src/lib/glyphs.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/glyphs.ts | The one glyph a select or a value render shows for "no value chosen" or "the record carries none": a colon, never an em dash or a hyphen. | 0 | 10 |
| packages/tcip-web/frontend/src/lib/imageLoader.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/imageLoader.ts | Shared image loader for /api/images serves. | 1 | 7 |
| packages/tcip-web/frontend/src/lib/imageStatus.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/imageStatus.ts | (none found) | 1 | 4 |
| packages/tcip-web/frontend/src/lib/joinRunSeries.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/joinRunSeries.ts | Overlay-chart helper for the Training tab's run comparison (kept out of the .tsx so it's unit-testable). | 2 | 2 |
| packages/tcip-web/frontend/src/lib/labelProblemToast.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/lib/labelSerde.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/labelSerde.ts | The single mapping between the unified name-based label file (one Annotation list per image) and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). | 1 | 4 |
| packages/tcip-web/frontend/src/lib/openProject.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/openProject.ts | Opening a workspace project: making it the backend's open project, then pointing the GUI at a dataset inside it (the project's own tree) via /dataset/select. | 5 | 3 |
| packages/tcip-web/frontend/src/lib/paths.ts | The paths the browser reads off the dataset selection. | 1 | 6 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/polygonGeometry.ts | Pure hit-testing and structural-edit helpers for the annotate canvas' polygon geometry. | 1 | 7 |
| packages/tcip-web/frontend/src/lib/recentProjects.ts | The ids of the last few projects the user opened, most recent first, for the status-bar fast-track; each is named through the current project listing. | 0 | 3 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/reconnectingSocket.ts | One reconnecting-WebSocket shape, shared by every socket this app opens: capped exponential backoff on an unexpected close, a guard against stacking a second attempt while one is already open or connecting, supersession so a replaced socket's late events are no-ops, and a restartable start/stop pair. | 0 | 5 |
| packages/tcip-web/frontend/src/lib/registrySweep.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/registrySweep.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/lib/reviewColors.ts | (none found) | 0 | 6 |
| packages/tcip-web/frontend/src/lib/reviewFocus.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/lib/reviewFocus.ts | Drive the Review tab to a model's predictions on a specific frame/detection in response to the agent's `review_focus` event. | 3 | 3 |
| packages/tcip-web/frontend/src/lib/reviewGeometry.ts | The single source of a review detection's geometry. | 1 | 5 |
| packages/tcip-web/frontend/src/lib/runStatus.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/lib/runStatus.ts | (none found) | 1 | 3 |
| packages/tcip-web/frontend/src/lib/subjectColors.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/subjectColors.ts | (none found) | 0 | 8 |
| packages/tcip-web/frontend/src/lib/tabLabels.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/lib/toolMode.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/toolMode.ts | Cycling the Annotate toolbar's drawing tool (its `m` shortcut and any other stepper). | 1 | 2 |
| packages/tcip-web/frontend/src/lib/viewGeometry.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/lib/viewGeometry.ts | Shared view math for the canvas: fit the view to a pixel rect and clamp pan offsets. | 2 | 7 |
| packages/tcip-web/frontend/src/main.tsx | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/appState.ts | (none found) | 11 | 12 |
| packages/tcip-web/frontend/src/store/exportSurface.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/guiStateShape.test.ts | (none found) | 2 | 0 |
| packages/tcip-web/frontend/src/store/index.ts | (none found) | 13 | 69 |
| packages/tcip-web/frontend/src/store/mergeSnapshot.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/store/slices/agentActivity.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/bandSelection.ts | (none found) | 2 | 2 |
| packages/tcip-web/frontend/src/store/slices/banners.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/canvas.ts | (none found) | 3 | 3 |
| packages/tcip-web/frontend/src/store/slices/gui.ts | (none found) | 5 | 11 |
| packages/tcip-web/frontend/src/store/slices/pendingTerminalMessage.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/registryStatus.ts | (none found) | 3 | 3 |
| packages/tcip-web/frontend/src/store/slices/review.ts | (none found) | 3 | 2 |
| packages/tcip-web/frontend/src/store/slices/terminalOpen.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/toasts.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/slices/user.ts | (none found) | 1 | 2 |
| packages/tcip-web/frontend/src/store/store.test.ts | (none found) | 3 | 0 |
| packages/tcip-web/frontend/src/store/tabRestore.test.ts | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/store/terminalOpenPolicy.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/store/types.ts | The GUI state's shape is generated from the backend's own model (types.generated.ts). | 1 | 44 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.test.tsx | (none found) | 10 | 0 |
| packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx | (none found) | 38 | 2 |
| packages/tcip-web/frontend/src/tabs/InferenceTab.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/tabs/InferenceTab.tsx | (none found) | 5 | 2 |
| packages/tcip-web/frontend/src/tabs/MetaTab.test.tsx | (none found) | 4 | 0 |
| packages/tcip-web/frontend/src/tabs/MetaTab.tsx | (none found) | 7 | 2 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/ResultsTab.tsx | (none found) | 10 | 2 |
| packages/tcip-web/frontend/src/tabs/ReviewTab.test.tsx | (none found) | 11 | 0 |
| packages/tcip-web/frontend/src/tabs/ReviewTab.tsx | (none found) | 35 | 2 |
| packages/tcip-web/frontend/src/tabs/RunMonitorLayout.tsx | The shell the Training and Tuning tabs share: a fixed-width scrolling sidebar of runs beside a detail region. | 0 | 2 |
| packages/tcip-web/frontend/src/tabs/SetupTab.test.tsx | (none found) | 5 | 0 |
| packages/tcip-web/frontend/src/tabs/SetupTab.tsx | (none found) | 6 | 2 |
| packages/tcip-web/frontend/src/tabs/TrainingTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/TrainingTab.tsx | (none found) | 18 | 2 |
| packages/tcip-web/frontend/src/tabs/TuningTab.test.tsx | (none found) | 6 | 0 |
| packages/tcip-web/frontend/src/tabs/TuningTab.tsx | (none found) | 14 | 2 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/agentPrompts.ts | Plain-language requests the run tabs stage for the agent, editable before they're sent. | 0 | 3 |
| packages/tcip-web/frontend/src/tabs/chartTheme.ts | Recharts takes literal color strings (not Tailwind classes), so the field-station tokens are mirrored here as hex. | 0 | 3 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts | (none found) | 1 | 0 |
| packages/tcip-web/frontend/src/tabs/trainingMetrics.ts | Metric-stream helpers for the Training tab (kept out of the .tsx so they're unit-testable). | 1 | 7 |
| packages/tcip-web/frontend/src/test/coverageOutbox.ts | Test-only reset for the shared coverage-outbox singleton: drop every queued payload and any pending retry timer, so one test's leftover state never leaks into the next. | 1 | 3 |
| packages/tcip-web/frontend/src/test/setup.ts | Extends Vitest's `expect` with jest-dom matchers (toBeInTheDocument, etc.) and registers automatic cleanup after each test. | 0 | 0 |
| packages/tcip-web/frontend/src/test/traitRecords.ts | (none found) | 1 | 3 |

## tools

| Module path | Ownership (one line) | In-repo imports | Imported by |
|---|---|---|---|
| tools/build_module_inventory.py | Builds a module inventory and real import graph for the repo's Python and TypeScript source trees. | 0 | 0 |
| tools/census_double_published_buckets.py | Census of prediction buckets published more than once before the live-bucket refusal. | 8 | 0 |
| tools/check_architecture_citations.py | Verify ARCHITECTURE.md's file:line citations against the code they quote, for CI. | 0 | 0 |
| tools/check_architecture_doc.py | Verify ARCHITECTURE.md's module-ownership tables against the tree, for CI. | 0 | 0 |
| tools/cross_family_ask.py | Pose one identical question to several agent harnesses and record comparable answers. | 0 | 0 |
| tools/gate_baseline.py | Run the quality gate CI actually declares, so a local pass means CI would pass too. | 0 | 0 |
| tools/generate_frontend_routes.py | Generate the browser's route-path module and the dev server's proxy from the backend's registered routes. | 1 | 0 |
| tools/generate_frontend_types.py | Generate the browser's backend-declared types from the pydantic models that declare them. | 13 | 0 |
| tools/generate_frozen_manifest.py | Generate frozen-formats.json, the shipped freeze commitment, from the store registry. | 2 | 0 |
| tools/generate_harness_discovery.py | Generate per-harness discovery files from the canonical domain-knowledge documents. | 1 | 0 |
| tools/generate_trait_fixture.py | Write the trait listings the frontend tests read, produced by the platform itself. | 6 | 0 |
| tools/line_delta.py | Sum a change's insertions and deletions by area: package code, tests, everything else. | 0 | 0 |
| tools/list_store_consumers.py | List every registered store's writers and readers, from the import graph plus a symbol scan. | 2 | 0 |
| tools/list_tools.py | Print the live MCP tool registry (count + names). | 1 | 0 |
| tools/prove_test_fails_before.py | Prove a test actually fails against the code it was written to catch. | 0 | 0 |
| tools/serve_capture_app.py | Start or stop the served web app under a scratch environment, for a GUI capture harness. | 0 | 0 |
| tools/smoke_phenology_e2e.py | Live e2e smoke: the agent's phenology pipeline on real geolocated images. | 12 | 0 |
| tools/smoke_terminal_e2e.py | One-shot smoke: the embedded agent terminal against one provider row's real harness. | 4 | 0 |
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

## Modules with zero importers (152)

A module counts as zero-importer when no other module in its own scanned tree resolves an in-repo import to it (`imported_by_count == 0` in the regenerated inventory). This includes package entry points (`__init__.py`, `__main__.py`), CLI scripts under `tools/` invoked as processes, package `cli/` command modules invoked by name through the `tcip` dispatcher rather than imported, and every TypeScript `*.test.ts`/`*.test.tsx` file, none of which are expected to have an in-repo importer.

| Root | Module path |
|---|---|
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__init__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/__main__.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/adopt_store.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/archive_project.py |
| tcip-mcp | packages/tcip-mcp/src/tcip_mcp/cli/calibrate_operating_point.py |
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
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/CoverageChrome.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/components/Canvas/CoverageOverlay.test.tsx |
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
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useActiveTabSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useBandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useCoverageGrid.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useCoverageTracking.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useEditableAgentRequest.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useEmbeddedToolRetry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageBands.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageNav.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useImageStatusHydrate.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useKeyboardShortcuts.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useOverviewBuild.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useRegionCompleteness.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/hooks/useRegionServes.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/index.css.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/annotateFocus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/authorshipSymbology.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/bandSelection.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/canvasSync.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/coverage.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/coverageTracker.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/ctrlWheelGuard.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/editGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/glyphs.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/imageLoader.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/imageStatus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/joinRunSeries.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/labelSerde.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/openProject.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/polygonGeometry.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reconnectingSocket.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/registrySweep.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/reviewFocus.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/lib/runStatus.test.ts |
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
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/ReviewTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/SetupTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/TrainingTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/TuningTab.test.tsx |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/agentPrompts.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/tabs/trainingMetrics.test.ts |
| tcip-web-frontend | packages/tcip-web/frontend/src/test/setup.ts |
| tools | tools/build_module_inventory.py |
| tools | tools/census_double_published_buckets.py |
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
| tools | tools/smoke_phenology_e2e.py |
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
says which.

Tables below group by defining module. Column "line" is the `def`/`async def` line.
Docstring is the function's docstring first line, verbatim.

### annotation_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `save_annotations` | `annotation_tools.py:83` | yes | Write an image's annotations to its single per-image label file (all subjects, one file). |
| `write_subject_registry` | `annotation_tools.py:406` | yes | Author the dataset's nested subject registry, a thin wrapper over ``subject_registry``. |

### data_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `freeze_selection` | `data_tools.py:18` | yes | Freeze a finished run's own drawn train/val partition into a selection, so a later run can |
| `draw_splits` | `data_tools.py:278` | yes | Compute a leakage-free, annotation-stratified train/val/calibration split. |

### experiment_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `get_experiment` | `experiment_tools.py:11` | no | Read one run's directory. |
| `list_experiments` | `experiment_tools.py:47` | no | Enumerate every run directory of the project: a training run and a calibration run of a |

### feedback_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `materialize_review_dataset` | `feedback_tools.py:104` | yes | Build a curated detection dataset from human review verdicts. |
| `prioritize_review_queue` | `feedback_tools.py:271` | no | Rank un-reviewed images by active-learning informativeness for the next review batch. |

### gui_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `push_panel_event` | `gui_tools.py:33` | yes | Push structured data to a TCIP GUI panel via the tcip-web backend. |
| `focus_human_attention` | `gui_tools.py:65` | yes | Drive the live GUI to a (subject, date) frame, the Annotate tab or the Review tab. |

### inference_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `run_inference` | `inference_tools.py:155` | no | Run a trained model over images or a raster, and persist the predictions as a bucket. |
| `clear_prediction_bucket` | `inference_tools.py:1193` | yes | Move a terminal experiment's own recorded prediction bucket into a dated archive under `predictions/.cleared/`, so the path re-publishes: the audited remedy `run_inference`'s own docstring and `delivery.md` name for a bucket the pointer lock has otherwise made unreachable a second time (a completed or failed experiment's bucket publishes once through the ordinary doors). |
| `deliver_per_image_counts` | `inference_tools.py:1791` | no | Export a CSV summary of detection counts per image, from a live run or a persisted bucket. |

### calibration_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `redraw_calibration_holdout` | `calibration_tools.py:19` | no, its library event | Deliberately redraw a locked calibration/holdout split. |
| `calibrate_scalar_operating_point` | `calibration_tools.py:197` | no | Calibrate and validate a trait's ordinal-rank or continuous-value prediction against a |
| `calibrate_count_operating_point` | `calibration_tools.py:394` | no | Calibrate and validate the count operating point against held-out GT, earning a claim |

### ingest_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `ingest_images` | `ingest_tools.py:208` | yes | Copy raw images into a structured project, bucketed by the capture date each file states. |
| `import_coco` | `ingest_tools.py:367` | no, its library event | Convert an external dataset-level COCO document into the dataset's per-image label documents. |

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
| `deliver_orthomosaic_plant_counts` | `orthomosaic_tools.py:494` | no, its library event | Per-plant detection counts from a persisted orthomosaic prediction bucket plus plant CSV(s). |

### delivery_tools.py (2 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `deliver_per_plant_csv` | `delivery_tools.py:21` | no, its library event | The general per-plant CSV door: `aggregate_per_plant`'s own output plus the existing |
| `supersede_delivery` | `delivery_tools.py:160` | yes | Record that a delivered file's number is withdrawn or replaced, without touching the file |

### phenology_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_plant_registry` | `phenology_tools.py:32` | yes | Register a plant-locations CSV set under a name, so `build_plant_mapping` and |
| `build_plant_mapping` | `phenology_tools.py:105` | no, its library event | Assign each geolocated image to a plant, then persist the mapping under this project. |
| `calibrate_classifier_operating_point` | `phenology_tools.py:418` | no | Calibrate and validate the trait's positive-class classifier against held-out GT. |
| `deliver_phenology_milestones` | `phenology_tools.py:556` | no, its library event | Per-plant phenology milestones from classified predictions + a plant mapping. |  <!-- queued: P5-43 unify -->

### project_tools.py (4 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `register_dataset` | `project_tools.py:97` | yes | Record a dataset's identity so a delivered number can be traced to the exact data behind it. |
| `initialize_project` | `project_tools.py:173` | yes | Create a TCIP project: ``.tcip/`` with its artifacts and models directories, and its record |
| `view_gui_state` | `project_tools.py:205` | yes | The live GUI session the human is looking at in this project: dataset, date, trait, tab, |
| `inspect_project` | `project_tools.py:220` | yes | Get an overview of the project. |

`tools/bundle.py` (not a tool module: no `@tool()` sites) is the one membership accounting
`archive_project` and `import_project` both compose from, `account_for(tree)`. It derives every
root a project tree is or holds (the fixed `.tcip` structure, plus every `selection.json`/
`curated_manifest.json` anchor under placement constraints that raise `AnchorMisplaced` when one
sits at the tree root, under `.tcip`, under a blob home, or under/above another derived root),
then classifies every file by precedence: bookkeeping, a record or log claimed by exactly one
derived root's own layout (`tcip_store.adoption.plan_root`; two roots claiming one file is a
`collisions` entry on the result, never a specificity tie), a recognized blob home, or
unaccounted. `archive_project` bundles the record/log and blob classes; `import_project` refuses
on any bookkeeping, collided, undecodable or unaccounted member before adopting or moving
anything.

### proposal_tools.py (3 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `propose_annotations` | `proposal_tools.py:173` | yes | Propose candidate annotations on an image for review, using a chosen auto-labeling engine. |
| `segment_prompt` | `proposal_tools.py:484` | yes | Turn an interactive prompt (points, a box, or grid cells) into mask polygon rings, via an engine. |
| `stage_proposals` | `proposal_tools.py:729` | yes | Stage model-/agent-proposed shapes as predictions for canvas review, the "show on canvas |

### scale_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `calibrate_physical_scale` | `scale_tools.py:75` | no | Derive and validate a physical per-pixel scale, and stamp it into ``pred_dir``'s |

### training_tools.py (6 tools)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `launch_training` | `training_tools.py:482` | yes | Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder. |
| `monitor_training` | `training_tools.py:625` | yes | Check the status of a training run, or of a hyperparameter sweep. |
| `cancel_training` | `training_tools.py:872` | yes | Request graceful cancellation of a running training run. |
| `run_hyperparameter_search` | `training_tools.py:1062` | yes | Run hyperparameter optimization on Ray Tune, training each trial for real. |
| `cancel_hyperparameter_search` | `training_tools.py:1392` | yes | Request cooperative cancellation of a running HPO sweep. |
| `evaluate_model` | `training_tools.py:1848` | no | Evaluate a trained checkpoint on a (held-out) dataset and return the result. |

### vision_tools.py (1 tool)

| tool | line | audited | docstring first line |
|---|---|---|---|
| `capture_live_canvas` | `vision_tools.py:696` | yes | Render exactly what the human's GUI canvas shows right now: image, shapes, viewport. |

## 2. HTTP routes and WebSocket endpoints

`packages/tcip-web/src/tcip_web/app.py` builds the FastAPI app, registers 5 HTTP routes
and 2 WebSocket routes directly, then calls `register_all(app)` from
`packages/tcip-web/src/tcip_web/routes/__init__.py`, which `include_router`s 17 route
modules under `routes/`, each with a fixed prefix. Verified: `routes/__init__.py`,
`routes/_metrics_common.py`, and `routes/_body_common.py` define no routes of their own (0
`@router.*` decorator sites in any of the three); `_metrics_common.py` holds `metrics_response`,
the response shape `tuning.py`'s trial-metrics route answers in, and `_body_common.py` holds
`EmptyBodyPayload`, the empty body model six path-parameter-only routes now declare so the
browser must send a preflighted request rather than reaching the handler as a simple one.

Total HTTP routes at HEAD: 89 (5 on `app.py` plus 84 across the 17 route modules, both counts
obtained by grepping `@app.get/post(` and `@router.get/post(` and summing);
websocket routes are counted separately, below, and excluded from this total. Each per-router
heading's own route count (and their sum, 87) includes any websocket route it lists, since
`routes/inference.py`, `routes/terminal.py` and `routes/training.py` each carry one; net of
those three, the 17 modules hold the 84 HTTP routes counted here.

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
| GET | `/api/state` | `get_state` | `app.py:149` |
| POST | `/api/state/tab` | `set_active_tab` | `app.py:158` |
| WS | `/ws/state` | `state_ws` | `app.py:167` |
| GET | `/health` | `health` | `app.py:258` |
| GET | `/` | `index` | `app.py:266` |
| POST | `/api/events/{panel}` | `post_panel_event` | `app.py:289` |
| WS | `/ws/panel/{panel}` | `panel_ws` | `app.py:335` |

### routes/annotate.py, prefix `/api/annotate` (2 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/labels` | `load_labels` | `routes/annotate.py:132` |
| POST | `/labels` | `save_labels` | `routes/annotate.py:156` |

### routes/canvas.py, prefix `/api/canvas` (1 route)

| method | path | handler | line |
|---|---|---|---|
| POST | `/state` | `push_canvas_state` | `routes/canvas.py:57` |

### routes/coverage.py, prefix `/api/coverage` (6 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/grid` | `get_grid` | `routes/coverage.py:222` |
| POST | `/grid_zoom` | `post_grid_zoom` | `routes/coverage.py:346` |
| GET | `` (root) | `get_coverage` | `routes/coverage.py:386` |
| POST | `` (root) | `post_coverage` | `routes/coverage.py:430` |
| GET | `/completeness` | `get_completeness` | `routes/coverage.py:574` |
| POST | `/completeness` | `post_completeness` | `routes/coverage.py:681` |

### routes/dataset.py, prefix `/api/dataset` (3 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/tree` | `get_dataset_tree` | `routes/dataset.py:119` |  <!-- queued: P5-83 unify -->
| POST | `/select` | `select_dataset` | `routes/dataset.py:172` |
| POST | `/nav` | `set_current_image` | `routes/dataset.py:242` |

### routes/fs.py, prefix `/api/fs` (1 route)

| method | path | handler | line |
|---|---|---|---|
| GET | `/list` | `list_dir` | `routes/fs.py:107` |

### routes/images.py, prefix `/api/images` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `` (root) | `serve_image` | `routes/images.py:472` |
| GET | `/bands` | `get_bands` | `routes/images.py:699` |
| POST | `/overviews` | `build_image_overviews` | `routes/images.py:823` |
| GET | `/overviews/status` | `get_overview_job` | `routes/images.py:841` |

### routes/inference.py, prefix `/api/inference` (3 HTTP + 1 WS)

| method | path | handler | line |
|---|---|---|---|
| POST | `/launch` | `launch_inference` | `routes/inference.py:206` |
| GET | `/jobs` | `list_jobs` | `routes/inference.py:309` |
| POST | `/jobs/{job_id}/cancel` | `cancel_job` | `routes/inference.py:314` |
| WS | `/jobs/{job_id}/stream` | `stream_job` | `routes/inference.py:324` |

### routes/meta.py, prefix `/api/meta` (2 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/reports` | `get_reports` | `routes/meta.py:20` |
| GET | `/retrospectives` | `get_retrospectives` | `routes/meta.py:42` |

### routes/projects.py, prefix `/api/projects` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `` (root) | `list_projects` | `routes/projects.py:84` |
| POST | `/open` | `open_project` | `routes/projects.py:139` |
| POST | `/remove` | `remove_project` | `routes/projects.py:181` |
| POST | `/rename` | `rename_project_route` | `routes/projects.py:217` |

### routes/results.py, prefix `/api/results` (10 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/plant_mapping/build` | `build_plant_mapping` | `routes/results.py:119` |
| POST | `/plant_mapping/load` | `load_plant_mapping` | `routes/results.py:247` |
| GET | `/plant_mapping/list` | `list_plant_mappings` | `routes/results.py:279` |
| POST | `/phenology_measurement` | `phenology_measurement` | `routes/results.py:542` |
| POST | `/export_csv` | `export_csv` | `routes/results.py:616` |
| POST | `/export_count_csv` | `export_count_csv` | `routes/results.py:751` |
| GET | `/traits` | `list_traits` | `routes/results.py:922` |
| POST | `/traits/confirm` | `confirm_trait_revision` | `routes/results.py:967` |
| GET | `/delivery-events` | `list_delivery_events` | `routes/results.py:862` |
| GET | `/models/registered` | `registered_models` | `routes/results.py:1004` |

### routes/review.py, prefix `/api/review` (8 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/matches` | `compute_image_matches` | `routes/review.py:360` |
| POST | `/action` | `record_action` | `routes/review.py:622` |
| POST | `/mark_complete` | `mark_complete` | `routes/review.py:789` |
| POST | `/backup_labels` | `backup_labels` | `routes/review.py:863` |
| GET | `/image_statuses` | `image_statuses` | `routes/review.py:912` |
| GET | `/generation_conf` | `get_generation_conf` | `routes/review.py:948` |
| POST | `/queue/launch` | `launch_priority_queue` | `routes/review.py:1076` |
| GET | `/queue/{job_id}` | `get_priority_queue_job` | `routes/review.py:1103` |

### routes/sessions.py, prefix `/api/sessions` (4 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/image_event` | `image_event` | `routes/sessions.py:73` |
| POST | `/start` | `start_session` | `routes/sessions.py:132` |
| POST | `/end` | `end_session` | `routes/sessions.py:148` |
| GET | `/load` | `load_sessions` | `routes/sessions.py:164` |

### routes/subjects.py, prefix `/api/subjects` (6 routes)

| method | path | handler | line |
|---|---|---|---|
| GET | `/load` | `load_subjects` | `routes/subjects.py:71` |
| POST | `/save` | `save_subjects` | `routes/subjects.py:125` |
| GET | `/image_status` | `get_image_status` | `routes/subjects.py:246` |
| POST | `/image_status` | `set_image_status` | `routes/subjects.py:260` |
| POST | `/image_status/bulk` | `set_image_status_bulk` | `routes/subjects.py:297` |
| POST | `/image_status/derive` | `derive_image_status` | `routes/subjects.py:335` |

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
| GET | `/runs` | `list_runs_route` | `routes/training.py:97` |
| GET | `/runs/{experiment_id}` | `get_run` | `routes/training.py:105` |
| POST | `/runs/{experiment_id}/tensorboard` | `launch_run_tensorboard` | `routes/training.py:112` |
| POST | `/runs/{experiment_id}/cancel` | `cancel_run_route` | `routes/training.py:142` |
| POST | `/compare` | `compare_runs_route` | `routes/training.py:162` |
| POST | `/compare/best` | `compare_best_route` | `routes/training.py:176` |
| GET | `/metric-directions` | `metric_directions_route` | `routes/training.py:217` |
| WS | `/runs/{experiment_id}/stream` (full path `/api/training/runs/{experiment_id}/stream`) | `training_stream_ws` | `routes/training.py:291` |

### routes/tuning.py, prefix `/api/tuning` (10 routes)

| method | path | handler | line |
|---|---|---|---|
| POST | `/sweeps` | `relaunch_sweep` | `routes/tuning.py:106` |
| POST | `/sweeps/{sweep_id}/cancel` | `cancel_sweep_route` | `routes/tuning.py:137` |
| GET | `/sweeps` | `list_sweeps` | `routes/tuning.py:149` |
| GET | `/sweeps/{sweep_id}` | `get_sweep` | `routes/tuning.py:162` |
| GET | `/sweeps/{sweep_id}/trials` | `list_trials` | `routes/tuning.py:176` |
| GET | `/sweeps/{sweep_id}/trials/{trial_id}/metrics` | `get_trial_metrics` | `routes/tuning.py:184` |
| GET | `/ray-dashboard` | `get_ray_dashboard` | `routes/tuning.py:202` |
| POST | `/sweeps/{sweep_id}/tensorboard` | `launch_sweep_tensorboard` | `routes/tuning.py:263` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard` | `launch_trial_tensorboard` | `routes/tuning.py:274` |
| POST | `/sweeps/{sweep_id}/trials/{trial_id}/tensorboard/stop` | `stop_trial_tensorboard` | `routes/tuning.py:288` |

### routes/validation.py, prefix `/api/review` (1 route)

| method | path | handler | line |
|---|---|---|---|
| POST | `/validate_reference` | `validate_reference` | `routes/validation.py:67` |

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

`packages/tcip-annotation/src/tcip_annotation/__init__.py:33-63` defines `__all__`: 22 entries.

differs from phase0 record: `docs/audit/phase0/surface-formats/annotation-api.md` states
in prose ("re-exports the following 26 names") that `__all__` has 26 names; its own table
under "Package-level exports" (lines 40-66 of that file) lists 27 rows. The HEAD fact is 22:
the COCO writers `write_coco` and `to_coco_dataset`, the format-dispatch aliases
`load_annotations_any` and `save_annotations_any`, and `detect_format` are gone.

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
| `parse_coco_annotations` | `format_io` | 16 |
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
`json_io.LABEL_SUFFIX`, `format_io.coco_categories`,
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
from `tcip_mcp.server` and calls it: `packages/tcip-mcp/src/tcip_mcp/server.py:105`
(`def main(argv: list[str] | None = None) -> None:`), which takes `--project <path>`, the one
project the server acts on, and runs the server `build_server` (`server.py:87`) builds for it:
every function a `@tool()` decorator (`server.py:31`) in
`packages/tcip-mcp/src/tcip_mcp/tools/*.py` declared, each bound to that project
(`python tools/list_tools.py` lists them; the count is never written down, since it drifts).

`python -m tcip_web`: `packages/tcip-web/src/tcip_web/__main__.py` defines `main()`
(`packages/tcip-web/src/tcip_web/__main__.py:38`), which reads the workspace once
(`tcip_mcp.workspace.workspace_from_environment`, `workspace.py:32`, refusing an unset
`TCIP_WORKSPACE`), reads `TCIP_WEB_HOST` / `TCIP_WEB_PORT` (default `127.0.0.1:8765`, declared
at `packages/tcip-web/src/tcip_web/__main__.py:20` (`DEFAULT_HOST = "127.0.0.1"`), read at
`packages/tcip-web/src/tcip_web/__main__.py:48`
(`host = os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)`)), writes the bound port under that
workspace (`replace(backend_port_key(workspace), str(port))`,
`packages/tcip-web/src/tcip_web/__main__.py:51`), configures the app's `StateStore` with the
same workspace (`store.configure(workspace, image_roots_from_environment())`,
`packages/tcip-web/src/tcip_web/__main__.py:59`), and serves the app via
`uvicorn.run(app, host=host, port=port)` (`packages/tcip-web/src/tcip_web/__main__.py:60`). The
app's lifespan (`app.py:43`) opens the project the workspace's last-opened pointer names
(`open_last_opened`, `packages/tcip-web/src/tcip_web/routes/projects.py:121`); with no pointer
the backend starts with no project open and every project-scoped route answers 409 until one is
opened from the picker. Every process reads the workspace once at its entry and hands it on as a
value; a test configures its own `StateStore`. Exposure is a property
of the accepted connection rather than the configured bind host, so this entry point always
binds the requested host and port; whether an arrival through a non-loopback address is served
is decided per request by `tcip_web.trust_boundary.TrustBoundaryMiddleware` (`trust_boundary.py:
296`), which refuses one unless `TCIP_WEB_ALLOW_INSECURE=1` is set (`insecure_opt_in`,
`trust_boundary.py:135`). The same middleware applies one Origin policy
(`origin_allowed`, `trust_boundary.py:267`) to every WebSocket scope and to every HTTP scope
whose method is state-changing (`STATE_CHANGING_METHODS`, `trust_boundary.py:41`), rather than
leaving each handler to call it for itself.

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
`packages/tcip-annotation/src/tcip_annotation/json_io.py:763`;
`tcip_mcp.pipelines.data.coco_import.import_coco_document`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/coco_import.py:23`;
`tcip_annotation.review_engine.ReviewEngine.save_gt`,
`packages/tcip-annotation/src/tcip_annotation/review_engine.py:739`;
`tcip_mcp.prediction_buckets.stage_prediction_shapes`,
`packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:420`.

Readers: `tcip_annotation.json_io.read_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:461`;
`tcip_mcp.dataset_layout.subjects_on_date`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:946`.

A prediction record's `created_by` is one spelling, `resolution.prediction_producer`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:886`, so every checkpoint-backed writer
stamps the same `model:<checkpoint-stem>@<sha256-prefix>` producer identity rather than each door
composing its own string.

Seam S17 ("Canonical per-image annotation JSON schema"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: every writer and the shared `json_io.read_annotations`
reader are exercised in round-trip tests
(`tests/test_tcip_web_routes.py:264,297,398,420,462`, `tests/test_review_engine.py:372`,
`tests/test_review_channel.py:103`, `tests/test_name_based_annotation_schema.py:200,246`). Gap
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
`packages/tcip-mcp/src/tcip_mcp/subject_registry.py:336`, the one write both registry doors call
(the GUI's `save_subjects` and the tool's `write_subject_registry`,
`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:406`).

Readers: `tcip_mcp.subject_registry.read_registry`, `subject_registry.py:201`;
`tcip_mcp.dataset_layout.list_subjects` (delegates to `subject_registry`),
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:744`.

`assign_class_ids`, `subject_registry.py:446`, derives the training-time name-to-id map from this
file's declared attribute order; no integer id is stored in the file itself.
`attribute_schema_digest`, `subject_registry.py:159`, hashes a subject's attribute
name/type/values for the `image_status_digest.json` staleness stamp (format 6).

Seam S20 ("subjects.json subject registry"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_name_based_annotation_schema.py:84` writes the
registry through the real `write_registry` and reads `num_classes` back through the real training
loader's call to `subject_registry.assign_class_ids`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:69`
(`return registry, subject_registry.assign_class_ids(registry, subject, attribute)`).
Gap: no test drives the actual `/api/subjects/save` HTTP route in the same test as the
training-side read.

## 4. `dataset.json`, dataset identity

Path: `<dataset_root>/dataset.json`.

Writer: `tcip_mcp.tools.project_tools.register_dataset`,
`packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:97`.

Reader: `tcip_mcp.pipelines.data.dataset_fingerprint.dataset_fingerprint` (recompute-on-read is
the stated authority; the stored value is a cache),
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py:138`
(`def dataset_fingerprint(dataset_root: str | Path) -> str | None:`).

Seam S26 ("dataset.json identity and fingerprint"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_dataset_identity_recording.py:78` calls the
real `register_dataset` writer, then the real `split_construction.dataset_identity` reader, both of
which call the identical `dataset_fingerprint` function, and asserts they agree. Gap: the seam's
named third consumer, `tcip check-dataset-identity`, is never executed by any test.

## 5. `image_status.json`, confirmed-negative / human-Complete store

Path: `<dataset_root>/.tcip/state/image_status.json`.

Writers: `set_image_status`,
`packages/tcip-web/src/tcip_web/routes/subjects.py:260`; `set_image_status_bulk`,
`routes/subjects.py:297`. A selection writes none: it lists the samples the admission
(`label_queries.admit`) already admitted, each by its own source and label path, so a confirmed
negative stays a fact about the dataset it was confirmed in and is never re-attributed to a
side's own copy of it.

Readers: `tcip_mcp.pipelines.data.label_queries.confirmed_negative_names`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:690`; `_status_bucket_for`,
`packages/tcip-web/src/tcip_web/routes/sessions.py:205`;
`tcip_mcp.subject_registry._sweep_schema_change`,
`packages/tcip-mcp/src/tcip_mcp/subject_registry.py:262`, which enumerates every bucket of a
subject whose attribute schema is about to change so the confirmations under it can be stamped
before the outgoing digest is gone; `routes.subjects.get_image_status`,
`packages/tcip-web/src/tcip_web/routes/subjects.py:246`, through
`tcip_mcp.pipelines.data.label_queries.stale_finished_names`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:666`, the public reader over a
resolved dataset root.

`IMAGE_STATUSES = ("complete", "partial", CONFIRMED_NEGATIVE, "unannotated")`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:656`, imported by the web route module.

The token a Complete stores here is subject-scoped before it ever reaches a writer: `mark_complete`,
`packages/tcip-web/src/tcip_web/routes/review.py:789`, derives it from the GT file through
`annotations_hold_subject`, scoped to the confirmed subject, and the browser posts that value on
through `set_image_status`.

Seam S22 ("image_status.json confirmed-negative store"), verdict `both-sides-restated`,
`phase0_implementation: mixed`: `tests/test_tcip_web_subjects_routes.py:132,149`,
`tests/test_confirmations_travel_with_dataset.py:65,95`, `tests/test_doctor.py:48`.
`tests/test_status_digest_stamp_writer.py` writes statuses through the real
`/api/subjects/image_status` and `/image_status/bulk` routes and reads them back through the real
`confirmed_negative_names`. Gap: whether `doctor.py` is ever driven against a route-written store
has not been re-verified; MCP-side readers test membership through
`dataset_layout.is_confirmed_negative` against the vocabulary declared beside the resolver.

## 6. `image_status_digest.json`, attribute-schema staleness stamp

Path: `<dataset_root>/.tcip/state/image_status_digest.json`.

Writers: `_stamp_digest`, `packages/tcip-web/src/tcip_web/routes/subjects.py:218`, called from
`set_image_status`/`set_image_status_bulk` at confirmation time; and
`tcip_mcp.subject_registry._sweep_schema_change`,
`packages/tcip-mcp/src/tcip_mcp/subject_registry.py:262`, called through `replace_registry`
(`packages/tcip-mcp/src/tcip_mcp/subject_registry.py:336`) by both registry writers,
`save_subjects` (`packages/tcip-web/src/tcip_web/routes/subjects.py:125`) and `write_subject_registry`
(`packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:406`), before the new registry lands.
A status and its stamp are two transactions, status first, so unstamped confirmations
legitimately exist; the outgoing registry is the last moment their digest is recoverable, so the
sweep records it there and they read as predating the change instead of as made under the new
vocabulary. Both writers reach the store through the one transactional writer
`dataset_layout.stamp_image_status_digests`,
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:774`, whose `only_unstamped` argument keeps
the sweep from re-dating a stamp the confirmation-time writer already set.

Reader: `tcip_mcp.pipelines.data.label_queries._stale_finished`,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:624`, the quarantine logic
shared by `confirmed_negative_records` and the public `stale_finished_names`; a name whose stamp
no longer matches the registry's current schema is dropped as `quarantined_stale_definition`
rather than trained by its stored confirmation, complete or negative alike,
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/label_queries.py:213`
(`counts["quarantined_stale_definition"] += 1`).

Seam S23 ("image_status_digest.json attribute-schema stamp"), verdict `both-sides-restated`,
`phase0_implementation: once, shared`: `tests/test_confirmations_travel_with_dataset.py:119-266`,
`tests/test_coco_training_assembly.py:591-680`. The shared function across both sides is
`attribute_schema_digest`, `subject_registry.py:159`. Both sides run their real implementations
end to end in `tests/test_status_digest_stamp_writer.py` (the HTTP status routes produce the
sidecar, `confirmed_negative_names` reads it) and in
`tests/test_schema_change_staleness_sweep.py` (both registry writers run the sweep,
`confirmed_negative_names` reads the result). Reader-only tests elsewhere still hand-write the
`{bucket: {image_name: digest}}` shape, reusing `attribute_schema_digest` for the value.

## 7. `view_coverage.json`, reference-grid coverage record, advisory

Path: `<dataset_root>/.tcip/state/view_coverage.json`.

Path/shape definition: `tcip_mcp.dataset_layout.view_coverage_path`, `dataset_layout.py:420`,
naming the stored record's own shape as `tcip_web.routes._coverage_models.CoverageRecord`.
Writer and reader: `routes/coverage.py`'s `post_coverage` (`routes/coverage.py:430`) and
`get_coverage` (`routes/coverage.py:386`), each validating the stored record against
`CoverageRecord` before merging into or serving it.

Seam S24 ("view_coverage.json advisory coverage record"), verdict: single. `_coverage_models.py`
declares `GridGeometry`, `StatsSource`, `WorkingScale`, `CoverageViewing` and `CoverageRecord`
once; `routes/coverage.py`'s `CoveragePayload` types its `grid` and `viewing` fields against those
models; `tools/generate_frontend_types.py` renders `frontend/src/api/types.generated.ts` from
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
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:489`;
`region_completeness_digest_path`, `dataset_layout.py:520`. Every reader reads the store as
written, so a record lacking a key fails at the read.

Writer/reader named by phase0 but not independently opened this session:
`pipelines/region_completeness.py` as the digest sidecar's writer.

Seam S25 ("region_completeness.json attestation store"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_coverage_routes.py:449`,
`tests/test_block_calibration.py:126,164,203`. The shared functions across the HTTP-route side and
the calibration-gate side are `region_completeness_path` (`dataset_layout.py:489`),
`region_completeness_key` and `status_bucket` (`dataset_layout.py:586`). Gap: `test_block_calibration.py`'s
`_attest_regions_complete` helper bypasses the HTTP route, writing the store via the same shared
functions the route calls internally rather than via a POST to `/api/coverage/completeness`, so a
bug confined to the route's own HTTP layer would not be caught by the calibration-gate tests.

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
each of its acts (`import_coco`, `build_plant_mapping`, `redraw_calibration_holdout`,
`deliver_per_plant_csv`, `deliver_orthomosaic_plant_counts`, `deliver_phenology_milestones`,
`register_model`, `run_inference`, `deliver_per_image_counts`, `propose_trait`, and the four
calibrate doors
`calibrate_count_operating_point`, `calibrate_scalar_operating_point`,
`calibrate_classifier_operating_point` and `calibrate_physical_scale`, whose lock, record, curve
and stamp each leave their library's line; a stamp's is `stamp_written`, written by
`resolution.write_sidecar` and `resolution.update_sidecar`): bare, the project's own log;
`@audited(scope_arg=...)` names the argument carrying
a dataset or project location, resolved via `dataset_scope_of` (`audit.py:154`) (through the tool's own
canonicalizer when the declaration passes one as `scope_via`). Eight doors declare one: seven
dataset-scoped (`save_annotations`, `tools/annotation_tools.py:83`; `write_subject_registry`,
`tools/annotation_tools.py:406`; `materialize_review_dataset`, `tools/feedback_tools.py:104`;
`clear_prediction_bucket`, `tools/inference_tools.py:1193`; `register_dataset`,
`tools/project_tools.py:97`; `propose_annotations`, `tools/proposal_tools.py:173`;
`stage_proposals`, `tools/proposal_tools.py:729`). A resolution that answers "no dataset" leaves
the entry in the project's own log; a resolver that raises refuses the call rather than filing it
there.

`record_event`/`record_event_or_raise` cover code that is neither an MCP tool nor a demoted door.
Project-scoped: the training envelope's open/close events
(`pipelines/training/envelope.py`), the model registry's write event (`model_registered`,
`model_registry.py`), a calibration curve's
first write (`calibration_curve_written`, `tools/inference_tools.py`'s `keep_calibration_curve`),
and `routes/terminal.py`'s one line per agent-terminal launch (`agent_terminal_started`, `routes/terminal.py:99`). The `@audited(scope_arg=...)`
doors span every category by whatever root their declared argument resolves; this paragraph
names the explicit-emitter files, not a closed census of the decorator's doors.
Dataset-scoped: two GUI route writers passing the dataset root their own guard resolved
(`routes/subjects.py`'s `_audit_dataset_write`, `routes/subjects.py:40`; `routes/review.py`'s `_audit`,
`routes/review.py:84`), plus `routes/annotate.py`'s `_audit_gui_write` (`routes/annotate.py:112`), dataset-scoped
when its guard resolves one and project-scoped otherwise (a label path confined to an allowed
root but outside any dataset tree), `resolution.py`'s `record_delivery_binding_event`
(`resolution.py:2256`, dataset-scoped when a
delivery's buckets share one dataset root, project-scoped otherwise),
the calibration/holdout lock's draw event (`calibration_holdout_drawn`,
`pipelines/data/splits.py:1026`), written by every first draw and redraw whichever door triggers
it, the COCO import's `coco_document_imported` (`pipelines/data/coco_import.py`), written
once a document has committed, on success and on a later failed write alike, and a prediction
bucket's `prediction_bucket_published` (`tools/inference_tools.py`'s `_publish_predictions`),
written by the publishing library for an image bucket and a raster bucket alike, whichever door
published it (the GUI's inference worker included), naming the documents written, under status
`failed` with the error when a pass raises after its first document and before its stamp, and every bucket stamp's `stamp_written`
(`pipelines/resolution.py`'s `write_sidecar` and `update_sidecar`), naming the stamp as stored,
filed under the bucket's dataset root (`dataset_layout.bucket_dataset_root`) or the project's log
when the bucket sits under none.
Project-scoped: `routes/results.py`'s `_audit` (`routes/results.py:97`, its delivery and confirmation routes) and
`pipelines/postprocessing/plant_mapping.py`'s `persist_mapping` (`pipelines/postprocessing/plant_mapping.py:1331`), whose two callers file
its receipt in the project's own log: the MCP tool `build_plant_mapping` passes the project its
server was started for and the web build route passes the backend's open project.

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
(`pipelines/postprocessing/plant_mapping.py:1508`) and `_require_receipt`
(`pipelines/postprocessing/plant_mapping.py:1535`), the hard receipt gate `load_mapping` runs
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
`packages/tcip-mcp/src/tcip_mcp/experiments.py:103`, under the project the caller names. A run
is a directory of plain files, not store records: each file is written by one process, the launch
record once by the parent and the final status once by the child, and nothing in it is rewritten.
Immutability is the files' own: `publish_once`, `experiments.py:201`, writes a file's whole bytes
to a staging name and publishes them under the final name without replacing anything, so a file
under its final name is always whole. A relaunch or a resume is a new directory naming its source
(`parent_experiment`, `resume_from`); `create_run_directory`, `experiments.py:165`, refuses a
directory that already exists (`RunDirectoryExists`). A sweep is the same shape one level up,
`.tcip/hpo/<study_name>/` (`sweeps_dir`, `experiments.py:108`): its `sweep.json` input written
once before its thread starts, a heartbeat, its final status, and one `trial_<id>/` run directory
per trial, from which its projection derives.

- `run.json` (`RUN_FILE`, `experiments.py:45`): written once by `open_run`,
  `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:448` (`def open_run(`), before the child
  starts: the config as launched with its seed drawn onto it, what the launcher resolved it to
  (`resolved`: the data section, the partition and the objective with its direction, from
  `split_construction.resolve_run`; `null` for an HPO trial whose point failed to resolve, whose
  final status is written `failed` naming why), the environment, the dataset identity, who launched it, the
  run it relaunched and the checkpoint it resumes from, its wall clock, the model contract
  preflight proved, the source snapshot, and an HPO trial's sampled point. The child builds its
  loaders from `resolved` and resolves nothing again. A calibration run of a checkpoint no run
  produced (`open_calibration_run`, `experiments.py:708`) carries `config: null` and what it
  calibrates, the checkpoint's sha256 among it. Read through `observe`, `experiments.py:374`, and
  `run_resolution`, `experiments.py:440`, which `pipelines/block_calibration.py` and
  `pipelines/operating_point.py` take the partition from.
- `metrics.jsonl` (`METRICS_FILE`, `experiments.py:46`, append-only): created empty with the
  directory, appended through `append_row`, `experiments.py:242`, by the training envelope's one
  sink, `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:243`
  (`def _epoch_sink(`). Read by `read_rows`, `experiments.py:261`. A row landing after the final
  status is reported as late (`rows_after_end`), never read as reopening the run.
- `heartbeat` (`HEARTBEAT_FILE`, `experiments.py:47`): touched by `keep_heartbeat`,
  `experiments.py:287`, while the run's process lives; `last_alive`, `experiments.py:304`, falls
  back to the launch record before the first touch.
- `cancel_requested.json` (`CANCEL_FILE`, `experiments.py:48`): written once by `request_cancel`,
  `experiments.py:323`, which keeps the first request's time and refuses a directory whose final
  status is written.
- `final_status.json` (`FINAL_STATUS_FILE`, `experiments.py:49`): written once through
  `write_final_status`, `experiments.py:230`, by the envelope or by the child's pre-envelope
  failure path: the state, when it ended, its error, and for a completed run the checkpoint
  (path inside the run directory and sha256) the verified checkpoint reader admitted. `observe`
  answers that state once written, else `running` or `interrupted` by the heartbeat window; a
  run's summary, completed or live, is the best selection over its own metrics rows under its
  recorded objective (`run_summary`, `experiments.py:466`).
- `validations.jsonl` (`VALIDATIONS_FILE`, `experiments.py:50`, append-only): the claims earned
  against this run's evidence, appended by `append_validation`, `experiments.py:670`, which refuses
  a row missing any of `_VALIDATION_FIELDS` (`experiments.py:637`). A claim for a checkpoint a run
  produced is filed in that run; one for a checkpoint no run produced in a calibration run
  directory of its own, opened and finished through the same lifecycle. Read by
  `find_validation`, `experiments.py:701`, matching rows by recomputed `validation_digest`,
  `experiments.py:665`. A validation is made after a run ends, so a final status never closes it.
- The checkpoints: the files the run's body saves (`model_best.pt`, held in memory until the run
  ends, `model_final.pt`, `checkpoint_epoch_*.pt`, a bespoke loop's own tags), each written once
  under a name no other write takes. An evaluation of the run writes nothing here.

Readers over the whole directory: `get_experiment`, `experiments.py:493`; `compare_experiments`,
`experiments.py:545`; `get_experiment_lineage`, `experiments.py:613`.

A validation row's `train_disjointness` is `{"checked": bool, "group_check": str | None}` for the four
  documents whose gate runs the check, `null` for `resolve_scale`. `selection_disjointness` is the
  parallel field for whether the calibration used to validate a checkpoint's own reference was kept
  disjoint from that checkpoint's own selection (val) side: `{"applicable": bool, "reason": str |
  None, "checked": bool, "unresolvable": bool, "leaked_groups": list, "leaked_stems": list,`
  `"group_check": str | None, "labels_moved_draw_to_run": list | None, "labels_moved_run_to_now":
  list | None, "calibration_labels_moved": list | None, "selection_redrawn": bool | None}` for
  the four documents `resolver_selection_disjointness`
  covers, `null` for `resolve_scale`; `applicable` is `False` when no selection is in play (no
  `selection_dir` named and no `selection` on the checkpoint's own run's partition, a within-image
  `spatial_strip` split, an empty `val`, an `external` group_by, no `calibration_labels_dir`
  named at all, or one none of the run's own members live under), each such case
  carrying its own `reason`; `unresolvable` marks
  the one case the ruling refuses rather than skips, a named selection with no resolved record
  to read a selection side from.

The run directory's lifecycle, each case through the real launcher and a real child process
(launch to completion, cancel by id, resume into a new directory, a child killed before its
final status, a row logged after the final status, a launch into an existing directory, and an
HPO trial as a run directory reporting its sweep's one objective), is held by
`tests/test_run_directory_lifecycle.py`. The metrics log's writer and the training stream's reader
are held against each other by `tests/test_metrics_row_writer_reader_agreement.py`, which writes
rows through the real run body and reads them back through the real websocket route; the tuning
trial-metrics route reads the same file through `read_rows`. The partition's writer (the
launcher's `resolve_run`, recorded by `open_run`) and its readers (`operating_point.py`'s disjointness check,
`block_calibration.py`'s `resolve_block_calibration_records`) are driven against the same
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
`model_registry.py:234`), replaces an earlier entry of the same sha256 and returns the owner the
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
it: `ModelRegistry.list_models`, `model_registry.py:422`, and the module's `best_model`,
`model_registry.py:418`, over its entries. `best_model` takes `metric_key` and `higher_is_better`
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

## 17. Prediction buckets, verdict-guarded prediction directories

A prediction bucket is not a score bin or a quota allocation: it is one directory holding a
single model run's per-image prediction documents, its identity the directory's own path
(relative to the dataset root under one, its own resolved path otherwise, `bucket_key_of`; the
canonical `predictions/<model>/<date>` layout is one regime's convention for building that
path), mutable until a human records a review verdict against any image inside it, after which
the default write is redirected to a fresh variant and an `overwrite=True` write is refused
(`BucketHasVerdicts`) rather than allowed to overwrite it in place; a bucket under no dataset
root has no verdict store and so no guard for the verdict behavior. `resolve_writable_bucket`'s
`count_review_state` keyword (default off) widens that same guard, for the staging door alone, to
every image a reviewer has finished, a bulk accept included, not detection verdicts alone; the
three publishers below leave it off. A second, narrower rule applies only to the callers that opt
in (`resolve_writable_bucket`'s `refuse_documents`, `run_inference`,
`deliver_per_image_counts`'s live path, and the web route's own launch): a requested bucket that
already holds a prediction document, with no verdict yet recorded, refuses outright
(`BucketHoldsDocuments`) whatever `overwrite` says, regardless of a dataset root;
`stage_prediction_shapes` alone leaves this off.

Path: `<dataset_root>/predictions/<model_name>/[<date>/]`, via
`tcip_mcp.dataset_layout.prediction_dir`.

Writer: `stage_prediction_shapes`, `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:420`, the
underlying per-image files written via `tcip_annotation.json_io.write_annotations`,
`packages/tcip-annotation/src/tcip_annotation/json_io.py:763` (format 1's writer).
`resolve_prediction_bucket`, `prediction_buckets.py:387`, resolves a `(dataset_root, model_name,
date)` triple to a writable directory; `resolve_writable_bucket`, `prediction_buckets.py:321`,
redirects to the next free `<model_name>@r2`/`@r3` variant once any image in a bucket has a
recorded review verdict, or, under `stage_prediction_shapes`'s own `count_review_state=True`,
once any image simply carries review state (a bulk accept included); `BucketHasVerdicts`,
`prediction_buckets.py:203`, is raised instead when `overwrite=True` is requested against a bucket
answered for under the caller's own reading, or when the variant search itself is exhausted.
`bucket_document_stem_count`, `prediction_buckets.py:266`, is the document count
`BucketHoldsDocuments`, `prediction_buckets.py:231`, names; `run_inference` and
`deliver_per_image_counts` reach both classes through the shared
`_resolve_writable_bucket_for`, `tools/inference_tools.py:753`.

Readers: `bucket_stems`, `prediction_buckets.py:44`, walks each dir through
`tcip_annotation.json_io.prediction_documents`, which excludes every provenance stamp named in
that module's own `SIDECAR_FILENAMES`, so a stamp added for a new measurement dimension is
excluded here too; `verdict_count`, `prediction_buckets.py:155`, delegates
to `tcip_annotation.review_engine.ReviewEngine.verdict_count_for_images` against the store
`project_state_dir`, `project_paths.py:9`, names; `review_state_count`,
`prediction_buckets.py:170`, counts every image a reviewer has finished under a bucket, a bulk
accept included, bucket-wide with no `names` argument or stem-scoped with one, the reading
`stage_prediction_shapes` opts into through `count_review_state`.

Seam S29 ("Prediction-bucket immutability"), verdict `both-sides-one-implementation`,
`phase0_implementation: once, shared`: `tests/test_prediction_bucket_resolution.py:39,48`,
`tests/test_review_channel.py:287,312,325`, `tests/test_run_inference_bucket_handling.py:60`,
`tests/test_orthomosaic_tools.py:241-248`, `tests/test_tcip_web_routes.py:898`. Gap: the seam's
fourth named caller, inside `stage_proposals`'s assignments-regime `except BucketHasVerdicts`
block, has no test coverage; every test of that regime writes into a fresh, verdict-free bucket.

## 18. `operating_point.json`, prediction-bucket provenance sidecar

Path: `<prediction_bucket_dir>/operating_point.json`, sibling of the bucket's per-image
prediction files (format 17's immutability-scoped prediction directory, not a score bin), one
of the stamp filenames `tcip_annotation.json_io.SIDECAR_FILENAMES`
(`packages/tcip-annotation/src/tcip_annotation/json_io.py:215`) names and every bucket
enumeration excludes (format 17).

Writer: `write_sidecar`, `resolution.py:771`, the one write, under the stamp's own lock;
`update_sidecar`, `resolution.py:796`, is the one merge into an existing stamp, reading and writing inside
one transaction so a promotion cannot drop what the producing run recorded. Both refuse, through
one shared check (`_check_stamp_claim`, `resolution.py:736`), a stamp claiming validation with no
well-formed `validated_by` or no trait, so a producer cannot omit what every reader compares.
The stamp itself is built by `operating_point_stamp`, `resolution.py:828`, which requires every provenance
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

Reader: `read_operating_point_sidecar`, `resolution.py:925`, which never raises: an unreadable
stamp floors the dimension it describes to unvalidated at every reconciler rather than taking down
a delivery gate. A stamp claiming validation is checked against the record it names by
`verify_stamp_binding`, `resolution.py:1641`, called from inside every reconciler so no delivery door can
reach a validated result without it; a failed binding floors every dimension that stamp carries
and the reason lands in the reconciler's `binding_notes`. The claim a stamp asserts is subset by
`claim_payload`, `resolution.py:1045`, the one extractor the minting side and the reading side share. `routes/validation.py:407` merges the review promotion's own validation fields in
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
supplies (`test_stamp_carries_the_same_keys_whatever_the_producer_supplies`,
`tests/test_operating_point_sidecar_seam.py:66`), a write round-trips through the real reader
(`test_sidecar_write_and_read_round_trip`, `tests/test_operating_point_sidecar_seam.py:191`), a merge is made against
what is stored rather than against a pre-lock read (`test_sidecar_update_merges_against_what_is_stored`,
`tests/test_operating_point_sidecar_seam.py:210`), and the declared stamp filenames cover
every declared document (`test_declared_stamp_filenames_cover_every_declared_document`,
`tests/test_operating_point_sidecar_seam.py:307`).

## 19. `.tcip/state/gui.json`, live GUI state snapshot

Path: `<project_root>/.tcip/state/gui.json`, addressed by `gui_snapshot_key`,
`packages/tcip-mcp/src/tcip_mcp/web_client.py:65`.

Writer: `write_gui_snapshot`, `tcip_mcp/web_client.py:284`, called by `StateStore.mutate`
(`tcip_web/state.py:170`) for the project open when the change is made, before the change is
held; a write that fails raises and the change is not held. The store is declared
`durable=False`: losing the last snapshot costs a re-selection, not history.

Reader: `read_gui_snapshot`, `tcip_mcp/web_client.py:297`, run by `StateStore.open_project`
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

## 21. `.tcip/state/review/*.json`, review-verdict shard store

Path: `<state_dir>/review/<sanitized_bucket>/<sanitized_img_name>.json`
(`REVIEW_SHARD_DIRNAME = "review"`, `packages/tcip-annotation/src/tcip_annotation/review_engine.py:83`).
The store is keyed `("bucket", "image")`, the bucket being the prediction bucket's path relative to
its dataset root as `bucket_key_of` spells it (`prediction_buckets.py:165`), folded into one
directory name by `bucket_dirname` (`review_engine.py:112`). A verdict recorded with no prediction
bucket keeps its shard directly under `review/`.
Real-world `state_dir` is `<dataset_root>/.tcip/state`, derived once by
`project_state_dir`, `packages/tcip-mcp/src/tcip_mcp/project_paths.py:9`;
`verdict_count`, `prediction_buckets.py:155`, opens a `ReviewEngine` on that root rather than
composing a state dir of its own.

Writer: `ReviewEngine._save_image`, `review_engine.py:291`, called by `mark_image_reviewed`
(`review_engine.py:328`), `unmark_image_reviewed` (`review_engine.py:364`),
`record_detection_action` (`review_engine.py:586`), `check_image_review_complete`
(`review_engine.py:697`); `save_review_state`, `review_engine.py:308`, flushes every shard.

Readers: `ReviewEngine.load_review_state`, `review_engine.py:267`, which enumerates the store's
keys (`review_engine.py:206`) at construction; `find_reviewed_entry`, `review_engine.py:489`,
and its spatial-hash cache `_build_reviewed_lookup`, `review_engine.py:475`.

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
(`packages/tcip-web/src/tcip_web/routes/projects.py:111`) on every open.

Readers: `read_last_opened`, `workspace.py:80`, read by the backend's lifespan through
`open_last_opened` (`routes/projects.py:121`), which opens the project whose record holds that id
(`project_by_id`, `workspace.py:99`); when none does it opens nothing, and the project list
names the missing id (`last_opened_problem`). The
pointer is a preference, never the MCP server's project: each server is started for one project
by `--project`.

## 24. Formats named but not exhaustively enumerated in phase0

`classifier_operating_point.json`, `ordinal_operating_point.json`, `regression_operating_point.json`
(sibling sidecars to `operating_point.json`, named in `pipelines/operating_point.py`, not opened
for this section); `pipelines/region_completeness.py`'s own digest-writing logic beyond the
`dataset_layout.py` cross-reference already given for format 8; any format defined inside
`pipelines/postprocessing/` or `pipelines/feedback/` (`selection.json` is covered in format 26). No seam id covers this placeholder entry since
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
`packages/tcip-mcp/src/tcip_mcp/pipelines/data/selection.py:309`, under whatever directory the
caller asked the partition to be written to; no dataset resolver owns this layout.

Writer: `draw_splits`, `data_tools.py:278`, when `output_path` is given. A finished run's own
drawn train/val partition freezes into the identical shape through `freeze_selection`
(`data_tools.py:18`), the second writer; both go through the one `write_selection`
(`selection.py:446`), which composes the document through `selection_document`
(`selection.py:349`) and refuses a crossing partition before anything lands, so the two writers
can never disagree on what a selection carries or write one a reader would reject. A frozen
selection also carries `origin` (`{"experiment_id", "frozen_at"}`), absent on a drawn one, the
field `read_selection` and its callers use to tell the two apart.

The record is its sample list. Each sample carries `source` (the image path, the `.bandgroup`
manifest standing in for a grouped capture, or the raster path when the sample is a region),
`ground_truth` (the path to whatever answers for it, never derived from `source`), `group` (the
key that keeps related samples together), `side` (one of `selection.SIDES`: `train`, `val`,
`calibration`), `confirmation_bucket` (the `image_status.json` key whose human confirmations
admitted it, `status_bucket` over a subject and a capture date, per sample so a selection
spanning three dates answers from three buckets), and optionally `rect` (a half-open pixel rect
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

A selection write states all three ratios (`train_ratio`, `val_ratio`, `calibration_ratio`)
non-zero, refusing whichever is zero by name, since a draw always cuts all three sides. It
refuses, before any write, when the tree holds fewer foreground groups of `subject` (and
`attribute`, when scoped) than the three sides need at minimum (one each for `train`/`val`, two
for `calibration`), counted through a subject-scoped `count_label_lines` on every draw,
independent of `stratify_foreground` (which only gates the balancing pass's own foreground
signal). `realized_ratios` records each side's delivered member share: at a floor-sized tree the
minimum pass can consume every foreground group before the balancing pass ever sees the caller's
fractions, so the delivered shares can diverge from the ratios asked for. The answer also carries
`calibration_foreground_groups`, how many of the calibration side's own groups carry foreground
at all, since the floor above is over the whole draw and the calibration slice can still land
short of two; the calibration door's own floor is where that absence bites. A stats-only call (no
`output_path`) writes nothing, admits a zero or non-zero `calibration_ratio` either way, carries
neither `calibration_foreground_groups` nor `realized_ratios` (both write-only fields), but
answers `dataset_hashes_by_date` the same way, over whatever labels directories its plain
image/label scan found, plus a single `dataset_hash` only when that scan found exactly one such
directory; over more than one, `dataset_hash` is `null` rather than one directory's hash blind to
the rest.

Readers: `selection.read_selection` (`selection.py:461`), the one reader a training or tuning
run's `data.split.selection_dir` resolves through (`split_construction.auto_train_val`) and a
selection-restricted calibration resolves through
(`splits.selection_calibration_universe`), which refuses by name when the record is
absent, undecodable, not a mapping, lists no samples, holds a sample missing any of
`source`/`label`/`group`/`side`, names a side outside `selection.SIDES`, carries a malformed
`rect`, or holds a partition whose sides cross. That last check is `refuse_crossing_sides`
(`selection.py:260`), the same one the writer runs: one source identity on two sides is the same
pixels trained on and selected on, and one group key on two sides splits the crops of one parent
across sides. `tcip plant-aware-group-splits` reads no selection back, it only writes one through
`draw_splits`.

`selection.read_selection_checked` (`selection.py:475`) is the checked variant a listing calls in
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

## 27. `cal_holdout_split_lock`, `.tcip/artifacts/cal_holdout_split_<hash>.json`

Path: named for the identity hash it locks rather than a directory of its own, addressed by
`cal_holdout_lock_key`, `packages/tcip-mcp/src/tcip_mcp/pipelines/data/splits.py:741`, under the
scope root the split was drawn over (`cal_holdout_scope_root`).

Writer: `resolve_locked_cal_holdout_split`, `splits.py:893`, locking on first draw for a given
identity hash; every later call for the same identity answers from the lock unchanged unless
`force_redraw=True`. The record carries `identity_hash`, `calibration`, `holdout`, `group_by`,
`group_key_map`, `seed`, `holdout_ratio`, `selection_dir` (`null` for a whole-directory draw,
the identity hash otherwise being `dataset_hash` over the selection's own `calibration` side
rather than the whole directory), and `redraw_history` (one entry per draw, each carrying its own
declared policy, including `selection_dir`, and the old/new content hashes). `calibration`
and `holdout` here are the two halves `cal_holdout_split` cuts from whatever universe it is
given: the selection's own `calibration` side under a selection-restricted draw, the whole
labeled directory otherwise; the lock's own field names do not change with the source.

Readers: six callers draw a lock through this one function -
`pipelines.calibration.calibrate_operating_point`, `calibration_tools.redraw_calibration_holdout`,
`calibration_tools.calibrate_scalar_operating_point`,
`measurement.scale_calibration.resolve_physical_scale`,
`pipelines.count_calibration.resolve_count_operating_point` (the one draw both
`tcip calibrate-operating-point` and `calibration_tools.calibrate_count_operating_point`
reach through), and `feedback.review_calibration.
resolve_operating_point_from_review` - each answering its own identity's lock, so a whole-directory
draw and a selection-restricted draw over the same directory coexist as two distinct locks.

A one-off operator script added `selection_dir: null` to every lock (and each
`redraw_history` entry) written before this key existed; it conformed the repo root's own
thirteen pre-existing locks once, outside the test suite, and has since been applied and
retired along with the fixture-root test that covered it.

No seam id in `seam-coverage.json`'s inventory names this record.

## 28. `coverage_grid_zoom.json`, breeder-set inspection zoom, advisory

Path: `<dataset_root>/.tcip/state/coverage_grid_zoom.json`, addressed by `coverage_grid_zoom_key`
(`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:572`), built on
`packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:66` (`_STATE_DOC` locator), `frozen: false`, the
same declared classification `view_coverage.json` carries.

Path/shape definition: `tcip_mcp.dataset_layout.coverage_grid_zoom_path`, `dataset_layout.py:546`.
Shape: `{subject: {zoom, set_by, set_at}}`, one entry per subject, no default anywhere: a subject
absent from the store has no coverage lattice until the breeder sets one.

Writer: `routes/coverage.py`'s `post_grid_zoom` (`routes/coverage.py:346`), replacing one subject's entry wholesale
under the store's own lock. Readers: `get_grid` (`routes/coverage.py:222`) and `get_completeness` (`routes/coverage.py:574`) both read
it through the shared `_subject_zoom` helper (`routes/coverage.py:239`); `get_completeness` and `post_completeness`
(`routes/coverage.py:681`) render it as the served/stored `WorkingScale` shape through the same `_working_scale_of`
helper (`routes/coverage.py:277`), so the lattice a breeder sees and the scale an attestation is judged against can
never read the same entry two different ways.

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
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:26` (`BACKEND_PORT_STORE`, declared beside the reader because the reader cannot import `tcip_web`; `backend_port_key`, `web_client.py:41`, is the one address, read at `web_client.py:342`).
Side B: `packages/tcip-web/src/tcip_web/__main__.py:51` (`replace(backend_port_key(workspace), str(port))`, publishing through that same key under the workspace `main` resolved once, and raising rather than swallowing a failure, since the fallback silently misses an OS-picked port).
Phase 3 verdict: single.

## S04. Panel-event panel vocabulary (VALID_PANELS)  <!-- queued: P5-324 unify -->

Must agree: sender and receiver accept the same set of panel names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:317` (`VALID_PANELS = frozenset(`).
Side B: `packages/tcip-web/src/tcip_web/app.py:27` (`VALID_PANELS,`).
Phase 3 verdict: duplicated.

## S05. Panel event_type vocabulary  <!-- queued: P5-272 unify -->

Must agree: the Python poster, the FastAPI hub, and the browser handler use the same event_type strings.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py:291` (`result = post_panel_event("app", "annotate_focus", payload)`).
Side B: `packages/tcip-web/src/tcip_web/app.py:314` (`if event.event_type == PANEL_EVENT_REVIEW_FOCUS:`).
Phase 3 verdict: single. The posted payload carries `subject` beside `active_subject` (`packages/tcip-mcp/src/tcip_mcp/tools/gui_tools.py:205`), the key both readers take (`packages/tcip-web/src/tcip_web/app.py:296`, `frontend/src/lib/annotateFocus.ts:21,54`), held by `tests/test_event_integration.py`'s producer-driven test, which posts the focus_human_attention tool's own event and asserts the advisory state's `active_subject`.

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
Side B: `packages/tcip-web/src/tcip_web/routes/review.py:84`
(`def _audit(scope: str, tool: str, arguments: dict) -> None:`,
which calls `record_committed` (`review.py:90`, `routes/audit_gap.py`)
with the dataset root its own guard resolved, so a failed append raises `AuditEntryNotWritten`
rather than only warning; `routes/annotate.py:120` (`record_committed(`) does the same for its own
dataset, or the open project's log for a label path outside any dataset tree;
`routes/subjects.py:49` (`record_committed(`) likewise; the GUI
inference worker writes no line of its own and publishes through the one publisher,
`routes/inference.py:155` (`pub = publish_bucket(`),
whose receipts are the library's, catching `AuditEntryNotWritten` into the job's
`audit_warning`; `routes/results.py:103`
(`record_committed(`) does the same for a project root instead. Reader:
`pipelines/postprocessing/plant_mapping.py:1535` (`_require_receipt`)
trusts only a `plant_mapping_built` entry it finds in the log under the project its caller names
(the MCP server's started project; the web backend's open project), scanned by `_scan_receipts`
(`pipelines/postprocessing/plant_mapping.py:1508`), which refuses (never scans past) a page
reporting corruption or an unknown `schema_version`.
Phase 3 verdict: single. Each writer is exercised through a real append and checked for its own
tool name landing in the log its own scope names: `tests/test_tcip_web_routes.py:766,1193,1229`
(a dataset-scoped GUI write, checked against the same dataset's log, never the platform's);
`tests/test_tcip_web_results_routes.py:680,701` (a dataset-scoped delivery-binding event beside a
project-scoped export audit line, from the one route); `tests/test_tcip_web_subjects_routes.py:724,770`
(a dataset-scoped GUI write, checked against an empty project log for the same request); and
`tests/test_audit_row_core_field_agreement.py:26`, which runs a real GUI route and a real
platform write against one log and holds the two rows to the same core fields. The receipt
gate's own agreement is held by `tests/test_plant_mapping.py:293,326` (a version-refused line
still blocks the scan; a real receipt still admits) and
`tests/test_plant_mapping_binding.py:661,709` (a receipt that cannot be written fails the build
and the web route alike).

## S07. Experiment record .tcip/experiments/<id>/

Must agree: the launching process and the run's own child agree on the run directory's layout, and each file is written by one of them once.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:133` (`def experiment_dir(` plus the file names beside it, the one declaration of the directory's path and members).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/subprocess_worker.py:68` (`def run_directory(`, the child reading the launch record `open_run` wrote and writing the resolved record and final status beside it).
Phase 3 verdict: single.

## S08. metrics.jsonl row format

Must agree: the writer's row shape is what the reader and the stream consumer expect.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:243` (`def _epoch_sink(`, the one writer; the trainer and a bespoke loop hand rows to it).
Side B: `packages/tcip-web/src/tcip_web/routes/training.py:275` (`rows, cursor = await asyncio.to_thread(read_rows, observation.metrics_log, after=cursor)`, the training stream's incremental tail read off the event loop, pushed as a `TrainingMetricFrame` per row) and `routes/tuning.py:184` (`def get_trial_metrics(`, reading a trial's log through `read_rows`, answered in the shape `_metrics_common.metrics_response` builds).
Phase 3 verdict: single. An HPO trial is a run directory, so its log is the same file shape written by the same sink.

## S10. Live GUI state .tcip/state/gui.json  <!-- queued: P5-284 unify -->

Must agree: the MCP agent reading GUI context parses the snapshot the web backend wrote.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:284` (`def write_gui_snapshot(`, the one writer, through the one address `gui_snapshot_key`, which `StateStore.mutate` in `tcip_web/state.py` calls).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/project_tools.py:213` (`state = read_gui_snapshot(project)`, the one reader, which the backend's own `open_project` also calls).
Phase 3 verdict: single.

## S11. Live canvas state files canvas_live.json / canvas_shapes.json, under the backend's open project

Must agree: the push route writes canvas_live.json/canvas_shapes.json under the project the backend has open, and capture_live_canvas reads them from the project its MCP server was started for, so a capture reads the GUI's live canvas only while the two are the same project. Both sides address the documents through one locator pair (`canvas_meta_key`, `packages/tcip-mcp/src/tcip_mcp/web_client.py:95`; `canvas_geometry_key`, `web_client.py:100`). The push carries the project id the browser drew for (`packages/tcip-web/src/tcip_web/routes/canvas.py:57`, `def push_canvas_state(`), compared with the open project's own id (`StateStore.project_id`, `packages/tcip-web/src/tcip_web/state.py:108`) before anything is written, and a panel event from the MCP side carries its project's id the same way (`web_client.py:286`), delivered only when that project is open.
Phase 3 verdict: single.

## S12. Friction reports and retrospectives under .tcip/

Must agree: the GUI reader finds, decodes and orders what the MCP writer produced.
Side A: `packages/tcip-mcp/src/tcip_mcp/tools/meta_tools.py:151` (`def report_documents(`, the one enumeration, decode and ordering of the friction reports, with `retrospective_documents`, `meta_tools.py:184`, doing the same for the retrospectives). Both stores are records, enumerated through the seam and ordered by the timestamp each document states (a report's own `timestamp` field, a retrospective's own `## Retrospective:` section headers), never by when the bytes landed. A report is one whole JSON document, not a line of a stream.
Side B: `packages/tcip-web/src/tcip_web/routes/meta.py:20` (`get_reports`) and `routes/meta.py:42` (`get_retrospectives`), both routes importing those MCP-side enumerators directly and presenting the rows they return, rather than walking a directory of their own.
Phase 3 verdict: single.

## S13. image_status carried in annotation_stats.json

Resolved: `annotation_stats.json` no longer carries an `image_status` block. Every writer
(`packages/tcip-web/src/tcip_web/routes/sessions.py`'s `_normalized`, `image_event`, `start_session`,
`end_session`) put an empty `{}` there because the shape guard defaulted an absent document to it,
and nothing ever read it back out: the dataset's real confirmed-negative store is
`image_status.json`, addressed by `image_status_key`, the seam this entry once asked to agree
with. The key stopped being written; a project's existing record still carrying it was corrected
by a one-off operator script, since applied and retired.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:693` (`def is_confirmed_negative(`, the one membership predicate, beside `is_finished_status` at `:926` for the wider complete-or-negative pair; `status_confirmations` is the one decoder of the stored records, and `status_tokens` its status-token projection, instead of inline re-implementations).
Side B: `packages/tcip-web/src/tcip_web/routes/subjects.py` (`set_image_status`, writing through the registered store).
Phase 3 verdict: single. `packages/tcip-web/src/tcip_web/routes/sessions.py:250` (`if is_confirmed_negative(status):`, session time classification) calls the same predicate against the real `image_status.json` rather than restating it.

## S14. dataset_layout.py as the on-disk path resolver

Must agree: agent writes and GUI reads resolve to the same files.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:118` (`def image_root(`, with `annotation_root`/`prediction_root` and the dated dir calls built on them, plus `bucket_subject_date`, `dataset_layout.py:595`, as the published inverse of `status_bucket`).
Side B: `packages/tcip-web/src/tcip_web/routes/dataset.py` (`select_dataset` resolves every directory through the resolver; `tcip_mcp.cli.doctor`, `data_tools`, `project_tools` and `annotation_tools` no longer re-spell the tree).
Phase 3 verdict: single.

## S15. Per-image label filename convention

Must agree: the browser's label path and the Python resolver's label path name the same file.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:908` (`def label_filename(`, with `annotation_path`/`prediction_path` built on it).
Side B: `packages/tcip-web/frontend/src/lib/paths.ts:24` (`labelPath`, the browser's one join site over the directories the backend resolves; a gate test pins the record extension against the resolver).
Phase 3 verdict: single. The browser still joins directory plus filename client-side at that one site; handing fully resolved per-image paths across the API would add a backend round trip to image navigation, an open owner question in the batch report.

## S16. ReviewEngine shard-store directory

Must agree: the verdict writer and the bucket-immutability reader look at the same review store
(a bucket here is format 17's immutability-scoped prediction directory, not a score bin).
Side A: `packages/tcip-mcp/src/tcip_mcp/project_paths.py:9` (`def project_state_dir(`, the one derivation of the store root; `verdict_count`, `prediction_buckets.py:155`, counts one bucket's verdicts through the store the engine writes into).
Side B: `packages/tcip-annotation/src/tcip_annotation/review_engine.py:163` (`REVIEW_VERDICTS_STORE`, which owns the shard layout inside that root). `packages/tcip-web/src/tcip_web/routes/review.py:72`, `routes/inference.py:428` and `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:1285` open on the derived root instead of composing a state dir each.
Phase 3 verdict: single.

## S17. Canonical per-image annotation JSON schema

Must agree: every writer produces, and every reader accepts, the same name-based record shape.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:344` (`def annotation_from_payload(`, the one conversion from a client payload to a record, with `json_io.py:425` (`_annotations_of`) the one parse back and `json_io.py:763` (`write_annotations`) the one writer).
Side B: `packages/tcip-web/src/tcip_web/routes/annotate.py:191`, `packages/tcip-web/src/tcip_web/routes/review.py:667` and `packages/tcip-mcp/src/tcip_mcp/tools/annotation_tools.py:179` (every save entry point converts through it rather than assembling records of its own).
Phase 3 verdict: single.

## S18. Bounding-box coordinate convention across the HTTP boundary

Must agree: the browser and the route use corner coordinates while the file uses xywh, with the conversion happening once.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:46` (`bbox: Optional[list[float]] = None          # [x1, y1, x2, y2], pixel`, the wire form).
Side B: `packages/tcip-annotation/src/tcip_annotation/json_io.py:650` (`def xywh(`, the one corner-to-xywh conversion and the 2-decimal grid the stored document lives on, applied on write and, via the import at `packages/tcip-mcp/src/tcip_mcp/pipelines/training/evaluation.py:26` (`from tcip_annotation.json_io import xywh`), to every box scored against a stored label so both sides of a match sit on one grid; a wire box becomes a `BBox` in `annotation_from_payload` (`json_io.py:344`), and `_annotations_of` (`json_io.py:425`) is the inverse read).
Phase 3 verdict: single.

## S19. Annotation format detection scope (json, coco)

Must agree: a label document's shape is decided by its one reader, never guessed beside it.
Side A: `packages/tcip-annotation/src/tcip_annotation/json_io.py:146` (`def parse_label_document(text: str, *, source: str) -> dict:`), which refuses a dataset-level COCO and the old `objects` schema for every reader of a per-image document.
Side B: none; admission, the doctor and the read tool take that reader's answer.
Phase 3 verdict: single.

## S20. subjects.json subject registry

Must agree: the GUI editor, the path resolver, and the training loader read one registry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py:4` (`The on-disk registry (`` `<dataset_root>/subjects.json` ``) is self-describing and name-based::`).
Side B: `packages/tcip-web/src/tcip_web/routes/subjects.py:139` (`from tcip_mcp.dataset_layout import subjects_path`).
Phase 3 verdict: single.

## S21. Training name-to-id assignment versus inference decode map

Must agree: a prediction's integer label decodes to the class name the run trained it as.
Side A: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py:446` (`def assign_class_ids(`, the one assignment, reached by the loader through `pipelines/data/label_queries.py:69` (`return registry, subject_registry.assign_class_ids(registry, subject, attribute)`)).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:728` (`scope=ClassScope.of(checkpoint.data_config)`, the checkpoint's own recorded `scope`, map included, taken by the prepared pass every inference regime and the GUI worker share; both calibrations read it off that pass, block calibration at `pipelines/block_calibration.py:226` (`scope = p.scope.admitted_for(DOCUMENT`); nothing re-derives a map from a live registry).
Phase 3 verdict: single.

## S22. image_status.json confirmed-negative store

Must agree: a negative is empty labels plus an explicit human Complete, every consumer applies the same bucket keying and status vocabulary, and each stored status carries the actor who set it and when, so a person's Complete and a status a harvest wrote stay distinguishable.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:673` (`def derive_status(`, with `dataset_layout.py:656` (`IMAGE_STATUSES`) as the one vocabulary, `dataset_layout.py:603` (`status_of`) as the one predicate for what the store holds, and `record_image_statuses`/`replace_image_status_store` as the two declared writers, both through the registered store).
Side B: `packages/tcip-web/src/tcip_web/routes/subjects.py` and `routes/review.py` call `derive_status`; the browser imports one `ImageStatus` type from `api/subjects.ts`, pinned against the Python vocabulary by a gate test.
Phase 3 verdict: single.

## S23. image_status_digest.json attribute-schema stamp

Must agree: writer and reader compute the digest the same way for a stale stamp to be detectable.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:774` (`def stamp_image_status_digests(`, the one transactional read-merge writer, called by the web route, the materializer, the split tools and the schema-change sweep `subject_registry._sweep_schema_change`, which passes `only_unstamped` so a confirmation-time stamp is never re-dated).
Side B: `packages/tcip-mcp/src/tcip_mcp/subject_registry.py:159` (`attribute_schema_digest`, the one digest computation).
Phase 3 verdict: single.

## S24. view_coverage.json advisory coverage record

Must agree: backend store shape and the browser's coverage payload match, keyed by status_bucket.
Side A: `packages/tcip-web/src/tcip_web/routes/_coverage_models.py:12` (`class GridGeometry`) through `routes/_coverage_models.py:92` (`class CoverageRecord`), the one declaration `routes/coverage.py`'s `post_coverage`/`get_coverage` validate against.
Side B: `tools/generate_frontend_types.py`, rendering `packages/tcip-web/frontend/src/api/types.generated.ts` from Side A; the browser imports the generated types (`lib/coverageTracker.ts`) rather than hand-declaring its own, held current by `tests/test_generated_frontend_types.py` and round-tripped through the real route by `tests/test_coverage_routes.py`'s `test_post_from_a_plain_rgb_view_round_trips`/`test_post_from_a_composite_view_round_trips`.
Phase 3 verdict: single.

## S25. region_completeness.json attestation store

Must agree: an attestation written by the GUI is readable, and staleness-checkable, by the calibration path that relies on it.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:489` (`def region_completeness_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/region_completeness.py:152` (`def stale_cells(`).
Phase 3 verdict: restated-in-test.
Differs from phase0 record: phase0 cited a line inside the function's body rather than its header; the function itself is defined at `region_completeness.py:152` (`def stale_cells(`).

## S26. dataset.json identity and fingerprint

Must agree: the stored fingerprint and the recomputed one cover the same inputs.
Side A: `packages/tcip-mcp/src/tcip_mcp/dataset_layout.py:312` (`def dataset_identity_path(dataset_root: str | Path) -> Path:`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/dataset_fingerprint.py:138` (`def dataset_fingerprint(dataset_root: str | Path) -> str | None:`).
Phase 3 verdict: single.

## S27. Trained-model registry .tcip/models/registry.json

Must agree: the MCP registrar and the GUI model pickers read one registry entry shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/model_registry.py:121` (`def registered_entries(`, the one read every consumer goes through: one owner per sha256, the completed run whose final status names the bytes, else the index's foreign entry; `_write_registry_entry`, `model_registry.py:293`, replaces one foreign entry by sha256 inside one `tcip_store.transaction` on the key `registry_index_key`, `model_registry.py:85`, mints).
Side B: `packages/tcip-web/src/tcip_web/routes/results.py:1003` (`@router.get("/models/registered")`, serving `model_tools.rank_registered_models`'s listing view) and the browser's one entry declaration, `packages/tcip-web/frontend/src/api/inference.ts:29` (`export interface RegisteredModel {`), held field by field against an entry the real registrar wrote by `tests/test_registry_entry_shape_agreement.py`.
Phase 3 verdict: single.

## S28. operating_point.json prediction-bucket sidecar

Must agree: every writer stamps, and every consumer finds, the same provenance keys next to a bucket's predictions.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:828` (`def operating_point_stamp(`, the one stamp constructor, every field required of every producer; the stamp's whole key set is declared beside it as `STAMP_KEYS`, the constructor's own, plus `STAMP_EXTENSION_KEYS`, the producer-local additions each named for the producer that writes it; `write_sidecar`, `resolution.py:771`, and `update_sidecar`, `resolution.py:796`, are the only writers, both through `sidecar_key`, `resolution.py:671`, both refusing an unearned validation claim, and for this document refusing a top-level key outside that declared union: a fresh mint on its whole body, an update on the whole merged body it would store). A validated claim is earned in two phases beside the resolvers it selects among: `open_validation`, `resolution.py:1290`, runs the document's own resolver over the evidence and refuses a result that cleared no accepted reference, and `seal_validation`, `resolution.py:1446`, takes the covered buckets' content identity from the files as they landed, files the row through the experiment record's validations member, and returns the stamp body with its pointer merged in.
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:925` (`read_operating_point_sidecar`, the one reader, with `verify_stamp_binding`, `resolution.py:1641`, deciding inside every reconciler whether a claiming stamp is answered for). The producers are the per-image publisher every per-image door publishes through, the GUI's inference worker included, and the raster pass, `tools/inference_tools.py:1310` and `tools/inference_tools.py:991` (`op_stamp = operating_point_stamp(`); the review promotion merges into the stored stamp under its lock at `routes/validation.py:407`.
Phase 3 verdict: single.

## S29. Prediction-bucket immutability

Must agree: no writer overwrites a bucket whose predictions already carry human review verdicts,
or, for the staging door alone under `count_review_state`, any review state at all (a bulk accept
included); scoped, for the opted-in publishers only (`run_inference`, `deliver_per_image_counts`'s
live path, and the web route's own launch), to a second agreement that no writer publishes into a
bucket that already holds a prediction document with no verdict yet recorded, whatever
`overwrite` says.
Side A: `packages/tcip-mcp/src/tcip_mcp/prediction_buckets.py:321` (`def resolve_writable_bucket(`, the one guard, its `refuse_documents` keyword the document agreement's opt-in and its `count_review_state` keyword the staging door's own wider reading's opt-in; `bucket_stems`, `prediction_buckets.py:44`, excludes every provenance stamp through `tcip_annotation.json_io.prediction_documents` rather than naming one filename).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/proposal_tools.py:444` (`from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes`) and `tools/proposal_tools.py:589` (`from tcip_mcp.prediction_buckets import BucketHasVerdicts, stage_prediction_shapes`, both leaving `refuse_documents` at its default off and reaching `stage_prediction_shapes`, which turns `count_review_state` on), `tools/inference_tools.py:753` (`_resolve_writable_bucket_for`, passing `refuse_documents=True` on every branch) and `packages/tcip-web/src/tcip_web/routes/inference.py:263` (`_resolve_writable_bucket_for(`, the route resolving its bucket through that same function).
Phase 3 verdict: single.

## S30. split.json train/val partition

Must agree: the calibration holdout is disjoint from the split the run actually trained on, and,
when a selection is in play, from the checkpoint's own selection (val) side too.
Side A: `packages/tcip-mcp/src/tcip_mcp/experiments.py:440` (`def run_resolution(`, the one reader of the partition the launcher resolved and wrote into the run's `run.json`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/block_calibration.py` (precheck and resolver share one spatial-strip predicate over that reader) and `pipelines/operating_point.py` (`_train_disjointness` and `_selection_disjointness` both read through it and share `_resolve_group_stem_disjointness`, the one group/stem-overlap implementation).
Phase 3 verdict: single.

## S31. Checkpoint payload structural markers

Must agree: a checkpoint written by the training envelope is kind-routable and rebuildable by the predictor that later loads it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:24` (`MODEL_SOURCE_KEY` and `STATE_DICT_KEY`, the one key vocabulary).
Side B: the three checkpoint writers in `pipelines/training/generic_trainer.py` and the readers (`generic_predictor.py`, `inference/predictor.py`, `training/eval_runners.py`) all bind through the constants.
Phase 3 verdict: single.

## S32. Single operating-point resolution for all consumers

Must agree: the same model and images yield the same conf/NMS/max_dets/tile whichever entry point asks for them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py:735` (`def resolve_operating_point(`, the calibrated regime; a caller-supplied `max_dets` earns a derivation label only by naming where it came from, and otherwise records itself as a caller override).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:489` (`raw_operating_point`) and `resolution.py:527` (`block_calibrated_export_operating_point`), the two uncalibrated regimes. Every entry point takes its bundle from one of the three: `tools/inference_tools.py:304,797,982,1001`, `packages/tcip-web/src/tcip_web/routes/inference.py:234`, `pipelines/training/envelope.py:213`, `pipelines/training/eval_runners.py:231` (the full-frame regime).
Phase 3 verdict: single.

## S33. Shared inference defaults DEFAULT_CONF / DEFAULT_NMS_IOU / DEFAULT_MAX_DETS

Must agree: the MCP entry point and the GUI entry point start from the same unresolved defaults, and both read a caller's unstated parameter off the `None` sentinel rather than off equality with the default, so a caller who states the default value is honored as an override instead of being resolved as if they had stated nothing.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:148` (`DEFAULT_CONF = 0.5`, with `DEFAULT_NMS_IOU`, `DEFAULT_OVERLAP` and `DEFAULT_MAX_DETS` declared beside it).
Side B: `packages/tcip-web/src/tcip_web/routes/inference.py:291` (`conf=payload.conf,`: the GUI launch carries every stated value unresolved, `None` where omitted, and its worker resolves them through the MCP pass's own `packages/tcip-mcp/src/tcip_mcp/tools/inference_tools.py:975`, `def _prepare_pass(`) and `packages/tcip-mcp/src/tcip_mcp/pipelines/training/eval_runners.py:15` (the tile-level regime's own default conf, `DEFAULT_CONF`). The full-frame runner, `evaluate_model`, and `run_inference`'s verified body and raster branch all resolve a stated-or-default conf and cap through one function, `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:178` (`def applied_operating_point(`), and a stated, derived or default merge threshold through one other, `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:187` (`def resolve_cross_tile_nms(`), so the point a run is selected at starts where the point it ships at does; the cap parameters of `run_inference` and `deliver_per_image_counts` default to `None`, the shared constant supplies the pass, and the unstated parameter travels to the resolver as unstated so it can derive one from the data.
Phase 3 verdict: single.

## S34. check_delivery_gate behind every delivery path

Must agree: no delivered result ships an unvalidated parameter without an explicit acknowledgment.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2861` (`def check_delivery_gate(`, judging each dimension against `_DIMENSION_REFERENCES`, `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2841`, so a reference clears only the dimension whose kind earned it, with `DeliveryGateResult.column_stamp`, `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2769`, as the one derivation of what a deliverable's validity column carries).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:2008` (`def delivered_tail(`, the one composition every delivered tail's validity columns route through, deriving `own_column` from which of `_DIMENSION_TO_COLUMN`'s columns a door's own column list carries, never a second list stating so). `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py:341` and `pipelines/postprocessing/aggregation.py:478` (`stamp = delivered_tail(provenance, (operating_point_recon or {}).get("bindings", {}), gate,`, each delivery door composing its tail through it rather than stamping the column itself); `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/export.py:330` (`summary = {`, `export_detection_csv`'s own gate-and-reconciliation summary, `tools/inference_tools.py`'s `deliver_per_image_counts` sourcing its `tile_size_validated` response field from that returned summary's own stamp in both regimes that read a bucket, while `operating_point_validated` sources from the delivered tail's own floored cell above instead, neither response field re-deriving its column itself); and `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:580` (`delivered_tail(`, inside `_write_phenology_delivery`, the one writer both phenology delivery doors call through, `tools/phenology_tools.py`'s `deliver_phenology_milestones` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv`, rather than stamping the column themselves). The aggregated per-plant door also floors a claim-scope dimension read from each bucket's sidecar (`resolution.reconcile_claim_scope_validity`, whose accepted values, `CLAIM_SCOPE_REFERENCES`, are narrower than `VALIDATED_SHIPPABLE`, so an annotation reference cannot clear a raster-scope claim): a bucket that records no claim scope never acquires the dimension.
Phase 3 verdict: single.

## S35. ResolvedParam validation firewall

Must agree: a parameter needing validation is un-consumable as a bare number unless checked against the right kind of reference.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:228` (`class ResolvedParam:`; `.value` raises at `resolution.py:293` (`raise UnvalidatedOperatingPointError(`), and `unvalidated_value`, `resolution.py:301`, is how a door reads an unvalidated number).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/resolution.py:106` (`_ACCEPTED_REFERENCES`, whose geometry entry is built from `_GEOMETRY_REFERENCE_BY_SOURCE`, `resolution.py:86`, the one statement of which tile-size source earns which reference; `tile_size_source_of`, `resolution.py:125`, reads it back the other way).
Phase 3 verdict: single. One read of the raw value survives outside the class, in `packages/tcip-mcp/src/tcip_mcp/cli/calibrate_operating_point.py:125`'s console line.

## S36. Count-objective vocabulary versus registered pickers

Must agree: every named count objective has a registered picker function.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:31` (`COUNT_OBJECTIVES`, over the three names declared at `traits.py:28` (`COUNT_UNBIASED = "count_unbiased"`)).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/operating_point.py:71` (`COUNT_OBJECTIVE_PICKERS`, with the reconciling `operating_point.py:76` (`assert set(COUNT_OBJECTIVE_PICKERS) == COUNT_OBJECTIVES,`)). A picker's provenance label, and the review-verdict variant it earns through `REVIEW_VERDICT_LABEL_SUFFIX`, `operating_point.py:80`, are read off that registry by `pipelines/derivations.py:640`, so registering a picker registers its labels.
Phase 3 verdict: single.

## S37. Trait entries against crops.yml controlled vocabulary

Must agree: a trait entry's delivered phenotypes exist in the crops.yml vocabulary.
Side A: `packages/tcip-mcp/src/tcip_mcp/knowledge/__init__.py:124` (`def crops_yml_path(`, the one placement of `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml`), reached through `packages/tcip-mcp/src/tcip_mcp/traits.py:94` (`def crops_yml_path(`, delegating), read whole or raising by `_crops_traits`, `traits.py:101`.
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:212` (`def check_proposed_entry(`, a proposal's one check of `delivers` against that read; a stored record decodes without it) and `tools/verify_skill_traits.py:26` (`load_vocab` checks a skill's trait tokens through that same read).
Phase 3 verdict: single.

## S38. Per-project trait records .tcip/state/traits/*.json

Must agree: the proposing tool, the confirmation door, the delivery doors and the GUI trait list read one record per trait and agree on which revision a delivery ships under.
Side A: `packages/tcip-mcp/src/tcip_mcp/traits.py:323` (`def trait_key(`, the one placement, with `TRAITS_STORE`, `traits.py:308`, the store every reader and writer addresses).
Side B: `packages/tcip-mcp/src/tcip_mcp/traits.py:398` (`def propose_trait(`, the one append) and `packages/tcip-mcp/src/tcip_mcp/operationalization.py:69` (`def confirmed_revision(`, the one read of the latest confirmed revision every delivery door makes). `packages/tcip-web/src/tcip_web/routes/results.py:922` (`def list_traits(`) and `packages/tcip-mcp/src/tcip_mcp/cli/doctor.py:463` (`def check_traits(`) read the same record through `read_trait`.
Phase 3 verdict: single.

## S39. Phenology CSV column vocabulary

Must agree: the delivered CSV's column names derive from the trait spec on every path that writes them.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:59` (`def phenology_csv_columns(spec) -> list[str]:`, the one owner).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/postprocessing/phenology.py:635` (`phenology_csv_columns(revision.entry)`, handed to `_write_phenology_delivery`, the one writer both `tools/phenology_tools.py`'s `deliver_phenology_milestones` and `packages/tcip-web/src/tcip_web/routes/results.py`'s `export_csv` call through instead of assembling the names themselves).
Phase 3 verdict: single.

## S40. Per-band normalization stats for a non-3-channel detector

Must agree: the values passed as image_mean/image_std are per-band stats of the same length as in_chans.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/derivations.py:420` (`def band_normalization_stats(`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/components/detectors.py:64` (`def _normalization(adapter: Any, in_chans: int | None, image_mean, image_std,`).
Phase 3 verdict: single.

## S41. model_source bespoke build seam  <!-- queued: P5-320 unify -->

Must agree: the dict an agent writes into the config carries the keys the builder, the snapshotter, and the predictor all read.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:24` (`MODEL_SOURCE_KEY = "model_source"`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/inference/generic_predictor.py:86` (`self.model_source = ckpt.get("model_source")`).
Phase 3 verdict: duplicated.

## S42. training_source bespoke train(ctx) seam  <!-- queued: P5-321 unify -->

Must agree: a bespoke train(ctx) callable is importable and accepts the TrainContext the envelope hands it.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/envelope.py:361` (`training_source = run.config.get("training_source")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py:418` (`training_source = normalized.get(TRAINING_SOURCE_KEY)`).
Phase 3 verdict: duplicated.

## S43. dataset_source bespoke dataset seam

Must agree: the builder the reader resolves off `data.dataset_source` returns a Dataset the trainer's loaders accept.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_build.py:307` (`dataset_source = (config.get("data") or {}).get(DATASET_SOURCE_KEY)`).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/data/datasets.py:1092` (`def build_dataset(`).
Phase 3 verdict: duplicated.

## S44. Model-contract smoke batch versus the trainer's real batch

Must agree: the smoke batch has the same shape the trainer actually feeds model.forward for the task.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/model_contract.py:72` (`def _synth_batch(`, which synthesizes per-sample `(image, target)` items shaped like a dataset's `__getitem__` and hands them to the trainer's own collate).
Side B: `packages/tcip-mcp/src/tcip_mcp/pipelines/training/collation.py:35` (`def task_collate(task: str):`, the collate the DataLoader assembles the training batch with).
Phase 3 verdict: single.
Differs from phase0 record: phase0 cited a line inside the function's body rather than its header; the function is defined at `model_contract.py:72` (`def _synth_batch(`).

## S45. Review verdicts promoted into a calibration reference

Must agree: a breeder-confirmed sample reaches the operating-point gate evidence in the same record shape GT annotations do, and passes the same gate.
Side A: `packages/tcip-annotation/src/tcip_annotation/review_engine.py:586` (`def record_detection_action(`, the one writer of a stored verdict entry).
Side B: `packages/tcip-annotation/src/tcip_annotation/verdicts.py:89` (`decode_verdict`, the one read of that entry, over the affirming actions declared at `verdicts.py:23` (`POSITIVE_ACTIONS`)), called by `pipelines/feedback/review_calibration.py:225` for the calibration reference, `pipelines/feedback/materialize.py:101` for the curated dataset, and the engine's own lookup and the review routes. What each consumer then emits from the affirmed box (COCO xywh scaled by the image, pixel corners for a label file) stays its own.
Phase 3 verdict: single.

## S46. Frontend api/ layer against backend route paths

Must agree: every URL the browser builds matches a registered FastAPI route path and method.
Side A: `packages/tcip-web/frontend/src/api/routes.ts` (generated: the browser's only copy of the paths, each named for its method).
Side B: `packages/tcip-web/src/tcip_web/routes/__init__.py` (`register_all` mounts 17 routers with fixed prefixes).
Phase 3 verdict: single. The api/ helpers keep their hand-written signatures and reference a generated name; `tools/generate_frontend_routes.py` projects the registered routes into that module, and `tests/test_frontend_route_paths.py` fails when the projection is stale or a call site writes a path of its own.

## S47. GuiState shape between state.py and store/types.ts  <!-- queued: P5-287 unify -->

Must agree: the snapshot the backend serializes deserializes into the store's typed shape.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:202` (`class GuiState(_GuiFields):`).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts:444` (`export interface GuiState {`), generated from Side A by `tools/generate_frontend_types.py` and re-exported by `store/types.ts`.
Phase 3 verdict: single.

## S48. State WebSocket snapshot protocol  <!-- queued: P5-288 unify -->

Must agree: the browser knows which slices of a broadcast snapshot are backend-authoritative and orders them by version.
Side A: `packages/tcip-web/src/tcip_web/app.py:166` (`@app.websocket("/ws/state")`).
Side B: `packages/tcip-web/src/tcip_web/state.py:157` (`def version(self) -> int:`, "Monotonic version, bumped on every state change.").
Phase 3 verdict: duplicated.

## S49. Terminal PTY WebSocket protocol  <!-- queued: P5-289 unify -->

Must agree: control-message type names and field names match, and output frames are treated as raw text rather than JSON.
Side A: `packages/tcip-web/src/tcip_web/routes/terminal.py:476` (`@router.websocket("/ws/{session_id}")`).
Side B: `packages/tcip-web/frontend/src/components/TerminalRail.tsx:302` (`send({ type: "input", data });`).
Phase 3 verdict: duplicated.

## S50. Inference job stream WebSocket  <!-- queued: P5-304 unify -->

Must agree: the browser recognizes the terminal frame and the status vocabulary the backend uses.
Side A: `packages/tcip-web/src/tcip_web/routes/inference.py:323` (`@router.websocket("/jobs/{job_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/experiments.py:94` (`TERMINAL_STATES = frozenset({*FINAL_STATES, "interrupted"})`, the one declaration the job registry and the generated frontend vocabulary both read).
Phase 3 verdict: duplicated.

## S51. Training run stream WebSocket  <!-- queued: P5-297 unify -->

Must agree: the status payload the MCP tool returns is renderable by the browser's training view.
Side A: `packages/tcip-web/src/tcip_web/routes/training.py:290` (`@router.websocket("/runs/{experiment_id}/stream")`).
Side B: `packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py` (`monitor_training` supplies the status payload).
Phase 3 verdict: duplicated.

## S52. Image-serve response headers  <!-- queued: P5-301 unify -->

Must agree: header names and value encodings match.
Side A: `packages/tcip-web/src/tcip_web/routes/images.py:677` (`"X-TCIP-Stats-Source": json.dumps(stats_source.model_dump(), allow_nan=False),`).
Side B: `packages/tcip-web/frontend/src/lib/imageLoader.ts:55` (`const statsSourceRaw = headers.get("X-TCIP-Stats-Source");`).
Phase 3 verdict: duplicated.

## S53. Optimistic-concurrency token for label saves

Must agree: the token the browser echoes is the same token the backend minted for that label file.
Side A: `packages/tcip-web/src/tcip_web/routes/annotate.py:151` (`"base_mtime": token,`, the token the load route mints; the save route compares the echoed one at `routes/annotate.py:200`).
Side B: `packages/tcip-web/frontend/src/tabs/AnnotateTab.tsx:467` (`base_mtime: paths.mtime,`).
Phase 3 verdict: single.

## S54. Built frontend bundle location  <!-- queued: P5-305 unify -->

Must agree: the directory Vite writes is one of the directories the backend looks in.
Side A: `packages/tcip-web/frontend/vite.config.ts:26` (`outDir: "../static",`).
Side B: `packages/tcip-web/src/tcip_web/app.py:223` (`def _find_static_dir() -> Path:`).
Phase 3 verdict: duplicated.

## S55. Vite dev-server proxy prefixes  <!-- queued: P5-306 unify -->

Must agree: every backend path the browser calls in dev falls under a proxied prefix.
Side A: `packages/tcip-web/frontend/vite.config.ts:20` (`proxy: {`).
Side B: `packages/tcip-web/src/tcip_web/app.py:166` (`@app.websocket("/ws/state")`, one of the endpoints not under the `/api` prefix).
Phase 3 verdict: duplicated. The prefix literals still stand on their own, but `tests/test_frontend_route_paths.py` now fails when a path the frontend references falls outside them, sockets under the API prefix included.

## S56. Tab-name vocabulary  <!-- queued: P5-290 unify -->

Must agree: the tab a panel event targets, the tab the browser can restore, and the tab the backend persists are the same set of names.
Side A: `packages/tcip-mcp/src/tcip_mcp/web_client.py:127` (`ActiveTab = Literal["annotate", "review", "training", "tuning", "inference", "results", "meta"]`, with `TAB_NAMES = get_args(ActiveTab)` beside it, `tcip_web.state` importing both).
Side B: `packages/tcip-web/frontend/src/api/types.generated.ts:17` (`export const TAB_NAMES = [`, generated from the same declaration).
Phase 3 verdict: duplicated.

## S57. Review match computation and its response shape

Must agree: the TP/FP/FN classification the browser draws is the one the matching library computed.
Side A: `packages/tcip-annotation/src/tcip_annotation/matching.py` (`compute_matches` / `compute_classified_trait_matches`).
Side B: `packages/tcip-web/src/tcip_web/routes/review.py:283` (`class MatchesResponse(BaseModel):`).
Phase 3 verdict: restated-in-test.

## S58. Reference-grid geometry

Must agree: the cell name the agent points at and the cell the GUI highlights are the same rectangle.
Side A: `packages/tcip-mcp/src/tcip_mcp/pipelines/reference_grid.py:41` (`def reference_cells(`, which builds the cells, with `grid_geometry`, `reference_grid.py:122`, the geometry handed over beside them).
Side B: `packages/tcip-annotation/src/tcip_annotation/sam_wrapper.py:319` (`def grid_to_rect(`, the one cell-name lookup, with `grid_to_pixel`, `sam_wrapper.py:347`, built on it) and `packages/tcip-web/src/tcip_web/routes/coverage.py:221` (`@router.get("/grid")`, `get_grid`, whose cell list the browser consumes verbatim).
Phase 3 verdict: single.

## S59. Path confinement (the derived allow-set)

Must agree: every route that accepts a client-supplied path confines it to the same allowed roots.
Side A: `packages/tcip-web/src/tcip_web/paths.py:41`
(`def allowed_roots() -> list[Path]:`).
Side B: `packages/tcip-web/src/tcip_web/routes/annotate.py:76` (`p = allowed_path(path)`, the one adapter every route shares, `paths.py:147`).
Phase 3 verdict: single.

## S64. MCP tool registry against documented tool names  <!-- queued: P5-303 unify -->

Must agree: any document naming a tool names one the server actually registers.
Side A: `packages/tcip-mcp/src/tcip_mcp/server.py:99` (`def list_registered_tools() -> list[str]:`).
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
