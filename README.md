# TCIP Agent

An agentic ML/CV system for automated phenotyping in tree crop breeding programs, entering alpha testing. A Claude agent (ML/CV engineer persona) drives annotation, model training, inference, and per-plant result delivery through an MCP tool server, while a browser-based GUI supports human annotation, review, and training oversight. The system is freestanding: the MCP server is a transport-neutral stdio server (any MCP client) and the GUI is a standalone browser app; no editor required.

Who it's for: plant breeders, not CV engineers. They know their crops and traits; they don't write PyTorch or make CV decisions. The objective is to replace the CV scientist in the loop: an agent that guides a breeder end to end, from imagery plus a trait to a validated per-plant phenotype, maintaining scientific rigor on the breeder's behalf. There are two distinct users and two UX surfaces: the agent's UX is the tools/skills/code/docs/MCP API surface it reasons through (see `CLAUDE.md`), and the breeder's UX is the browser GUI, the only surface they experience the platform through (label a few examples, confirm/correct the model, receive the result).

Current scope: 2D imagery (RGB + N-channel) from any capture modality, object detection first. Two different capture shapes are supported: an ordinary per-plant image (one photo of one or a few plants, from a phone, DSLR, GoPro, or ground rig) and a single large georeferenced orthomosaic covering many plants in one file (a drone survey mosaic, potentially tens of gigabytes, read via windowed/tiled access rather than loaded whole, with each detection resolved to a real-world coordinate and matched to the plant it belongs to). The data layer reads RGB and multi-band rasters (GeoTIFF / NPZ / grayscale) for both shapes; 3D point-cloud (LiDAR / SfM) support is not built yet (see [Roadmap](#roadmap)).

Six crops in scope: hazelnut, chestnut, currant, elderberry, persimmon, black locust. Phase 1 target is hazelnut catkin phenology.

Status: the browser GUI is built out across all tabs (Annotate / Review / Training / Tuning / Inference / Results / Meta), the MCP tool surface is in place (run `python tools/list_tools.py` for the current count), and every delivered phenotype (a count, a dimensional measurement, a phenology milestone) is gated on validated measurement inputs end to end: an uncalibrated confidence threshold, an unproven physical scale, or a fabricated tile geometry refuses delivery. Phase 1 focus remains hazelnut catkin phenology; the orthomosaic capability described above is built, not yet exercised end to end against a first live delivery.

## Architecture

```
┌──────────────────────────────┐
│  Claude agent (any MCP host) │  ML/CV engineer persona
│  (see CLAUDE.md)             │  designs pipelines, trains, evaluates
└──────────┬───────────────────┘
           │ MCP (stdio)
           ▼
┌──────────────────────────────┐
│  Python MCP server           │  domain tools: data, annotation,
│  (packages/tcip-mcp)         │  training, inference, experiments, viz
└──────────┬───────────────────┘
           │ HTTP
           ▼
┌──────────────────────────────┐
│  FastAPI + React GUI         │  Annotate / Review / Training / Tuning /
│  (packages/tcip-web)         │  Inference / Results / Meta tabs
└──────────────────────────────┘
```

All three processes share `.tcip/` on disk (experiment state, model registry, audit logs, GUI state).

Supporting libraries: `packages/tcip-annotation` (headless annotation engine: label I/O, IoU matching, SAM wrapper) and `packages/tcip-store` (the storage seam: one locked, atomic interface for the platform's records, append-only logs and blobs). `tcip-store` is the bottom of the stack, depending on nothing else here; `tcip-annotation` depends on it and on neither of the other two.

Records and append-only logs go into one SQLite database per root, `<root>/.tcip/store.db`; blob bytes stay files under every backend, so imagery, labels and predictions travel with the dataset. Set `TCIP_STORE_BACKEND=file` to bind the file backend instead, which reads and writes the same records as loose files. A root whose records are still loose files is refused rather than read as empty: `tcip adopt-store <root>` moves them into a database, and `tcip export-store <root>` writes them back out.

## Repository layout

```
CLAUDE.md                      # agent operating contract (persona, invariants, conventions)
packages/
  tcip-mcp/                    # MCP server (python -m tcip_mcp)
    src/tcip_mcp/
      knowledge/               # domain knowledge modules (crops, annotation, training, ...)
      tools/                   # domain tools (repo-root tools/list_tools.py prints the current list)
      pipelines/               # ML: model_build, trainer, predictor (windowed/tiled raster
                               #   reads), envelope + plain nn.Module blocks; postprocessing
                               #   (per-image plant mapping + phenology, plus orthomosaic
                               #   georeferencing)
  tcip-annotation/             # headless annotation library
  tcip-store/                  # storage seam: keyed records, append-only logs, blobs
    src/tcip_store/            # the contract (model, errors, registry, store) + file backend
  tcip-web/                    # FastAPI backend + React frontend
    src/tcip_web/
      routes/                  # annotate, review, training, tuning, inference, results, ...
    frontend/src/              # Vite + React 18 + TypeScript + Tailwind + Konva
tools/                         # CI + development tooling, never a project-facing command; see tools/README.md
tests/                         # pytest suite
data/                          # sample hazelnut dataset (gitignored)
```

## Setup

```bash
# Python: creates the env and installs the four packages (editable). Run from the repo root.
# Installs the CUDA (cu126) torch wheel by default (environment.yml's --extra-index-url); runs fine without a GPU too.
conda env create -f environment.yml
conda activate tcip-agent

# Frontend
cd packages/tcip-web/frontend
npm install
```

`docker build -f packages/tcip-web/Dockerfile` plus `docker run --network host` builds and runs the platform as a container, serving the GUI on loopback only. Host networking is a Linux Docker feature; it is not available on Docker Desktop for Windows or macOS.

### Developer tooling

`.mcp.json` (repo root) declares only the platform's own `tcip` MCP server; that is what an
agent driving TCIP needs. Some maintainers additionally run a semantic code-search server
(claude-context, backed by an Ollama embedding model and a Milvus instance) as their own
development tooling, configured outside this tracked file and per machine. It is not part of
the platform, not required to run or contribute to it, and its presence or absence changes
nothing about what the platform does.

## Running

```bash
# Web backend (serves pre-built React app at http://127.0.0.1:8765)
conda activate tcip-agent
python -m tcip_web

# Frontend dev server (proxies /api and /ws to backend)
cd packages/tcip-web/frontend
npm run dev        # http://127.0.0.1:5173
npm run build      # rebuild production bundle → ../static/

# Tests
pytest tests/ -n 4 --tb=short --timeout=300 -q
mypy               # type gate; roots come from mypy.ini's files list, run from the repo root
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
# CI-parity gate: runs the steps .github/workflows/ci.yml declares (see CLAUDE.md's Commands block)
python tools/gate_baseline.py --out <dir>

# End-to-end smokes (tools/smoke_*_e2e.py)
python tools/smoke_phenology_e2e.py   # phenology pipeline: mapping -> milestones (offline)
python tools/smoke_terminal_e2e.py    # in-app agent terminal (costs one model turn)
python tools/smoke_fence_e2e.py       # agent permission fence (costs one model turn)
```

The MCP server starts automatically when an MCP client connects (see `.mcp.json`).

## From images to a first delivered number

The sample hazelnut dataset under `data/` is gitignored and not shipped with the repository; a
stranger starts from their own imagery. What follows is the path an agent walks with the
platform's own tools, in the order a first run needs them.

| Tool | Purpose |
|------|---------|
| `initialize_project(project_path, site)` | Scaffolds `.tcip/` under the project directory and records the breeder-stated `site` (the orchard or station the plants stand in, asked of the breeder). `site` is a required argument; there is no default. |
| `ingest_images(source, name, site)` | Copies a raw folder of photos into the canonical `images/<YYYY-MM-DD>/` layout under a workspace project. |
| `register_dataset(dataset_root, crop)` | Records the dataset's identity (crop, id, content fingerprint) so a later delivered number can be traced back to the exact data behind it. |
| `write_class_map(dataset_root, subjects)` | Authors the dataset's class registry: the subjects (object classes to isolate) and their attributes, the expert's own vocabulary, never inferred from labels. |

Annotation itself happens in the GUI's Annotate tab: a human labels a sample of images, and an
image with nothing to label is marked done as a negative there, never inferred from an empty
label file alone.

For a first training run, `draw_splits(folder_path, subject=subject, train_ratio=0.7,
val_ratio=0.15, calibration_ratio=0.15, output_path=<path>)` draws a fresh leakage-free
train/val/calibration split over the labeled data and writes its manifest, binding a later run
that reads it to the same partition; `output_path` (or `materialize=True`) and `subject` are both
required to write one, and all three ratios must be non-zero, since a manifest always draws all
three sides. A call with only `folder_path` answers with split stats over the whole tree instead,
no manifest written and no subject needed. `freeze_split_manifest` is for afterward, binding a
later run to a partition an earlier run already drew, not for drawing the first one.

| Tool | Purpose |
|------|---------|
| `create_experiment(experiment_id, config)` | Registers a new experiment to track a training run's config, metrics, artifacts, and lineage before it starts. |
| `launch_training(config)` | Launches training in an isolated subprocess from an agent-written `model_source` builder; the Training tab's config picker drives the same launch from the GUI side. |
| `evaluate_model(run_id_or_ckpt, images_dir)` | Evaluates a trained checkpoint on a held-out dataset and writes `test_results.json`. |

Before any number can ship, its confidence operating point needs validating against held-out
ground truth: `tcip calibrate-operating-point` runs one model pass over a disjoint
calibration/holdout split, derives a count-unbiased detection
operating point, and checks its held-out count bias (a trait whose delivery reads a classified
positive state, such as a phenology milestone, instead calibrates through the
`calibrate_classifier_operating_point` MCP tool; see the `evaluation` and `phenology` skills).

| Tool | Purpose |
|------|---------|
| `author_trait_spec(project_root, trait, delivers, rationale)` | Registers a trait that does not yet exist, recording the agent's account of why, in the breeder's own terms; the breeder confirms it from the web GUI before it can back a delivery. |
| `state_trait_operationalization(project_root, trait, delivery_kind, statement, mechanism, measured_subject, delivered_phenotypes)` | Records what the trait's delivered number means, in the breeder's own terms, for one delivery kind. Writing this does not itself clear the delivery gate; the breeder confirms it in the Results tab, and only that confirmation lets a delivery door proceed. |
| `deliver_per_image_counts` | Delivers a per-image `image, detection_count, avg_confidence` CSV, gated on the confirmed operationalization and the validated operating point. |
| `run_inference` | Runs a checkpoint and persists a prediction bucket other doors (including a per-plant CSV built from it) treat as ground truth. |

Read the `delivery` skill before choosing between `deliver_per_image_counts` and `run_inference`
(and the per-plant aggregation tools built on top of a prediction bucket): they answer different
questions and carry different CSV schemas, and no MCP tool ships a bare unvalidated phenotype;
only a Results tab delivering route can, through the breeder's own acknowledged act.

## Conventions

- Annotations: per-image COCO-shaped JSON (with `created_by`/`accepted_by` provenance), plus the dataset-level COCO assembled from it for training.
- Experiments: one record per run holding config, metrics, artifacts and lineage, with the run's own files (weights, TensorBoard events, the source snapshot) under `.tcip/experiments/<id>/` beside it.
- Audit log: all MCP tool calls logged via the `@audited` decorator, into the append-only log under the root each call's scope names, and written out at `.tcip/audit.jsonl` by the file backend and by `export_store.py`. The append runs after the tool body; an entry the decorator cannot append raises.
- Lazy imports: within the MCP server's import closure, heavy deps (torch, torchvision) are imported inside function bodies; other modules under `packages/*/src` (the training and inference pipelines, model components) import them at module level.
- Crop traits: controlled vocabulary defined in `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`.
- Measurement-integrity gates: every parameter a delivered phenotype depends on (confidence
  threshold, tile geometry, mask-binarize threshold, physical pixel-to-real-world scale) carries
  its own validation state and a record of what cleared it, never a bare number. A
  delivery door refuses to write a bare unvalidated result; only that result's own delivering
  route in the Results tab can ship a flagged unvalidated one, through the breeder's own
  acknowledged act, never a caller of any door in general. The same shared gate
  (`check_delivery_gate` in `pipelines/resolution.py`) backs every delivery path.

## Roadmap

The pitch above describes the long-term target; what's built today is a narrower slice.

Working now:

- 2D-image detection, instance/semantic segmentation, and classification, end to end, via an
  agent-written `nn.Module` that imports the plain building blocks (necks, heads, losses,
  backbone wrappers, and `build_detector`, one of whose four builders is mask-capable for
  `instance_seg`), on RGB and N-channel imagery (multi-band GeoTIFF/NPZ/grayscale; `num_channels`
  threads to the backbone's `in_chans`, and an `in_chans != 3` detector takes per-band
  `image_mean`/`image_std` from `derivations.band_normalization_stats`).
- Training that loads the native per-image JSON labels directly, experiment tracking,
  annotation/review, SAM-assisted labeling, calibration of a trait's positive-class operating
  point (`calibrate_classifier_operating_point`), and per-plant CSV export, including a
  percentile-crossing phenology-milestone deliverable (per-plant `<trait>_05/50/95per_date` = the
  dates a plant's classified positive-state fraction of detected objects crosses 5/50/95%; the
  positive state is a validated per-object classifier call, never a geometric proxy). Phase 1's
  own shipped example is hazelnut catkin bloom phenology (`catkin_05/50/95per_date`, elongation
  as the positive state).
- Ordinal and regression, trainable and evaluable through the same model/training machinery and
  their own heads, losses, and metrics, calibrating through their own door
  (`calibrate_scalar_operating_point`). Neither has an annotation/review surface built for it:
  both read labels from a hand-authored external CSV of image stem plus rank or value, and both
  are excluded from the platform's automatic train/val split.
- The agent composes the working slice end to end via `build_plant_mapping` → tiled inference →
  `deliver_phenology_milestones`, and the same milestone code backs the Results tab.
- Tiled detection/instance_seg inference over a single georeferenced orthomosaic too large to
  load into memory, instead of the one-photo-per-plant path above. `OrthomosaicGeoreference`
  reads a GeoTIFF's own tags to turn a pixel into a real-world coordinate, refusing on a rotated
  raster or one whose CRS it can't determine. `raster_source.GdalSource` serves windowed reads
  through GDAL's budgeted block cache (overview-aware when the raster carries an `.ovr` pyramid)
  so the raster is never decoded whole. `GenericPredictor.predict_tiled`'s windowed-reader source
  kind runs the same tiled-inference core the per-photo path uses, including `instance_seg` masks
  kept as small tile-local patches with a full-raster offset rather than one full-raster-sized
  array per detection.
- Each detection resolves to a real-world coordinate and is matched to the nearest plant in a
  plant-locations CSV (`assign_detections_to_plants`, which records each detection's `source`
  and `distance_m` and no confidence value); an unmatched detection stays unmatched. Two MCP
  tools compose the whole path end to end the same way `build_plant_mapping` →
  `deliver_phenology_milestones` do for the per-photo case:
  `run_inference`'s `raster_path` regime (tile, persist a prediction bucket) and
  `deliver_orthomosaic_plant_counts` (map detections to plants, aggregate, deliver through the
  same measurement-integrity gate every other per-plant CSV goes through).

Not yet built for the orthomosaic path: a composed pipeline for a dimensional (not count) trait
measured this way, and an automated smoke test against a real multi-gigabyte file; the automated
suite uses a synthetic fixture.

Trained ML models are the deliverable; classical image analysis (OpenCV, scikit-image) is
available for the agent to compose as a situational bootstrapping assist, cheaply producing soft
labels or seeding training data when it fits.

The detection training pipeline mirrors a production drone-phenotyping workflow:

- Metrics & selection: real per-task validation metrics (detection/instance-seg
  mAP via `pycocotools` `COCOeval`; accuracy/F1; MAE/rank-acc) and a composite
  best-model objective (blends loss, F1, mAP50) instead of raw `val_loss`.
- Progressive unfreezing: multi-stage training with optimizer-momentum handoff
  between stages, optional inter-stage LR warmup, and effective-batch LR scaling.
- Small objects: opt-in SAHI-style sliding-window tiling at train and inference
  time (core-region reconstruction + global NMS), plus an FCOS/RetinaNet anchor-free
  detector option and an extra high-resolution (P2) pyramid level.
- Splits: group-aware, annotation-stratified train/val/calibration splitting
  (no source-image leakage) with automatic validation loaders; the calibration side is
  held out from both training and checkpoint selection.
- Imbalance & augmentation: class-weighted / focal losses and a nadir-imagery
  augmentation preset (free rotation + flips; mosaic/copy-paste off).
- HPO: Ray Tune search over the composite, with a pluggable searcher/scheduler
  (ASHA-style pruning and known-good warm start available, not mandatory).
- Reproducibility: global seeding, checkpoint resume (model + optimizer +
  scheduler), and pydantic + neck/head channel-compatibility config validation.
- Review → retrain: turn human review verdicts into a curated training set
  (accepted/edited → labels, rejected → hard negatives) with experiment lineage, and
  prioritize the next review batch by active-learning score.

Not built yet (contributions/experiments welcome):
- 3D point clouds (LiDAR / SfM). There is no point-cloud dataset/loader or task type,
  so this is new work rather than a config flag. (Multispectral / hyperspectral / depth
  as additional 2D channels is now supported via the N-channel path above.)
- Temporal / relational pipeline patterns in general. The one temporal trait built today is the
  percentile-crossing phenology milestone described above (see "Working now"; Phase 1's shipped
  example is hazelnut catkin bloom, per-plant elongated-fraction 05/50/95-per-date milestones);
  broader phenology-sequence and relational patterns beyond the per-image case remain future work.
- Fully automated active learning loop without human-in-the-loop.
- Plant-tag identity (a QR or barcode physically tied to the plant) for capture with no
  georeferencing. Today per-plant identity rests on geolocated capture (`build_plant_mapping`,
  GPS EXIF plus a plant-locations CSV) or a georeferenced orthomosaic; an ungeoreferenced
  dataset has no per-plant path today.
- Provider/LLM-agnostic support. The platform is built against Claude specifically today (the MCP
  server plus Claude Code as the driving agent); supporting other providers/agents (Gemini, Codex,
  open models) alongside it is future work, not a config flag.
- Cloud storage for centralized data. Project state and imagery live in a local `.tcip/`
  directory and local project folders today; centralized or cloud-backed storage for multi-machine
  or multi-user access is future work.
- Web deployment. The GUI binds to loopback (`127.0.0.1`) by default and is built as a
  single-operator local desktop tool (see `packages/tcip-web/README.md`'s trust-boundary note);
  deploying it as a hosted, multi-user web service (including the token auth already noted there
  as a planned follow-on) is future work.

## Security and data egress

Local project state and imagery answer where data rests, not what leaves the machine when an agent
drives the platform. When the agent works, the breeding data it reads is sent to your model provider,
the same as pasting it into a chat, and the platform does not bound or redact it. Read
[SECURITY.md](SECURITY.md) before using TCIP on commercially sensitive breeding data: it inventories
every channel that leaves the machine, the one phone-home you can disable, and the trust boundary the
loopback bind rests on.

Contributing a change: [CONTRIBUTING.md](CONTRIBUTING.md) has the gates it must pass, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) the standards the project holds contributors to. What an
adopter can build against today, independent of the 0.x version number, is
[STABILITY.md](STABILITY.md)'s; [VERSIONING.md](VERSIONING.md) explains what that version number
does and doesn't promise. Adopter-visible changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

TCIP Agent is released under the [Apache License 2.0](LICENSE) (© 2026 Zack Loken).
Commercial use is permitted. Bundled third-party components (e.g. timm, SAM2, also
under Apache-2.0) are attributed in [NOTICE](NOTICE).