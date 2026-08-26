# TCIP Agent

Prototype (in progress) of an agentic ML/CV system for automated phenotyping in tree crop breeding programs. A Claude agent (ML/CV engineer persona) drives annotation, model training, inference, and per-plant result delivery through an MCP tool server, while a browser-based GUI supports human annotation, review, and training oversight. The system is freestanding: the MCP server is a transport-neutral stdio server (any MCP client) and the GUI is a standalone browser app; no editor required.

Who it's for: plant breeders, not CV engineers. They know their crops and traits; they don't write PyTorch or make CV decisions. The objective is to replace the CV scientist in the loop: an agent that guides a breeder end to end, from imagery plus a trait to a validated per-plant phenotype, maintaining scientific rigor on the breeder's behalf. There are two distinct users and two UX surfaces: the agent's UX is the tools/skills/code/docs/MCP API surface it reasons through (see `CLAUDE.md`), and the breeder's UX is the browser GUI, the only surface they experience the platform through (label a few examples, confirm/correct the model, receive the result).

Current scope: 2D imagery (RGB + N-channel) from any capture modality, object detection first. Two genuinely different capture shapes are both supported, not just "any modality" as a vague umbrella: an ordinary per-plant image (one photo of one or a few plants, from a phone, DSLR, GoPro, or ground rig) and a single large georeferenced orthomosaic covering many plants in one file (a drone survey mosaic, potentially tens of gigabytes, read via windowed/tiled access rather than loaded whole, with each detection resolved to a real-world coordinate and matched to the plant it belongs to). The data layer reads RGB and multi-band rasters (GeoTIFF / NPZ / grayscale) for both shapes; 3D point-cloud (LiDAR / SfM) support is not built yet (see [Roadmap](#roadmap)).

Six crops in scope: hazelnut, chestnut, currant, elderberry, persimmon, black locust. Phase 1 target is hazelnut catkin phenology.

Status: the browser GUI is built out across all tabs (Annotate / Review / Training / Tuning / Inference / Results / Meta), the MCP tool surface is in place (run `python scripts/list_tools.py` for the current count), and every delivered phenotype (a count, a dimensional measurement, a phenology milestone) is gated on validated measurement inputs end to end: an uncalibrated confidence threshold, an unproven physical scale, or a fabricated tile geometry refuses delivery rather than shipping a confident, unvalidated number. Phase 1 focus remains hazelnut catkin phenology; the orthomosaic capability described above is built and verified against a real 90+ GB drone survey file, not yet exercised end to end against a first live delivery.

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
           │ HTTP / WebSocket
           ▼
┌──────────────────────────────┐
│  FastAPI + React GUI         │  Annotate / Review / Training / Tuning /
│  (packages/tcip-web)         │  Inference / Results / Meta tabs
└──────────────────────────────┘
```

All three processes share `.tcip/` on disk (experiment state, model registry, audit log, GUI state).

Supporting libraries: `packages/tcip-annotation` (headless annotation engine: label I/O, IoU matching, SAM wrapper) and `packages/tcip-store` (the storage seam: one locked, atomic interface for the platform's records, append-only logs and blobs). `tcip-store` is the bottom of the stack, depending on nothing else here; `tcip-annotation` depends on it and on neither of the other two.

Records and append-only logs go into one SQLite database per root, `<root>/.tcip/store.db`; blob bytes stay files under every backend, so imagery, labels and predictions travel with the dataset as they always did. Set `TCIP_STORE_BACKEND=file` to bind the file backend instead, which reads and writes the same records as loose files. A root whose records are still loose files is refused rather than read as empty: `python scripts/adopt_store.py <root>` moves them into a database, and `python scripts/export_store.py <root>` writes them back out.

## Repository layout

```
CLAUDE.md                      # agent operating contract (persona, invariants, conventions)
.github/
  skills/                      # domain knowledge modules (crops, annotation, training, ...)
packages/
  tcip-mcp/                    # MCP server (python -m tcip_mcp)
    src/tcip_mcp/
      tools/                   # domain tools (run scripts/list_tools.py for the current list)
      pipelines/               # ML: model_build, trainer, predictor, envelope + plain
                               #   nn.Module blocks; postprocessing (per-image plant mapping +
                               #   phenology, plus orthomosaic georeferencing/windowed inference)
  tcip-annotation/             # headless annotation library
  tcip-store/                  # storage seam: keyed records, append-only logs, blobs
    src/tcip_store/            # the contract (model, errors, registry, store) + file backend
  tcip-web/                    # FastAPI backend + React frontend
    src/tcip_web/
      routes/                  # annotate, review, training, tuning, inference, results, ...
    frontend/src/              # Vite + React 18 + TypeScript + Tailwind + Konva
scripts/                       # agent one-off scripts + end-to-end smokes (smoke_*_e2e.py); see scripts/README.md
tests/                         # pytest suite
data/                          # sample hazelnut dataset (gitignored)
```

## Setup

```bash
# Python: creates the env and installs the three packages (editable). Run from
# the repo root. Installs a CPU/-or-platform torch wheel; see environment.yml for
# the CUDA option.
conda env create -f environment.yml
conda activate tcip-agent

# Frontend
cd packages/tcip-web/frontend
npm install
```

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
python scripts/gate_baseline.py --out <dir>

# End-to-end smokes (scripts/smoke_*_e2e.py)
python scripts/smoke_phenology_e2e.py   # phenology pipeline: mapping -> milestones (offline)
python scripts/smoke_terminal_e2e.py    # in-app agent terminal (costs one model turn)
python scripts/smoke_fence_e2e.py       # agent permission fence (costs one model turn)
```

The MCP server starts automatically when an MCP client connects (see `.mcp.json`).

## Conventions

- Annotations: per-image COCO-shaped JSON (with `created_by`/`accepted_by` provenance), plus the dataset-level COCO assembled from it for training.
- Experiments: one record per run holding config, metrics, artifacts and lineage, with the run's own files (weights, TensorBoard events, the source snapshot) under `.tcip/experiments/<id>/` beside it.
- Audit log: all MCP tool calls logged via the `@audited` decorator, into the append-only log under the root each call's scope names, and written out at `.tcip/audit.jsonl` by the file backend and by `export_store.py`. An entry the decorator cannot append raises, since the append runs after the tool body.
- Lazy imports: heavy deps (torch, torchvision) imported inside function bodies for fast MCP startup.
- Crop traits: controlled vocabulary defined in `.github/skills/crops/`.
- Measurement-integrity gates: every parameter a delivered phenotype depends on (confidence
  threshold, tile geometry, mask-binarize threshold, physical pixel-to-real-world scale) carries
  its own validation state and a record of what actually cleared it, never a bare number. A
  delivery door refuses to write a bare unvalidated result; it either ships a validated one or a
  caller must explicitly acknowledge shipping a flagged provisional one. The same shared gate
  (`check_delivery_gate` in `pipelines/resolution.py`) backs every delivery path so none of them
  can drift into disagreeing about when a number is trustworthy.

## Roadmap

The pitch above describes the long-term target. What's actually built today is a
narrower slice; this section keeps the two honest.

Working now: 2D-image detection, instance/semantic segmentation, and classification, end
to end, via an agent-written `nn.Module` that imports the plain building blocks (necks, heads,
losses, backbone wrappers, and `build_detector`; `instance_seg` via Mask R-CNN), on RGB and
N-channel imagery (multi-band GeoTIFF/NPZ/grayscale; `num_channels` threads to the backbone's
`in_chans`, and an `in_chans != 3` detector takes per-band `image_mean`/`image_std` from
`derivations.band_normalization_stats`), with training that loads the native per-image JSON
labels directly, experiment tracking, annotation/review, SAM-assisted labeling, calibration of
a trait's positive-class operating point (`calibrate_classifier_operating_point`), and per-plant
CSV export, including a percentile-crossing phenology-milestone deliverable (per-plant
`<trait>_05/50/95per_date` = the dates a plant's classified positive-state fraction of detected
objects crosses 5/50/95%; the positive state is a validated per-object classifier call, never a
geometric proxy). Ordinal and regression are also trainable and evaluable, through the same
model/training machinery and their own heads, losses, and metrics, and calibrate through their
own door (`calibrate_ordinal_regression_operating_point`), but neither has an annotation/review
surface built for it: both read labels from a hand-authored external CSV of image stem plus rank
or value rather than the platform's own annotation/review UI, and both are excluded from the
platform's automatic train/val split. The agent composes the
working slice end to end via `build_plant_mapping` → tiled inference → `compute_phenology`, and
the same milestone code backs the Results tab, so a milestone date means one thing on both
surfaces. Phase 1's own shipped example is hazelnut catkin bloom phenology
(`catkin_05/50/95per_date`, elongation as the positive state).

Also working now: tiled detection/instance_seg inference over a single georeferenced orthomosaic
too large to load into memory (confirmed against a real 90.7 GB, 141130x239921px, 4-band drone
survey file), instead of the one-photo-per-plant path above. `OrthomosaicGeoreference` reads a
GeoTIFF's own tags to turn a pixel into a real-world coordinate (refusing cleanly, never guessing,
on a rotated raster or one whose CRS it can't determine), and `raster_source.GdalSource` serves
windowed reads through GDAL's budgeted block cache (overview-aware when the raster carries an
`.ovr` pyramid) so the raster is never decoded whole. `GenericPredictor.predict_tiled`'s windowed-
reader source kind runs the same tiled-inference core the per-photo path uses, including
`instance_seg` masks (kept as small tile-local patches with a full-raster offset, not one
full-raster-sized array per detection). Each detection resolves to a real-world coordinate and is
matched to the nearest plant in a plant-locations CSV (`assign_detections_to_plants`, honest
`source`/`distance_m`, no fabricated confidence, an unmatched detection stays unmatched rather than
being forced onto the nearest plant regardless of distance). Two MCP tools compose the whole path
end to end the same way `build_plant_mapping` → `compute_phenology` do for the per-photo case:
`export_predictions`'s `raster_path` regime (tile, persist a prediction bucket) and
`deliver_orthomosaic_plant_counts` (map detections to plants, aggregate, deliver through the same
measurement-integrity gate every other per-plant CSV goes through). Not yet built: a composed
pipeline for a dimensional (not count) trait measured this way, and an automated smoke test against
a real multi-gigabyte file (verified manually against the file above; the automated suite uses a
synthetic fixture, since a real 90+ GB file can't live in CI).

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
- Honest splits: group-aware, annotation-stratified train/val/test splitting
  (no source-image leakage) with automatic validation loaders.
- Imbalance & augmentation: class-weighted / focal losses and a nadir-imagery
  augmentation preset (free rotation + flips; mosaic/copy-paste intentionally off).
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

## License

TCIP Agent is released under the [Apache License 2.0](LICENSE) (© 2026 Zack Loken).
Commercial use is permitted. Bundled third-party components (e.g. timm, SAM2, also
under Apache-2.0) are attributed in [NOTICE](NOTICE).