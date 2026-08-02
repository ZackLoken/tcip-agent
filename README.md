# TCIP Agent

Prototype (in progress) of an agentic ML/CV system for automated phenotyping in tree crop breeding programs. A Claude agent (ML/CV engineer persona) drives annotation, model training, inference, and per-plant result delivery through an MCP tool server, while a browser-based GUI supports human annotation, review, and training oversight. The system is freestanding: the MCP server is a transport-neutral stdio server (any MCP client) and the GUI is a standalone browser app; no editor required.

**Who it's for:** plant breeders, not CV engineers. They know their crops and traits; they don't write PyTorch or make CV decisions. The objective is to replace the CV scientist in the loop, an agent that guides a breeder end-to-end, from imagery plus a trait to a validated per-plant phenotype, maintaining scientific rigor on the breeder's behalf. There are two distinct users and two UX surfaces: the agent's UX is the tools/skills/code/docs/MCP API surface it reasons through (see `CLAUDE.md`), and the breeder's UX is the browser GUI, the only surface they experience the platform through (label a few examples, confirm/correct the model, receive the result).

**Current scope: 2D imagery (RGB + N-channel) from any capture modality (aerial/drone, ground-based, rover-mounted, lab/benchtop), object detection first.** The data layer reads RGB and multi-band 2D rasters (GeoTIFF / NPZ / grayscale); 3D point-cloud (LiDAR / SfM) support is not built yet (see [Roadmap](#roadmap)).

Six crops in scope: hazelnut, chestnut, currant, elderberry, persimmon, black locust. Phase 1 target is hazelnut catkin phenology from ground imagery.

Status: the browser GUI is built out across all tabs (Annotate / Review / Training / Tuning / Inference / Results / Meta) and the MCP tool surface is in place. Phase 1 focus is hazelnut catkin phenology on ground imagery.

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

Supporting library: `packages/tcip-annotation` (headless annotation engine: label I/O, IoU matching, SAM wrapper). No dependency on the other packages.

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
                               #   nn.Module blocks; postprocessing (plant mapping + phenology)
  tcip-annotation/             # headless annotation library
  tcip-web/                    # FastAPI backend + React frontend
    src/tcip_web/
      routes/                  # annotate, review, training, tuning, inference, results, ...
    frontend/src/              # Vite + React 18 + TypeScript + Tailwind + Konva
scripts/                       # agent one-off scripts + end-to-end smokes (smoke_*_e2e.py)
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
pytest tests/ -v --tb=short
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build

# End-to-end smokes (scripts/smoke_*_e2e.py)
python scripts/smoke_phenology_e2e.py   # phenology pipeline: mapping -> milestones (offline)
python scripts/smoke_terminal_e2e.py    # in-app agent terminal (costs one model turn)
python scripts/smoke_fence_e2e.py       # agent permission fence (costs one model turn)
```

The MCP server starts automatically when an MCP client connects (see `.mcp.json`).

## Conventions

- **Annotations**: per-image COCO-shaped JSON (with `created_by`/`accepted_by` provenance), plus the dataset-level COCO assembled from it for training. A negative is an empty label set **plus** an explicit human "Complete"; an empty label file alone is not a negative.
- **Experiments**: tracked in `.tcip/experiments/<id>/` with config, metrics JSONL, artifacts, lineage.
- **Audit log**: all MCP tool calls logged to `.tcip/audit.jsonl` via `@audited` decorator.
- **Lazy imports**: heavy deps (torch, torchvision) imported inside function bodies for fast MCP startup.
- **Crop traits**: controlled vocabulary defined in `.github/skills/crops/`.

## Roadmap

The pitch above describes the long-term target. What's actually built today is a
narrower slice; this section keeps the two honest.

**Working now:** 2D-image tasks end to end (detection, instance/semantic
segmentation, classification, ordinal, regression) via an agent-written `nn.Module`
that imports the plain building blocks (necks, heads, losses, backbone wrappers, and
`build_detector`; `instance_seg` via Mask R-CNN), on **RGB and N-channel imagery** (multi-band
GeoTIFF/NPZ/grayscale; `num_channels` threads to the backbone's `in_chans`, and an `in_chans != 3`
detector takes per-band `image_mean`/`image_std` from `derivations.band_normalization_stats`), with
training that loads the native per-image JSON labels directly, experiment tracking,
annotation/review, SAM-assisted labeling, and per-plant CSV export, including a
percentile-crossing phenology-milestone deliverable (per-plant `<trait>_05/50/95per_date` = the
dates a plant's classified positive-state fraction of detected objects crosses 5/50/95%; the
positive state is a validated per-object classifier call, never a geometric proxy). The agent
composes it end to end via `build_plant_mapping` → tiled inference → `compute_phenology`, and the
same milestone code backs the Results tab, so a milestone date means one thing on both surfaces.
Phase 1's own shipped example is hazelnut catkin bloom phenology (`catkin_05/50/95per_date`,
elongation as the positive state).

Trained ML models are the deliverable; classical image analysis (OpenCV, scikit-image) is
available for the agent to compose as a situational bootstrapping assist, cheaply producing soft
labels or seeding training data when it fits, never the end product itself. High-throughput
phenotyping doesn't scale on classical analysis alone, and its output is always reviewed and
validated before it's trusted as ground truth.

The detection training pipeline mirrors a production drone-phenotyping workflow:

- **Metrics & selection**: real per-task validation metrics (detection/instance-seg
  mAP via `pycocotools` `COCOeval`; accuracy/F1; MAE/rank-acc) and a composite
  best-model objective (blends loss, F1, mAP50) instead of raw `val_loss`.
- **Progressive unfreezing**: multi-stage training with optimizer-momentum handoff
  between stages, optional inter-stage LR warmup, and effective-batch LR scaling.
- **Small objects**: opt-in SAHI-style sliding-window tiling at train and inference
  time (core-region reconstruction + global NMS), plus an FCOS/RetinaNet anchor-free
  detector option and an extra high-resolution (P2) pyramid level.
- **Honest splits**: group-aware, annotation-stratified train/val/test splitting
  (no source-image leakage) with automatic validation loaders.
- **Imbalance & augmentation**: class-weighted / focal losses and a nadir-imagery
  augmentation preset (free rotation + flips; mosaic/copy-paste intentionally off).
- **HPO**: Ray Tune search over the composite, with a pluggable searcher/scheduler
  (ASHA-style pruning and known-good warm start available, not mandatory).
- **Reproducibility**: global seeding, checkpoint resume (model + optimizer +
  scheduler), and pydantic + neck/head channel-compatibility config validation.
- **Review → retrain**: turn human review verdicts into a curated training set
  (accepted/edited → labels, rejected → hard negatives) with experiment lineage, and
  prioritize the next review batch by active-learning score.

**Not built yet (contributions/experiments welcome):**
- 3D point clouds (LiDAR / SfM). There is no point-cloud dataset/loader or task type,
  so this is new work rather than a config flag. (Multispectral / hyperspectral / depth
  as additional 2D channels *is* now supported via the N-channel path above.)
- Temporal / relational pipeline patterns in general. The one temporal trait built today is the
  percentile-crossing phenology milestone described above (see "Working now"; Phase 1's shipped
  example is hazelnut catkin bloom, per-plant elongated-fraction 05/50/95-per-date milestones);
  broader phenology-sequence and relational patterns beyond the per-image case remain future work.
- Fully automated active learning loop without human-in-the-loop.
- **Provider/LLM-agnostic support.** The platform is built against Claude specifically today (the MCP
  server plus Claude Code as the driving agent); supporting other providers/agents (Gemini, Codex,
  open models) alongside it is future work, not a config flag.
- **Cloud storage for centralized data.** Project state and imagery live in a local `.tcip/`
  directory and local project folders today; centralized or cloud-backed storage for multi-machine
  or multi-user access is future work.
- **Web deployment.** The GUI binds to loopback (`127.0.0.1`) by default and is built as a
  single-operator local desktop tool (see `packages/tcip-web/README.md`'s trust-boundary note);
  deploying it as a hosted, multi-user web service (including the token auth already noted there
  as a planned follow-on) is future work.

## License

TCIP Agent is released under the [PolyForm Noncommercial License 1.0.0](LICENSE)
(© Zack Loken). Bundled third-party components (e.g. timm, SAM2 under Apache-2.0)
are attributed in [NOTICE](NOTICE).