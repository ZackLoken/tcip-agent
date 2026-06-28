# TCIP Agent

Prototype (in progress) of an agentic ML/CV system for automated phenotyping in tree crop breeding programs. A Claude agent (ML/CV engineer persona) drives annotation, model training, inference, and per-plant result delivery through an MCP tool server, while a browser-based GUI supports human annotation, review, and training oversight. The system is freestanding — the MCP server is a transport-neutral stdio server (any MCP client) and the GUI is a standalone browser app; no editor required.

**Current scope: RGB imagery, object detection first.** The data layer loads RGB images only; multispectral / depth / 3D point-cloud support is not built yet (see [Roadmap](#roadmap)).

Six crops in scope: hazelnut, chestnut, currant, elderberry, persimmon, black locust. Phase 1 target is hazelnut catkin phenology from ground imagery.

Active development: consolidating the MCP tool surface and building out the Annotate and Review tabs of the GUI. 

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
│  FastAPI + React GUI         │  Annotate / Review / Training /
│  (packages/tcip-web)         │  Tuning / Inference / Results tabs
└──────────────────────────────┘
```

All three processes share `.tcip/` on disk (experiment state, model registry, audit log, GUI state).

Supporting library: `packages/tcip-annotation` — headless annotation engine (label I/O, format conversion, IoU matching, SAM wrapper). No dependency on the other packages.

## Repository layout

```
CLAUDE.md                      # agent operating contract (persona, invariants, conventions)
.github/
  skills/                      # domain knowledge modules (crops, annotation, training, ...)
  prompts/                     # slash-command templates
packages/
  tcip-mcp/                    # MCP server (python -m tcip_mcp)
    src/tcip_mcp/
      tools/                   # domain tools (run scripts/list_tools.py for the current list)
      pipelines/               # composable ML: registry, composer, trainer, predictor
  tcip-annotation/             # headless annotation library
  tcip-web/                    # FastAPI backend + React frontend
    src/tcip_web/
      routes/                  # annotate, review, training, tuning, inference, results, ...
    frontend/src/              # Vite + React 18 + TypeScript + Tailwind + Konva
scripts/                       # one-off ingestion scripts (created by agent)
tests/                         # pytest suite
data/                          # sample hazelnut dataset (gitignored)
```

## Setup

```bash
# Python — creates the env and installs the three packages (editable). Run from
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
npm run typecheck  # from packages/tcip-web/frontend
```

The MCP server starts automatically when an MCP client connects (see `.mcp.json`).

## Conventions

- **Annotations**: YOLO by default. Auto-detects YOLO / COCO / PASCAL VOC / LabelMe. Empty label files are valid negatives.
- **Experiments**: tracked in `.tcip/experiments/<id>/` with config, metrics JSONL, artifacts, lineage.
- **Audit log**: all MCP tool calls logged to `.tcip/audit.jsonl` via `@audited` decorator.
- **Lazy imports**: heavy deps (torch, torchvision) imported inside function bodies for fast MCP startup.
- **Crop traits**: controlled vocabulary defined in `.github/skills/crops/`.

## Roadmap

The pitch above describes the long-term target. What's actually built today is a
narrower slice; this section keeps the two honest.

**Working now:** RGB-image tasks end to end — detection, instance/semantic
segmentation, classification, ordinal, regression — via the composable
backbone/neck/head spec, with experiment tracking, annotation/review, SAM-assisted
labeling, and per-plant CSV export.

The detection training pipeline mirrors a production drone-phenotyping workflow:

- **Metrics & selection** — real per-task validation metrics (detection/instance-seg
  mAP via `pycocotools` `COCOeval`; accuracy/F1; MAE/rank-acc) and a composite
  best-model objective (blends loss, F1, mAP50) instead of raw `val_loss`.
- **Progressive unfreezing** — multi-stage training with optimizer-momentum handoff
  between stages, optional inter-stage LR warmup, and effective-batch LR scaling.
- **Small objects** — opt-in SAHI-style sliding-window tiling at train and inference
  time (core-region reconstruction + global NMS), plus an FCOS/RetinaNet anchor-free
  detector option and an extra high-resolution (P2) pyramid level.
- **Honest splits** — group-aware, annotation-stratified train/val/test splitting
  (no source-image leakage) with automatic validation loaders.
- **Imbalance & augmentation** — class-weighted / focal losses and a nadir-imagery
  augmentation preset (free rotation + flips; mosaic/copy-paste intentionally off).
- **HPO** — Optuna search with ASHA pruning + known-good warm start on the composite.
- **Reproducibility** — global seeding, checkpoint resume (model + optimizer +
  scheduler), and pydantic + neck/head channel-compatibility config validation.
- **Review → retrain** — turn human review verdicts into a curated training set
  (accepted/edited → labels, rejected → hard negatives) with experiment lineage, and
  prioritize the next review batch by active-learning score.

**Not built yet (contributions/experiments welcome):**
- Non-RGB data paths: multispectral / hyperspectral, depth, and 3D point clouds.
  The dataset layer currently loads RGB images only. A `pointnet++` backbone is
  registered as an **experimental component** but has no point-cloud dataset/loader,
  so it is not trainable through the normal pipeline yet.
- N-channel input for 2D backbones (the timm builder does not yet thread `in_chans`).
- Temporal / phenology-sequence and relational pipeline patterns beyond the
  per-image case.

## License

TCIP Agent is released under the [PolyForm Noncommercial License 1.0.0](LICENSE)
(© Zack Loken). Bundled third-party components (e.g. timm, SAM2 under Apache-2.0)
are attributed in [NOTICE](NOTICE).