# TCIP Agent

Prototype (in progress) of an agentic ML/CV system for automated phenotyping in tree crop breeding programs. A Claude Code / GitHub Copilot agent (ML/CV engineer persona) drives annotation, model training, inference, and per-plant result delivery through an MCP tool server, while a browser-based GUI supports human annotation, review, and training oversight.

Six crops in scope: hazelnut, chestnut, currant, elderberry, persimmon, black locust. Phase 1 target is hazelnut catkin phenology from ground imagery.

## Architecture

```
┌──────────────────────────────┐
│  Claude Code / Copilot agent │  ML/CV engineer persona
│  (.github/agents/tcip.agent) │  designs pipelines, trains, evaluates
└──────────┬───────────────────┘
           │ MCP (stdio)
           ▼
┌──────────────────────────────┐
│  Python MCP server           │  54 domain tools: data, annotation,
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
.github/
  agents/tcip.agent.md         # agent persona and workflow
  copilot-instructions.md      # primary agent system prompt
  skills/                      # domain knowledge modules (crops, annotation, training, ...)
  prompts/                     # slash-command templates
packages/
  tcip-mcp/                    # MCP server (python -m tcip_mcp)
    src/tcip_mcp/
      tools/                   # 10 files, 54 tools
      pipelines/               # composable ML: registry, composer, trainer, predictor
  tcip-annotation/             # headless annotation library
  tcip-web/                    # FastAPI backend + React frontend
    src/tcip_web/
      routes/                  # annotate, review, training, tuning, inference, results, ...
    frontend/src/              # Vite + React 18 + TypeScript + Tailwind + Konva
scripts/                       # one-off ingestion and analysis scripts
tests/                         # pytest suite
data/                          # sample hazelnut dataset (gitignored)
```

## Setup

```bash
# Python — create env from lockfile, then install packages in editable mode
conda env create -f environment.yml
conda activate tcip-agent
pip install -e packages/tcip-annotation
pip install -e packages/tcip-mcp
pip install -e packages/tcip-web

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

The MCP server starts automatically when Claude Code connects (see `.mcp.json`).

## Conventions

- **Annotations**: YOLO by default. Auto-detects YOLO / COCO / PASCAL VOC / LabelMe. Empty label files are valid negatives.
- **Experiments**: tracked in `.tcip/experiments/<id>/` with config, metrics JSONL, artifacts, lineage.
- **Audit log**: all MCP tool calls logged to `.tcip/audit.jsonl` via `@audited` decorator.
- **Lazy imports**: heavy deps (torch, torchvision) imported inside function bodies for fast MCP startup.
- **Crop traits**: controlled vocabulary defined in `.github/skills/crops/`.