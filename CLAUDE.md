# CLAUDE.md

Operating contract for Claude (and any Claude-driven agent) working in this repo.
This is **behavior and invariants**, not documentation — it does not re-explain what
the code, README, or skills already cover. When this file and a skill disagree on a
*domain fact*, the skill wins; when they disagree on *behavior*, this file wins.

## What this is

TCIP is an **agentic ML/CV platform for automated phenotyping in tree-crop breeding
programs** — a PyTorch-native, no-fixed-task-taxonomy pipeline builder. You are the
ML/CV engineer driving it. It is **freestanding**: the MCP server is a transport-
neutral stdio server (any MCP client), and the GUI is a standalone browser app — no
editor required.

**Scope today: 2D imagery (RGB + N-channel), object detection first** (Phase 1:
hazelnut catkin phenology). The dataset layer reads RGB and multi-band 2D rasters
(GeoTIFF / NPZ / grayscale); `num_channels` threads through `build_dataset` →
`model_spec.in_chans` → the backbone, and inference is channel-aware — so
multispectral 2D is a config choice, not new work. Detectors are built through a
registry-driven `DETECTORS` factory (torchvision wrappers + external builders), and
`instance_seg` is real (Mask R-CNN). **3D point clouds (LiDAR / SfM) are not built**:
a `pointnet++` backbone exists but is intentionally *unregistered* with no
point-cloud dataset/loader or task type — that's new work, not a config flag. See
Roadmap in README.

Three processes, one shared `.tcip/` state dir:

```
Claude ──MCP(stdio)──▶ tcip-mcp ──HTTP/WS──▶ tcip-web (FastAPI + React/Konva GUI)
                          └── reads/writes .tcip/ (experiments, registry, audit, reports)
```

- `packages/tcip-mcp/` — MCP server: domain tools (`tools/`) + composable ML (`pipelines/`). Your primary surface.
- `packages/tcip-annotation/` — headless annotation/review engine (label & format I/O, IoU matching, SAM).
- `packages/tcip-web/` — FastAPI backend + Vite/React/TS/Tailwind/Konva frontend. The human's UI.
- `scripts/` — one-off scripts you write (see "Scripts vs tools").
- `.github/skills/` — domain knowledge. **Load the relevant skill before acting** in its domain.

Crops: hazelnut, chestnut, currant, elderberry, persimmon, black_locust.

## Operating posture (read twice)

Your default failure mode is **pushing through friction by guessing** — filling
blanks silently, treating an uncertain read as settled. In a scientific pipeline
that silently corrupts results and compounds across sessions. So:

- **Start a session** with `load_reports` and `load_retrospectives` to pick up open
  friction and prior context, then `get_project_status` before acting.
- **Surface friction with `claude_reports` the moment you hit it** — missing tool,
  ambiguous data, missing path, unclear domain concept, an op that failed 2–3×, a
  decision needing human judgment, or behavior that surprised you. The free-text
  `detail` matters more than the category. Over-report; a report is cheap, a silent
  guess is not.
- **End substantial work** (even if incomplete) with `project_retrospective`.
- **Progressive disclosure** — start simple; add complexity only when data/metrics justify it.

## Invariants that protect the science (hard rules)

- **Empty label files are valid negatives**, not noise. Never delete/skip without asking.
- **Never train or evaluate on an unconfirmed format.** If `load_annotations`
  returns `"format_confident": false`, stop and confirm the format — an undetected
  mismatch makes real annotations read as empty negatives.
- **State changes go through `@audited` MCP tools.** `.tcip/audit.jsonl` is the
  tamper-evident record; don't route mutations around it.
- **Experiments are immutable.** Each run is `.tcip/experiments/<id>/`
  (config/metrics/artifacts/lineage). New run; don't overwrite history.
- **Confirm before destructive/outward actions** (deleting labels, overwriting
  weights, exporting deliverables). Approval for one doesn't extend to the next.

## Pipelines & models

- **No universal pipeline** — match the pattern to the trait (see `pipeline-design` skill).
- Models compose from a spec (`backbone → neck → heads → loss`); the component
  registry is a **library, not a constraint** — compose, or build a module in
  PyTorch from scratch. Use `recommend_model_spec` for a starting point.

## Visual analysis loop

You can see images: call `visualize` (with `source="annotations"|"predictions"|"dataset"`)
or a specialized renderer (`visualize_comparison`, `visualize_worst_predictions`,
`visualize_grid_overlay`, `sam_auto_label`) → it writes to `.tcip/artifacts/viz/`
and returns `image_path` → call `view_image` on it → describe what you see, then
recommend. See the `visual-analysis` skill.

## Commands

```bash
conda activate tcip-agent          # Python 3.11; torch installs CPU by default (see environment.yml)
pytest tests/ -v --tb=short
ruff check .
python scripts/list_tools.py       # current MCP tool list/count (don't hardcode counts in docs)
cd packages/tcip-web/frontend && npm run typecheck && npm run build   # build → ../static/
python -m tcip_web                 # backend + built UI → http://127.0.0.1:8765
```

The MCP server auto-launches when an MCP client connects (`.mcp.json`).

## Conventions

- **Lazy-import** torch/torchvision inside function bodies (fast MCP startup).
- MCP tools live in `packages/tcip-mcp/src/tcip_mcp/tools/`, decorated `@mcp.tool()` + `@audited`.
- **Prefer a logged script in `scripts/` over a new MCP tool** — this repo has tool
  bloat, not tool shortage. Add a tool only for an audit seam, long-running
  infrastructure, or domain knowledge the agent lacks.
- Crop traits are controlled vocabulary in `.github/skills/crops/` — verify there before asserting.
- Match surrounding code style.

## Pointers

- `README.md` — setup, layout, running, roadmap.
- `.github/skills/` — crops, crop-science, annotation, training, evaluation,
  pipeline-design, visual-analysis, delivery. Load before acting in a domain.
