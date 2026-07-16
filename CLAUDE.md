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

- **Measurement integrity — the highest rule.** This is a computer-vision *and* a
  breeding platform: a confident, precise, wrong phenotype is the worst thing it can
  produce. So:
  - **The domain expert defines each trait's measurement**, grounded in the imagery. You
    (Claude) are the CV engineer who operationalizes *their* definition; never substitute
    your own. If the definition is unclear, stop and ask — do not infer it.
  - **Never invent a proxy for a biological/phenotypic quantity.** A bounding-box height,
    aspect ratio, or any geometric surrogate is *not* a measurement of a morphological
    stage. If a trait can't yet be measured validly from pixels, say so — do not
    manufacture a number so a result appears. (A prior session defined catkin "elongation"
    from bbox height; it was invalid science, shipped fabricated phenology CSVs, and has
    been removed — do not do this.)
  - **Validate the measurement against expert-scored ground truth before producing any
    downstream result** (curve, milestone, CSV, delivery). No validated measurement → no
    result.
  - **Never commit unvalidated domain logic as if it were a definition.** Provisional logic
    is flagged provisional and validated or removed; it must not silently become
    institutional truth that the next session reuses.
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
- **Parameters: derive, don't pin.** When a threshold or operating point varies by
  dataset / model / trait (conf, IoU-for-a-hit, NMS, tile, anchors, `max_dets`), the deliverable
  is never the *value* — not one you pick, not one you derive from the current dataset and freeze
  "for future data" (a constant with extra steps, wrong on the next). Build the agent's capability
  to derive it from the data in hand, at runtime. The human defines a trait's *semantics* (what a
  catkin is; what a milestone means); the agent derives the *operating points* that realize it.

## Visual analysis loop

You can see images: call `visualize` (with `source="annotations"|"predictions"|"dataset"`)
or a specialized renderer (`visualize_comparison`, `visualize_worst_predictions`,
`visualize_grid_overlay`, `visualize_canvas`, `sam_auto_label`) → it writes to
`.tcip/artifacts/viz/` and returns `image_path` → call `view_image` on it → describe what
you see, then recommend. `visualize_canvas` shows the human's live GUI canvas — their
image, viewport, and unsaved shapes with the GUI's own symbology. See the
`visual-analysis` skill.

## Commands

```bash
conda activate tcip-agent          # Python 3.11; torch installs CPU by default (see environment.yml)
pytest tests/ -v --tb=short
ruff check .
python scripts/list_tools.py       # current MCP tool list/count (don't hardcode counts in docs)
cd packages/tcip-web/frontend && npm run typecheck && npm run build   # build → ../static/
python -m tcip_web                 # backend + built UI → http://127.0.0.1:8765
```

The MCP server auto-launches when an MCP client connects (`.mcp.json`). If the
`mcp__tcip__*` tools aren't available, the repo's `.mcp.json` didn't launch it — you're
not at the repo root; relaunch from there. Durable platform state (`.tcip/audit.jsonl`,
`.tcip/experiments/`) resolves via `$TCIP_PROJECT_ROOT` (the server/backend pin it to the
repo root at startup), so a process started from a subdir no longer fragments `.tcip/`.

## Conventions

- **Lazy-import** torch/torchvision inside function bodies (fast MCP startup).
- MCP tools live in `packages/tcip-mcp/src/tcip_mcp/tools/`, decorated `@mcp.tool()` + `@audited`.
- **Prefer a logged script in `scripts/` over a new MCP tool** — this repo has tool
  bloat, not tool shortage. Add a tool only for an audit seam, long-running
  infrastructure, or domain knowledge the agent lacks.
- Crop traits are controlled vocabulary in `.github/skills/crops/` — verify there before asserting.
- Match surrounding code style.
- **Comments and emphasis (hard rule).** Match the file's existing comment density — most edits need
  no new comment. Keep any comment to one line; never add a multi-line block to explain "why" on a
  routine change. Never use all-caps for emphasis in code or prose (write "not"/"never", not
  "NOT"/"NEVER"). Scan your own diff for both before presenting it.

## Pointers

- `README.md` — setup, layout, running, roadmap.
- `.github/skills/` — crops, crop-science, annotation, training, evaluation,
  pipeline-design, visual-analysis, delivery. Load before acting in a domain.
