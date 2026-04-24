# TCIP GUI — Design Document

Design spec for the unified TCIP graphical interface. Scope covers Phase 1 of the
hazelnut catkin phenology exercise and the broader 6-crop platform. Replaces the
partial VS Code extension panels with a browser-based application that both
human operators and the Claude agent can drive.

- **Last updated:** 2026-04-23
- **Author:** Claude Opus 4.7 + Zack Loken (PD)
- **Status:** draft for review → plan mode → implementation
- **Related:** [vision.md](./vision.md), [tool-audit-2026-04-20.md](./tool-audit-2026-04-20.md)

---

## 1. Context

Phase 1 of the hazelnut catkin exercise surfaced the single most consequential
friction in the current system: **there is no GUI up to the standard required
for daily use by breeders and annotators**. Two partial attempts exist:

- `packages/tcip-vscode/` — VS Code webview panels (Annotation, Review, Training,
  HPO, Inference, Results). Sketch-level: no cross-panel state sync, hardcoded
  path conventions, visual polish well below what the PD's team already uses
  with `yolo-annotator`.
- `packages/tcip-web/` — FastAPI server that wraps MCP tools as REST endpoints.
  Has WebSocket scaffolding for training progress but only a 104-line static
  HTML stub for a frontend.

Meanwhile **`yolo-annotator`** (YoloLabeler, `C:\Users\exx\Documents\GitHub\yolo-annotator`)
is a CustomTkinter desktop app with Annotate + Review tabs that the PD and
breeding colleagues consistently praise for appearance and functionality.

Phase 1 cannot finish without a GUI. The annotation review step *needs* the
review interface. Downstream training / inference / results steps need their
respective tabs. This document specifies what to build.

## 2. Requirements (from PD)

1. **Tech stack:** Browser + FastAPI (extend the existing `tcip-web` package).
2. **Process model:** Separate processes with shared state. GUI runs standalone;
   Claude agent (via MCP) reads and mutates state through the FastAPI backend.
3. **Scope:** All six tabs before Phase 1 can be called complete —
   Annotate, Review, Training, Tuning, Inference, Results.
4. **Code reuse:** Direct reuse of `yolo-annotator`'s non-GUI engines
   (`state.py`, `label_io.py`, `matching.py`, `utils.py`, `rendering.py`), extended
   to cover additional annotation formats (COCO, VOC, LabelMe) beyond YOLO.
   **The `yolo-annotator` GUI is the visual/UX spec** — layout, buttons, colors,
   theming, typography, keyboard map, interaction model.
5. **Primary flow for Phase 1:** *Agent prepares, human reviews.*
   Shared/interleaved co-driving (agent writes annotations while human works) is
   deferred to a later phase.

## 3. Non-goals

- **Not a replacement for the MCP server.** Claude Code continues to spawn
  `tcip-mcp` over stdio; MCP tools remain the audit-logged primitives the agent
  invokes. The GUI backend is a *client* of that system, not a re-implementation.
- **Not multi-user yet.** Designed for one human + one agent concurrently. Multi-user
  coordination is flagged as a future concern (see §10 risk 3).
- **Not a mobile or tablet interface.** Desktop browser only. Tablet-optimized
  UX for field use remains an open question in [vision.md §9.7](./vision.md#9-open-questions).
- **Not a replacement for the VS Code extension in the short term.** The
  extension stays installable; its webviews become thin clients that redirect
  to the web app, or are retired after migration. Decision deferred.

## 4. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Claude Code (MCP client over stdio)                   │
│                              │                                           │
│                              ▼                                           │
│                    ┌─────────────────────┐                               │
│                    │   tcip-mcp server   │  (Python, existing)           │
│                    │   @mcp.tool()       │                               │
│                    │   @audited          │                               │
│                    └─────────┬───────────┘                               │
│                              │ HTTP POST / GET                           │
│                              ▼                                           │
│                    ┌─────────────────────┐    WebSocket   ┌────────────┐ │
│                    │  tcip-web (FastAPI) │ ◀────────────▶ │  Browser   │ │
│                    │   state store       │                │  (React)   │ │
│                    │   REST + WS         │                │            │ │
│                    └─────────┬───────────┘                └────────────┘ │
│                              │                                           │
│                              ▼                                           │
│         ┌──────────────────────────────────────────────┐                 │
│         │   tcip-annotation (Python engine library)    │                 │
│         │   label_io, format_io, matching, rendering,  │                 │
│         │   sam_wrapper, review_engine, utils           │                 │
│         └──────────────────────────────────────────────┘                 │
│                              │                                           │
│                              ▼                                           │
│    ┌──────────────────────────────────────────────────────────────┐      │
│    │  .tcip/ workspace state (persistence + audit)                │      │
│    │  state/gui.json   sessions/*.jsonl   audit.jsonl              │      │
│    │  experiments/      artifacts/       retrospectives/           │      │
│    │  reports/          events/          (remove after migration)  │      │
│    └──────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

**Three runtime processes:**

1. **Claude Code + tcip-mcp server** — existing. Agent invokes tools over stdio.
   MCP tools that affect GUI state now make an HTTP call to the FastAPI backend.
   Audit logging via `@audited` decorator is unchanged.
2. **tcip-web FastAPI backend** — expanded. Holds live GUI state in memory,
   persists periodic snapshots to `.tcip/state/gui.json`, broadcasts state
   changes via WebSocket, serves the React frontend, exposes REST for all
   user actions, imports `tcip-annotation` for matching / I/O / rendering.
3. **Browser (React SPA)** — new. Connects to backend via REST + WebSocket.
   Six tabs matching yololabeler + extension surface.

**Shared libraries:**

- `packages/tcip-annotation/` — migrated from its current state, absorbing the
  relevant engine modules from `yolo-annotator`. One canonical place for
  label/prediction I/O, matching, SAM wrapper, review engine. Used by both
  `tcip-mcp` (for audited tool implementations) and `tcip-web` (for live GUI
  operations). No duplication.

## 5. State model

Live state lives in the FastAPI backend process. Persisted snapshots and audit
logs live under `.tcip/`. The backend is the single writer to its in-memory
store; all other parties (agent, browser) read/write through it via HTTP/WS.

### 5.1 In-memory state (FastAPI)

Python dataclasses (or Pydantic models) structured roughly as:

```python
class GuiState:
    project_path: Path
    active_tab: Literal["annotate", "review", "training", "tuning", "inference", "results"]

    # Dataset selection
    dataset_root: Path            # e.g. data/hazelnut/catkin_05-50-95-per_date/Valley_Farm
    annotation_type: str          # "catkin" | "bush" | ...
    date: str                     # "2-11-26"
    image_list: list[str]
    current_image_index: int

    # Shared annotation state (Annotate + Review both read)
    boxes: list[Box]              # active GT in memory, derived from disk on load
    polygons: list[Polygon]
    class_names: dict[int, str]
    class_colors: dict[int, str]
    mode: Literal["box", "polygon"]

    # Shared view state (the thing that makes Review↔Annotate sync trivial)
    scale: float
    offset_x: float
    offset_y: float

    # Review state
    review_matches: Matches | None
    review_detections: list[Detection]
    review_detection_idx: int
    review_filter_type: Literal["all", "tp", "fp", "fn"]
    review_filter_class: int | Literal["all"]
    review_status_filter: Literal["all", "reviewed", "not_reviewed"]
    review_state: dict[str, dict]         # per-image per-detection action log
    pred_reference: PredReference | None  # blue dashed overlay in Annotate

    # Training / Tuning state
    training_runs: list[RunSummary]       # mirrors .tcip/experiments/
    active_run_id: str | None
    live_metrics: dict[str, list[float]]  # streamed, subset of metrics.jsonl

    # Inference / Results state
    inference_jobs: list[InferenceJob]
    active_inference_id: str | None
    predictions_cache: dict[str, list[Prediction]]  # image_stem → preds
```

### 5.2 Persistent state (`.tcip/`)

Extends the existing layout:

```
.tcip/
  state/
    gui.json           # snapshot of backend GuiState on change (debounced, 500ms)
  sessions/*.jsonl     # append-only session event log (existing)
  audit.jsonl          # append-only audit log (existing, from @audited)
  experiments/         # unchanged
  artifacts/           # unchanged
  retrospectives/      # unchanged
  reports/             # unchanged
  events/              # DEPRECATED — file-watcher bridge no longer needed
```

- `gui.json` is written on every debounced state change. On backend startup,
  the last snapshot is restored. Not a database — just enough to survive a
  backend restart mid-session.
- `sessions/*.jsonl` gains new event types: `gui_action` (user clicks),
  `state_transition` (tab switch, image change). This gives the retrospective
  tool a richer signal than just MCP tool calls.
- `events/` (the current file-watcher bridge used by the VS Code extension) is
  retired in favor of direct HTTP posts from MCP tools. One less moving part.

### 5.3 Data layout (dataset-side)

The restructured Valley_Farm layout from Phase 1 (`images/<date>/`,
`annotations/<type>/<date>/{detect,segment}/`, `models/<name>/predictions/<task>/`,
`plant_locations/*.csv`) becomes the canonical shape. The FastAPI backend
resolves paths from these conventions. Legacy `data/labels/detect/` etc. in
older projects are supported by an ingest wizard that produces the new layout.

## 6. IPC and coordination

Three channels, each with a clear direction.

### 6.1 Agent → GUI (state mutations)

MCP tool invokes HTTP POST to the backend. Examples:

| MCP tool | HTTP call |
|---|---|
| `push_panel_data("review", "load_matches", data)` | `POST /api/review/matches` |
| `visualize_annotations(image)` → returns path | `POST /api/annotate/open` with image_path |
| `run_matching(image, iou, conf)` | backend computes, returns; GUI reads via WS broadcast |
| `launch_training(config)` | `POST /api/training/launch`; training run_id broadcasted |

Existing MCP tools that write to `.tcip/events/` get rewritten to POST to
`http://localhost:<port>/api/events/<panel>`. The port is discovered via an
env var (`TCIP_WEB_PORT`) or a file (`.tcip/state/web_port.txt`) the backend
writes on startup.

### 6.2 GUI → Agent (user actions)

User actions post to the backend's REST API. The backend:

1. Updates in-memory state.
2. Persists to `.tcip/state/gui.json`.
3. Appends `gui_action` event to `.tcip/sessions/<session_id>.jsonl`.
4. Broadcasts state delta via WebSocket to all connected browsers.

The agent reads recent session events (existing pattern) to learn what the
human did. There is no direct browser→Claude path — the agent observes state
via existing tools (`get_session`, `get_project_status`, custom `get_gui_state`).

### 6.3 Backend → Browser (live updates)

Single WebSocket connection per browser tab. Server pushes JSON messages:

```json
{"type": "state_delta", "path": "review.detection_idx", "value": 42}
{"type": "image_ready", "url": "/api/images/IMG_0134.JPG"}
{"type": "metrics_update", "run_id": "exp-001", "epoch": 23, "mAP50": 0.79}
{"type": "inference_progress", "job_id": "inf-004", "done": 340, "total": 1742}
```

The frontend uses a typed action router that updates its own Redux/Zustand
store on each message.

## 7. Module layout

### 7.1 Backend: `packages/tcip-web/`

```
tcip-web/
  pyproject.toml
  Dockerfile
  src/tcip_web/
    __init__.py
    __main__.py              # entrypoint: uvicorn
    app.py                   # FastAPI app mount
    state.py                 # GuiState, reducers, persistence
    hub.py                   # WebSocket connection manager + broadcast
    routes/
      __init__.py
      dataset.py             # /api/dataset/*
      annotation.py          # /api/annotate/*   (new)
      review.py              # /api/review/*     (new)
      training.py            # /api/training/*
      tuning.py              # /api/tuning/*     (new)
      inference.py           # /api/inference/*
      results.py             # /api/results/*    (new)
      images.py              # /api/images/*     (serve image files with CORS)
      events.py              # /api/events/*     (MCP tool entry point)
    ws/
      training.py            # live metrics
      inference.py           # live progress
      state.py               # generic state delta broadcaster
  static/                    # built React bundle output
  frontend/                  # React source (see §7.3)
```

### 7.2 Shared engine: `packages/tcip-annotation/`

Absorb these from `yolo-annotator/src/yololabeler/`:

- `state.py` → becomes `tcip_annotation/session_state.py` (renamed to avoid
  clash with `tcip_annotation.state` which exists today)
- `label_io.py` → keep current `tcip_annotation/label_io.py`, merge/extend
- `matching.py` → keep current, merge with yololabeler's for feature parity
- `utils.py` → `tcip_annotation/utils.py` — merge
- `rendering.py` → `tcip_annotation/rendering.py` — new module
- `review/engine.py` → `tcip_annotation/review_engine.py`
- `annotation/engine.py` → `tcip_annotation/annotation_engine.py`

Extend `format_io.py` to cover all formats (YOLO, COCO, VOC, LabelMe) with
clean internal representation. This addresses vision.md §3.3.

### 7.3 Frontend: `packages/tcip-web/frontend/`

React + TypeScript + Vite. Styling: Tailwind CSS with a theme file mirroring
yololabeler's palette (`#1E1E1E` BG, `#2D2D2D` canvas BG, `#E0E0E0` FG,
`#507754` SI_GREEN accent, `#E6976B` SI_PERSIMMON secondary). State via
Zustand (lightweight Redux alternative). Canvas via **Konva.js** — better
performance characteristics than Fabric for 2000+ polygons, first-class React
bindings via `react-konva`.

```
frontend/
  index.html
  vite.config.ts
  tailwind.config.ts
  src/
    main.tsx
    App.tsx                  # top-level layout: title bar + tabs + status bar
    theme.ts                 # yololabeler color palette, typography
    store/
      index.ts               # zustand store + WS subscriptions
      types.ts               # mirrors backend GuiState
    hooks/
      useWebSocket.ts
      useKeyboardShortcuts.ts
      useCanvas.ts
    api/
      client.ts              # typed REST client
      ws.ts                  # WS connection + routing
    components/
      TopBar.tsx             # class selector, mode toggle, image nav
      StatusBar.tsx          # zoom / user / stats
      Canvas/
        AnnotationCanvas.tsx # Konva stage
        ReviewCanvas.tsx
        shared/              # pan/zoom, coord helpers, overlays
    tabs/
      AnnotateTab.tsx        # Annotate tab (yololabeler parity)
      ReviewTab.tsx          # Review tab (yololabeler parity)
      TrainingTab.tsx
      TuningTab.tsx
      InferenceTab.tsx
      ResultsTab.tsx
```

## 8. Phase 1 feature list (per tab)

### 8.1 Annotate tab

- YOLO box and polygon drawing; toggle mode; class selector.
- Undo/redo (per-image), keyboard shortcuts matching yololabeler.
- Zoom (Ctrl+scroll), pan (middle-click drag), scroll up/down, shift-scroll left/right.
- Snap-to-vertex toggle, stream-draw polygon mode.
- Delete annotation (selection + Del).
- Visible-only filter (hide drawn annotations for verifying empty regions).
- Save on image change / explicit save; autosave toggle.
- Load GT from disk on image load; write GT back in YOLO format (and COCO/VOC/LabelMe exports).
- Prediction reference overlay (blue dashed) when switched-to from Review.
- **Deferred to Phase 2:** rotated boxes, keypoints, auto-suggest, multi-instance selection.

### 8.2 Review tab

- Compute TP/FP/FN via `tcip_annotation.compute_matches`; IoU & conf sliders.
- Walk through detections sequentially with auto-zoom (~1/3 canvas fill, 3× context).
- Filters: type (All/TP/FP/FN), class, review status (All/Reviewed/Not Reviewed).
- Accept/Edit/Reject per-detection with yololabeler semantics:
  - TP: A=confirm, E=edit GT in Annotate, R=delete GT.
  - FP: A=add pred to GT (→Annotate), E=same, R=dismiss.
  - FN: A=keep GT, E=edit GT in Annotate, R=delete GT.
- Sync with Annotate: on E or FP-A, copy `scale/offset`, mode, and pred reference; on return show confirm dialog; re-run matching; advance.
- Per-image complete marker; `review_state.json` persisted under `.tcip/state/`.
- **Deferred to Phase 2:** batch actions, multi-model comparison, segmentation diff view.

### 8.3 Training tab

- Form + YAML editor for model spec (backbone / neck / heads / loss) + hyperparameters. Preset templates for YOLO detect/seg, Mask R-CNN, etc.
- Validate config (wraps `validate_config` MCP tool).
- Launch button (wraps `launch_training`); status poll via `check_training_status`.
- Live loss curves via WebSocket; TP/FP/FN on val set per epoch.
- Runs list (from `list_training_runs`), compare runs (wraps `compare_experiments`).
- Multi-stage training schedule editor (PD preference): stages with LR + batch progression.
- Register-model button on best checkpoint (wraps `register_model`).
- **Deferred to Phase 2:** distributed training UI, checkpoint browser, loss surface viz.

### 8.4 Tuning tab

- HPO config builder: search space (categorical / continuous / log-uniform), objective metric, trial budget.
- Launch button (wraps `run_hpo`).
- Live sweep progress: trials completed, best metric so far.
- Parallel-coordinates plot of trials (Plotly), highlight best-config path.
- Export best config to Training tab as starting point.
- **Deferred to Phase 2:** population-based training, budget re-allocation.

### 8.5 Inference tab

- Model selector from `list_registered_models`.
- Image folder selector with preview grid (thumbnail + count).
- Conf / IoU / max_det sliders; SAHI tile toggle with slice size + overlap.
- Launch button (wraps `run_inference`) with progress bar (live via WS).
- First-N preview grid: image thumbnails with prediction overlays.
- Export predictions to YOLO format (wraps `export_predictions_yolo`).
- **Deferred to Phase 2:** batch comparison across models, prediction filtering by GPS.

### 8.6 Results tab

- Experiment compare table (mAP, precision, recall per run).
- Worst-predictions grid (wraps `get_worst_predictions`).
- Per-plant phenology chart (time series of catkin count + % elongated; onset-date markers for 5/50/95%).
- CSV export (wraps `export_results_csv`); breeder-ready per-plant rows.
- Pipeline retrospective view: reads `.tcip/retrospectives/*.md` for context.
- **Deferred to Phase 2:** multi-trait dashboards, cross-year comparison, statistical testing.

## 9. Visual and interaction spec

Exact match to yololabeler on the overlapping tabs (Annotate, Review):

- Color palette as stated in §7.3.
- Top bar: image name, image counter, prev/next buttons, class selector dropdown, mode toggle, visible-toggle. Right-side: filter / complete marker / review count badges.
- Status bar (bottom): zoom %, current user, per-image timing.
- Keyboard map identical: `a / e / r` accept/edit/reject; `m` mode toggle; `h` help; `← → ↑ ↓` nav.
- Help overlay shown on `h` with keyboard + mouse reference (mirror yololabeler's help panel verbatim).

New tabs (Training, Tuning, Inference, Results) adopt the same theme and
typography but introduce new UI patterns (forms, tables, charts) that
yololabeler doesn't have. These borrow from the existing VS Code panels'
structural decisions (e.g., a config sidebar + main content area) but
rendered in the browser.

## 10. Open risks and mitigations

1. **Scope is large.** Six tabs with synced state, a new canvas library,
   engine migration, and agent integration is 2-4 weeks of solid work
   depending on parallelism. Phase 1 exercise blocks on this.
   *Mitigation:* Slice the build into vertical increments. First increment:
   backend state skeleton + Annotate tab + Review tab with sync, on
   Valley_Farm catkin data. Second increment: Training tab wired to an
   already-working MCP training tool. Third: Inference + Results. Fourth:
   Tuning. Each increment produces a usable artifact.

2. **Canvas performance.** 5712×4284 images with 2000+ annotations can
   stutter. *Mitigation:* Konva virtualization (only draw objects intersecting
   the viewport); image tiling for zooming above 2x; debounced redraws on
   pan. Benchmark on IMG_0134 as acceptance gate.

3. **Multi-user coordination.** Two browsers pointed at one backend = last
   writer wins. Acceptable for Phase 1 (one PD + one agent). Flag for Phase 2
   when Savanna Institute colleagues join.

4. **MCP → FastAPI discovery.** Claude Code spawns the MCP server; how does
   MCP know where the FastAPI server runs? *Mitigation:* On backend startup,
   write `.tcip/state/web_port.txt` with the port. MCP tools read this file
   on first GUI-affecting call. Fall back: `TCIP_WEB_PORT` env var.

5. **SAHI-tiled inference in the browser preview.** Running SAHI on 1742 test
   images is heavy; the preview grid can't wait on the full sweep.
   *Mitigation:* Backend runs inference async in a worker; the preview grid
   renders each image as its result lands (WebSocket). User can start reviewing
   first images while later ones are still processing.

6. **Annotation canvas library choice.** Konva is the current proposal,
   Fabric is an alternative used by the VS Code extension. Fabric's object
   model is richer but its performance degrades earlier on dense polygon
   sets. *Mitigation:* Prototype the first 100 polygons in both and pick
   based on measured frame rate. Commit to one.

7. **MCP tool inventory churn.** §3.5 of vision.md already flags 54 tools is
   too many. This doc doesn't reduce the count — it just re-points several
   tools at HTTP instead of file events. The tool audit in §4.2 of vision.md
   should happen after the GUI lands, using friction data from GUI operations
   as the signal.

8. **Image rotation / EXIF handling.** This session's data showed labels are
   authored against EXIF-aligned canvas. The backend must apply
   `ImageOps.exif_transpose` uniformly at ingest and in all rendering. One
   helper, one place.

9. **Engine migration risk.** `tcip-annotation` already has `label_io.py`,
   `matching.py`, `state.py`, `utils.py`. The yololabeler versions have
   diverged features. *Mitigation:* Diff before merging; keep a test suite
   that covers both old and new call sites. Do the merge as its own commit
   with zero behavioral changes, then build on it.

10. **Deferring VS Code extension retirement.** If we commit to the web app,
    the VS Code extension becomes redundant. *Mitigation:* Don't decide now.
    Ship the web app first; if the extension adds value (inline file ops,
    Copilot Chat integration), keep it as a thin wrapper that embeds the web
    app in a webview panel. If not, retire it after Phase 2 retrospective.

## 11. What this unlocks for Phase 1

With the GUI in place:

1. PD runs the new Review tab on the Valley_Farm catkin data. Agent has
   already prepared matches at conf=0.5; PD walks through, adds missed
   catkins via the Annotate-sync flow, saves cleaned labels.
2. Agent builds the Training tab config for a clean detector + segmenter on
   the cleaned labels. PD inspects, clicks Launch. Training runs; curves
   visible live.
3. Agent builds the Bush segmenter config; same flow.
4. Agent builds the Inference config for the catkin + bush models across
   all five dates. PD launches; live progress; prediction previews arrive
   as each date finishes.
5. Agent runs plant-ID matching across dates via GPS clustering (not a GUI
   step, runs silently via MCP). Results ingested into the Results tab.
6. Results tab shows per-plant phenology curves and the final
   `catkin_05/50/95per_date` CSV. PD downloads.
7. Phase 1 retrospective captures everything that broke, everything that
   worked, and what to change before Phase 2.

This is the system that finishes Phase 1. This is the system colleagues will
recognize as a tool they want to use.

## 12. Revision history

- **2026-04-23** — initial draft by Claude Opus 4.7 + Zack Loken, after
  five-question alignment on tech stack, process model, scope, reuse, drive
  pattern.
