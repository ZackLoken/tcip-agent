# TCIP GUI — Implementation Plan

Step-by-step plan to build the unified browser-based TCIP GUI per [gui_design.md](./gui_design.md).

- **Slicing:** six vertical increments. Each ends at a validation gate that
  produces something usable and gives the PD a concrete decision point before
  the next slice begins. Main stays releasable at every gate.
- **Rough total effort:** 15–22 focused working days. Slice 1 is the largest
  and hardest; later slices compound on its infrastructure.
- **Critical path:** Slice 0 → Slice 1 blocks all others. Slices 2, 3, 4 can
  be parallelised later if needed, but default to sequential.

Each step lists: what changes, where it lives, what's tested, what I'll show
the PD before moving on.

---

## Slice 0 — Engine migration (vendor-copy, non-behavioral)

**Goal.** One canonical Python engine library used by both the MCP server and
the FastAPI backend. No behavioral change; only module consolidation.

**Estimated effort:** 1–2 days.

### Steps

1. **Snapshot baseline.** Run `pytest tests/ -v` on current `main`; record
   green baseline. Commit any loose work-in-progress.
2. **Diff audit.** Produce a side-by-side diff report of the overlapping
   modules (`label_io.py`, `matching.py`, `utils.py`) between yolo-annotator
   and tcip-annotation. Flag every function signature difference. Save to
   `docs/engine_migration_diff.md`.
3. **Merge `label_io.py`.** Adopt yolo-annotator's implementation where it
   has more features; preserve existing tcip-annotation public API (no
   signature changes). Ensure `parse_detect_labels`, `parse_segment_labels`,
   `parse_detect_predictions`, `parse_segment_predictions` return the same
   shape as today. Unit test: round-trip YOLO write-read on fixtures from
   Valley_Farm.
4. **Merge `matching.py`.** Same rule — API-compatible. Unit test:
   `compute_matches` on a synthetic TP+FP+FN fixture matches expected.
5. **Merge `utils.py`.** Add `auto_orient_image` and any yololabeler helpers
   (font, color, halo_text if not already there). Preserve existing imports.
6. **Add `rendering.py`.** New module. Move small pixel helpers (halo_text,
   overlay line width, color picker) into it. No existing callers to break.
7. **Port `annotation_engine.py`.** Copy from
   `yolo-annotator/src/yololabeler/annotation/engine.py`. Grep for
   `tkinter`, `ctk`, `customtkinter`, `messagebox` — refactor any Tk touch
   points into callback hooks (functions the caller provides) so the engine
   stays pure. Unit tests: polygon build, undo, redo, delete, select.
8. **Port `review_engine.py`.** Same pattern, from
   `yolo-annotator/src/yololabeler/review/engine.py`. This is 430 lines of
   careful logic (backup before first edit, deduplicate reviewed entries,
   rebuild_review_detections under filter changes, check_image_review_complete).
   Unit tests: each public method pinned to a fixture.
9. **Verify MCP tools still pass.** `pytest tests/ -v`. Every currently-passing
   test must stay green. If anything red, revert the offending merge step and
   iterate.
10. **Retire `.tcip/events/` file-watcher bridge.** Add FastAPI stub
    endpoint `POST /api/events/{panel}` in `tcip-web` that accepts the same
    payload shape. Update `push_panel_data` MCP tool to POST instead of write
    file. Delete `.tcip/events/` directory creation code. Leave the VS Code
    extension event reader code in place for now — it just stops receiving
    events; retirement of the extension is a separate decision (per
    design doc §3).

### Validation gate (0 → 1)

- `pytest tests/ -v` green.
- Engine module unit tests green.
- One manual smoke test: call `push_panel_data` from a Python REPL; see the
  payload arrive at `POST /api/events/review` on the (not-yet-built)
  FastAPI stub route.
- PD reviews the diff report and the resulting `tcip-annotation` module
  tree. Signs off before Slice 1.

---

## Slice 1 — Backend skeleton + Annotate tab + Review tab with sync

**Goal.** A working browser app where the PD can load Valley_Farm images,
walk through TP/FP/FN detections, add missed catkins via the Annotate tab
with synced zoom + prediction-reference overlay, and save cleaned labels.
Everything else is stubs.

**Estimated effort:** 5–7 days. This is the hardest slice — nails the
state sync architecture that slices 2–4 build on.

### Backend steps

1. **`tcip_web/state.py`.** `GuiState` as Pydantic model per design doc
   §5.1. Includes reducers for every state mutation type. Debounced
   persistence to `.tcip/state/gui.json` (500 ms window).
2. **`tcip_web/hub.py`.** WebSocket connection manager with broadcast.
   Message types: `state_delta`, `image_ready`, `metrics_update`,
   `inference_progress`. JSON-serialised; no binary.
3. **`tcip_web/port_discovery.py`.** On startup, write
   `.tcip/state/web_port.txt` with the port number. Respect
   `TCIP_WEB_PORT` env override.
4. **Routes: `tcip_web/routes/dataset.py`.** `GET /api/dataset/tree` returns
   the restructured layout tree; `GET /api/dataset/images?dataset=...&date=...`
   returns image list.
5. **Routes: `tcip_web/routes/images.py`.** `GET /api/images/{path:path}`
   serves image bytes with CORS; applies `auto_orient_image` on the fly
   so the browser receives EXIF-aligned pixels. Frontend never EXIF-rotates.
6. **Routes: `tcip_web/routes/annotate.py`.**
   - `GET /api/annotate/labels?image=...&task=detect|segment` — read YOLO labels.
   - `POST /api/annotate/labels` — save YOLO labels for an image. Writes through `tcip_annotation.label_io`.
   - `POST /api/annotate/open` — command to set Annotate tab on a given image with optional pred_reference overlay; broadcasts `state_delta` to browsers.
7. **Routes: `tcip_web/routes/review.py`.**
   - `GET /api/review/matches?image=...&iou=...&conf=...` — calls `tcip_annotation.compute_matches`, returns detections list.
   - `POST /api/review/action` — record accept/reject/edit for a detection; persists to `.tcip/state/review_state.json`; broadcasts delta.
   - `POST /api/review/navigate` — set current image/detection index; broadcasts.
8. **Routes: `tcip_web/routes/events.py`.** `POST /api/events/{panel}` — the
   entry point MCP tools use to push data to a panel; broadcasts to
   subscribers on that panel.
9. **Update MCP tools** to POST to FastAPI instead of writing
   `.tcip/events/`. Tools affected: `push_panel_data` (direct), any tool
   that currently writes to events dir (audit: grep `events_dir`).
10. **Parameterize `run_matching`.** Add optional `gt_detect_path`,
    `gt_segment_path`, `pred_detect_path`, `pred_segment_path` kwargs.
    Fall back to current auto-discovery when omitted. Used by Slice 3 Review
    tab flow with Valley_Farm's non-standard layout.

### Frontend steps

11. **Scaffold.** `packages/tcip-web/frontend/` with Vite + React +
    TypeScript + Tailwind + Zustand + react-konva. `vite.config.ts` proxies
    `/api/*` and `/ws/*` to the FastAPI backend in dev. Build output goes to
    `packages/tcip-web/static/` for production.
12. **Theme.** `frontend/src/theme.ts` with yololabeler palette (from
    `yolo-annotator/src/yololabeler/review/tab.py` constants:
    `#1E1E1E` / `#2D2D2D` / `#E0E0E0` / `#507754` / `#E6976B`). Tailwind
    config consumes it.
13. **Top bar, status bar, tab router.** Mirror yololabeler header: image
    nav (prev/next/counter/jump-to), class dropdown, mode toggle (box/polygon),
    visible/complete markers, review count badges. Status bar: zoom %, user,
    timing.
14. **Zustand store.** Types mirror backend `GuiState`. WebSocket listener
    applies `state_delta` messages. Reducers for UI-only state (hover, drag).
15. **Canvas component.** `components/Canvas/` with react-konva stage,
    pan/zoom (Ctrl+scroll, middle-click drag), coord conversion helpers,
    viewport culling (only render boxes intersecting the view).
16. **Annotate tab.** `tabs/AnnotateTab.tsx`:
    - Box tool, polygon tool, stream-polygon mode, snap-to-vertex toggle.
    - Class-colored overlays; selected polygon handles drawn on top.
    - Undo/redo per image.
    - Keyboard: `m` mode toggle, `Ctrl+Z` / `Ctrl+Shift+Z`, `Del`, `Esc`.
    - `pred_reference` overlay rendering (blue dashed) when set.
    - `POST /api/annotate/labels` on save or image change.
17. **Review tab.** `tabs/ReviewTab.tsx`:
    - On mount, `GET /api/review/matches` for current image.
    - Sequential walk-through; auto-zoom to each detection (match fills 1/3 of canvas, 3× context).
    - Filters: type (All/TP/FP/FN), class, status (All/Reviewed/Not Reviewed).
    - Context-aware buttons: TP=Confirm/Edit/Delete-GT, FP=Add-to-GT/Edit/Dismiss, FN=Keep-GT/Edit/Delete-GT.
    - Keyboard: `a/e/r`, `←→↑↓`, `h` help overlay.
    - `POST /api/review/action` on every decision.
18. **Review ↔ Annotate sync.** On `Edit` or `FP Accept`:
    - Frontend captures current `scale`, `offset_x`, `offset_y`.
    - Calls `POST /api/annotate/open` with pred_reference payload + captured view.
    - Backend broadcasts `state_delta`; Annotate tab picks up image + view + pred_reference overlay.
    - On return to Review tab (after user confirms annotation), show confirm dialog ("Save to GT? / Redo"); on Save → `POST /api/review/recompute` → match list refreshes, advance to next unreviewed.

### Validation gate (1 → 2)

- PD launches `python -m tcip_web` from a terminal in the repo root.
- Browser opens to `http://localhost:<port>`, defaults to Annotate tab on
  the Valley_Farm catkin dataset, image IMG_0134.
- PD walks through Review tab on IMG_0208 (the image they flagged as having
  missed catkins). Adds at least the known-missed catkin via Edit FP →
  draw new GT → save-and-return. Sees the detection list update.
- Cleaned label diff visible on disk: `git diff data/hazelnut/.../2-11-26/detect/IMG_0208.txt` shows one or more added lines.
- Canvas hits ≥30 fps during pan/zoom on IMG_0134 (289 polygons). Measured
  in Chrome performance tab.
- PD signs off. Commit-and-tag before Slice 2.

---

## Slice 2 — Training tab + live metrics

**Goal.** PD configures and launches a YOLOv11 catkin detector (fresh from
pretrained, on cleaned Slice-1 labels) from the browser. Live loss curves
stream as training runs. Best checkpoint registers with one click.

**Estimated effort:** 3–4 days.

### Backend steps

1. **`tcip_web/routes/training.py` expansion.** Most endpoints already exist
   (`launch`, `list`, `status`). Add:
   - `POST /api/training/validate` — wraps `validate_config` MCP tool.
   - `POST /api/training/register` — wraps `register_model` after a run finishes.
   - `GET /api/training/runs/compare?ids=...` — wraps `compare_experiments`.
2. **Multi-stage schedule support.** Extend training config schema to accept
   `stages: [{epochs, lr, batch, ...}, ...]` per PD preference.
   `launch_training` tool is extended (or a new wrapper in `tcip_web.training`
   orchestrates stages sequentially).
3. **WebSocket: `tcip_web/ws/training.py`.** Tails
   `.tcip/experiments/<exp_id>/metrics.jsonl`; broadcasts one
   `metrics_update` per new line. Handles run completion.

### Frontend steps

4. **`tabs/TrainingTab.tsx`.** Left pane: config editor (form + raw YAML),
   model architecture picker (presets for YOLOv11-det, YOLOv11-seg, Mask R-CNN),
   multi-stage schedule builder with per-stage lr/batch/epochs.
5. **Runs list.** Right side-panel with training runs, status badges
   (running/completed/failed), mAP50 / mAP50-95 quick-view.
6. **Live curves.** Center/bottom pane with plotly line chart: loss, mAP50,
   mAP50-95 over epochs. Updates via WS.
7. **Register-model button.** Enabled once a run completes. Prompts for
   model name/tag; wraps `register_model` MCP tool.
8. **Compare runs.** Multi-select in runs list opens a compare table view
   (metrics side-by-side).

### Validation gate (2 → 3)

- PD configures a YOLOv11-det training on Valley_Farm cleaned catkin labels,
  imgsz=640 with SAHI tiling as preprocessing, 2-stage schedule (coarse LR
  then fine), launches from browser. Sees live loss curves update during
  training.
- Training completes. PD clicks Register Model; the new model shows up in
  Inference tab's (stubbed) model dropdown via `/api/models/registered`.
- Same flow for YOLOv11-seg.
- Commit-and-tag.

---

## Slice 3 — Inference + Results tabs

**Goal.** PD runs SAHI-tiled inference from the catkin detector and segmenter,
plus the bush foreground segmenter (trained in-slice), across all five
Valley_Farm dates (5,363 images). Plant-IDs resolve via GPS clustering.
Results tab renders per-plant phenology curves and exports the
`catkin_05/50/95per_date` CSV.

**Estimated effort:** 3–4 days.

### Backend steps

1. **Bush segmenter training.** Within Slice 2's Training tab (already built),
   PD trains the bush segmenter on the 81 Valley_Farm 3-2-26 bush labels.
   Validates on other dates via inference spot-check (Slice 3 flow).
2. **`tcip_web/routes/inference.py` expansion.**
   - `POST /api/inference/launch` — wraps `run_inference` + SAHI.
   - `GET /api/inference/{job_id}/status` — progress.
   - `GET /api/inference/{job_id}/preview?limit=N` — returns first N predictions for grid.
3. **Inference worker.** Async job queue (simple: one worker process, jobs
   in SQLite queue or asyncio Queue). Worker runs SAHI-tiled inference
   per image; broadcasts `inference_progress` WS per image completed.
4. **Plant-ID mapping service.** `tcip_web/plant_mapping.py` (new module):
   - Read EXIF timestamps + GPS from each image in a date folder.
   - Cluster by GPS within date; order by timestamp within cluster.
   - Match cluster centroids to `plant_locations/*.csv` by nearest-neighbour
     within tolerance (RTK uncertainty); anchor row endpoints with human
     confirmation if ambiguous (lightweight modal in Results tab).
   - Persist mapping to `.tcip/state/plant_mapping.json`.
5. **`tcip_web/routes/results.py`.**
   - `GET /api/results/per_plant_curves?date_range=...` — return JSON of
     per-plant counts and elongation ratios over time.
   - `POST /api/results/export_csv` — wraps `export_results_csv`.
6. **Stage 3 elongation threshold service.** Based on detection bbox height
   (or segmentation principal axis length), classify each catkin as
   elongated vs dormant. Threshold calibrated from Feb-11 (all dormant)
   upper tail + 3-24 visual spot-check. Exposed as config parameter.

### Frontend steps

7. **`tabs/InferenceTab.tsx`.** Model dropdown (populated from
   `/api/models/registered`), folder picker, conf / IoU / max_det sliders,
   SAHI toggle with slice size / overlap, launch button, progress bar,
   preview grid with live-filling thumbnails.
8. **`tabs/ResultsTab.tsx`.**
   - Experiment compare table.
   - Worst-predictions grid (wraps `get_worst_predictions`).
   - Plant-ID mapping confirmation dialog (first time, or on-demand).
   - Per-plant phenology chart: plotly with x=date, y=elongated/total ratio, one line per plant, onset date markers for 5/50/95%.
   - CSV export button with schema preview.

### Validation gate (3 → 4)

- Inference runs across all 5 dates and 5,363 images without crashing.
- Plant-ID mapping produces non-null assignments for >80% of plants (RTK
  accuracy + row sequence ambiguity is the bottleneck; <100% is OK).
- Per-plant curves render for at least 50 plants.
- CSV exports with columns `plant_id, accession, catkin_05per_date,
  catkin_50per_date, catkin_95per_date` and passes a schema check.
- Commit-and-tag.

---

## Slice 4 — Tuning tab

**Goal.** PD configures an HPO sweep on the catkin detector, launches it,
reviews trial progress, exports the best config back to Training tab.

**Estimated effort:** 2–3 days.

### Backend steps

1. **`tcip_web/routes/tuning.py`.**
   - `POST /api/tuning/launch` — wraps `run_hpo` MCP tool.
   - `GET /api/tuning/{sweep_id}/trials` — trial-level metric stream.
2. **WebSocket: `tcip_web/ws/tuning.py`.** Broadcasts `trial_complete`
   events as trials finish.

### Frontend steps

3. **`tabs/TuningTab.tsx`.** Search space builder (categorical /
   continuous / log-uniform fields), objective picker (mAP50 /
   mAP50-95 / recall), trial budget.
4. **Parallel coordinates plot.** Plotly; axes = hyperparameters, line per
   trial colored by objective value. Best trial highlighted.
5. **Best-config export.** Button that drops the winning config into the
   Training tab's config editor as a new preset.

### Validation gate (4 → 5)

- PD launches a small-budget HPO on the catkin detector (e.g. 10 trials).
- Live trial progress visible.
- Best config exports to Training tab and is runnable from there.
- Commit-and-tag.

---

## Slice 5 — Phase 1 completion + polish

**Goal.** Everything works end-to-end through the GUI. Phase 1 exercise is
called complete. Retrospective written.

**Estimated effort:** 1–2 days.

### Steps

1. **Help overlay.** `h` key opens overlay mirroring yololabeler's help
   panel verbatim. Keyboard + mouse reference.
2. **Keyboard shortcut audit.** Every yololabeler shortcut works in the
   equivalent TCIP tab. Cross-tab shortcuts documented.
3. **EXIF handling audit.** Single code path in backend
   (`/api/images/{path}`) applies `auto_orient_image`. Grep for other
   image-reading code paths, verify each either goes through the backend
   or applies the same helper.
4. **Visual polish pass.** Side-by-side screenshot comparison of Annotate
   and Review tabs (TCIP GUI vs yololabeler). Fix any visible gaps in
   spacing, color, typography, icon weight.
5. **Full Phase 1 end-to-end rehearsal.** PD runs the whole Phase 1
   pipeline through the GUI, start to finish, on Valley_Farm data. Any
   friction discovered → `claude_reports`.
6. **`project_retrospective("hazelnut-catkin-phase1")`.** Captures
   everything that worked, broke, surprised. Feeds Phase 2 planning.

### Validation gate (5 → done)

- `catkin_05/50/95per_date` CSV on disk for the Valley_Farm plots.
- Retrospective markdown on disk.
- PD signs off on Phase 1 closure.

---

## Cross-cutting concerns

### Testing

- **Unit tests:** every engine module in `tcip-annotation` has tests. Each
  backend route has at least a smoke test. Each frontend tab has one e2e
  test (Playwright) covering the happy path.
- **Integration test:** Slice 1 includes a scripted end-to-end test:
  start backend → open test browser → load image → draw box → save →
  verify file on disk. Automated from Slice 1 onward.

### Git and CI

- One PR per slice, merged to `main` at each validation gate.
- Commit discipline: first commit in each slice is scaffolding (compiles,
  tests pass, no behavior); subsequent commits add one testable behavior
  each.
- Run existing CI (`pytest`, `npx tsc --noEmit`) on every PR.

### Documentation

- Each slice updates `docs/gui_design.md` if design decisions shift during
  implementation (expected).
- `packages/tcip-web/README.md` kept current with run / develop / deploy
  instructions.

### Audit trail

- Every user action that mutates state → `.tcip/sessions/<session_id>.jsonl`
  with event type `gui_action`.
- Every MCP tool that touches the GUI remains `@audited` — double-logged
  (once by decorator, once by GUI event), but cheap and the redundancy
  catches divergences.

---

## Open questions for PD

- **VS Code extension fate.** Once the web GUI lands, the extension panels
  become redundant. Recommend: keep extension repo alive as a thin wrapper
  that opens the web app in a VS Code simple-browser tab; retire the
  webview panels. Call this after Slice 5.
- **Dev server port.** Default to `8765` unless you prefer another. FastAPI
  reads `TCIP_WEB_PORT` env override; port is written to
  `.tcip/state/web_port.txt` for MCP tool discovery.
- **Authentication.** None planned for Phase 1 (localhost single-user).
  Flag: first multi-user use (colleagues accessing a shared server) needs
  auth before the GUI is exposed beyond localhost.

---

## Definition of done

All of the following true:

- Every slice's validation gate passed.
- `catkin_05/50/95per_date` CSV produced for Valley_Farm plots.
- `.tcip/retrospectives/hazelnut-catkin-phase1.md` exists.
- The new GUI is the default way to drive TCIP for a breeder working on a
  single-trait pipeline end-to-end.
- The `yolo-annotator` engine modules have been merged into
  `tcip-annotation` (no residual duplication).
- The `.tcip/events/` file-watcher bridge is retired.

## Revision history

- **2026-04-23** — initial plan drafted by Claude Opus 4.7 for PD approval.
