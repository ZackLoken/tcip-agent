# tcip-web ↔ yolo-annotator parity gap

Side-by-side audit of what yolo-annotator's Annotate + Review tabs do that
tcip-web does not (yet). Organized by impact. Items marked **[done]** have
already shipped.

- **Reference:** `c:/Users/exx/Documents/GitHub/yolo-annotator/src/yololabeler/`
- **Target:** `packages/tcip-web/`
- **Scope note:** Training / Tuning / Inference / Results tabs are explicitly
  out of scope for this document — they'll be expanded once Annotate + Review
  are parity-complete.

---

## Shell (top bar + status bar + tab host)

| yolo-annotator feature | tcip-web today | Priority |
|---|---|---|
| SI logo in top-left | — | T2 |
| "YoloLabeler" title / branding | "TCIP" text | T3 |
| Archivo custom font (packaged) | Fallback to system sans | T3 |
| "Open Folder" button replaced by DatasetPicker | **[done]** | — |
| Top bar right: image name + editable counter + `/ total` + Prev/Next | Name + counter + Prev/Next but counter is read-only | **T1** |
| Type-a-number → jump to that image (filtered-aware) | — | **T1** |
| `Visible` checkbox — toggle annotation overlay | — | **T1** |
| `Complete` checkbox — mark current image complete | — | **T1** |
| Status filter dropdown: All / Complete / Partial / Unannotated | — | **T1** |
| Mode toggle (Box / Polygon) button | **[done]** | — |
| `Stream: Off/On` toggle button (polygon mode only) | — | **T1** |
| `Snap: Off/On` toggle button (polygon mode only) | — | **T1** |
| Color picker button (class-colored swatch) | — | **T1** |
| Class dropdown with counts `0: catkin (281)` | Class dropdown with name only | **T1** |
| `<New Class>` sentinel in class dropdown → input dialog | — | **T1** |
| Status bar: per-image timer (`Image time: 1:23` / `Review time: 0:45`) | — | T2 |
| Status bar: `User: <osuser>` | — | T2 |
| Status bar: `Zoom: 100%` live | **[done]** | — |
| Status bar (review): Class / Type / Status filter + GT/Pred checkboxes + A/E/R buttons + ◀ det-count ▶ | Partial — has filters but flat toolbar, no GT/Pred toggles, no A/E/R in status bar | **T1** |
| Welcome screen ("Click Open Folder to load images") | — | T2 |
| Tab switch records review time and resets annotate timer | — | T2 |
| Tab switch syncs scale/offset Review → Annotate | Partial — only on Edit/FP-Accept, not on plain tab switch | T2 |
| Default mode = polygon if segment labels exist, else box | — | T2 |
| Help overlay with keyboard + mouse (h toggles; ? hotkey) | **[done]** but doesn't match yololabeler's structure | T2 |

## Annotate — drawing & editing

| yolo-annotator feature | tcip-web today | Priority |
|---|---|---|
| Box draw: press+drag, release to commit | **[done]** | — |
| Box delete: right-click on box | Delete key only on selected polygon | **T1** |
| Polygon: left-click to place vertex | **[done]** | — |
| Polygon: double-click to close | Close via toolbar button | **T1** |
| Polygon: Escape cancels in-progress polygon / clears selection | **[done]** (partial) | — |
| Polygon: click a polygon → selects it | **[done]** | — |
| Polygon vertex drag (selected polygon) | — | **T1** |
| Polygon vertex insert on edge click (selected polygon) | — | **T1** |
| Polygon vertex delete on right-click (selected polygon) | — | **T1** |
| Polygon delete on right-click (if selected, inside polygon) | — | **T1** |
| Hover highlighting: vertices appear on hover; selected polygon cyan stroke | — | **T1** |
| Polygon preview line: dashed from last vertex to cursor | — | T2 |
| Snap (`s`): snap new vertices to existing polygon vertices within 15px canvas; yellow halo indicator | — | **T1** |
| Edge snap: snap to nearest polygon edge (`s` + streaming) | — | T2 |
| Stream (`v`): after click-start, vertices auto-placed as mouse moves (≥6px image gap) | — | T2 |
| Spacebar = synthetic left-click at cursor | — | T3 |
| Zoom: discrete levels (0.05 → 10.0, ~20 steps), `zoom_index` snaps | Continuous factor-of-1.15 | **T1** |
| Scale-dependent stroke width, label size, dash length | Fixed per-pixel values | **T1** |
| Active-class-only rendering in box mode (polygons of other classes hidden) | All classes rendered | **T1** |
| `Visible` checkbox hides *all* annotations | — | **T1** |
| Halo text on labels (black outline + fill) | Plain text | **T1** |
| Class label rendered next to each bbox / polygon as `{cid}: {name}` | — | **T1** |
| Per-image timer starts on first annotation touch | — | T2 |
| Auto-save on image navigation (Prev/Next) | Only on Ctrl+S | **T1** |
| 0-9 hotkey → select class by ID | — | **T1** |
| Undo stack capped at 30, per-image | **[done]** | — |
| In-progress polygon vertex undo (Ctrl+Z pops one vertex before popping full-snapshot) | Snapshot-only | T2 |
| Prediction reference overlay in Annotate (dashed blue) when editing from Review | **[done]** | — |
| `Enter` closes in-progress polygon | **[done]** (in addition to double-click, which is missing) | — |

## Review — walkthrough & actions

| yolo-annotator feature | tcip-web today | Priority |
|---|---|---|
| TP / FP / FN overlays rendered | **[done]** | — |
| Auto-zoom to detection (1/3 of canvas, 3× context) | Partial — queries DOM for canvas size | T2 |
| Prev / Next detection (← →) | **[done]** | — |
| Prev / Next image (↑ ↓) | **[done]** | — |
| `a` / `e` / `r` keys | **[done]** | — |
| Accept / Edit / Reject buttons | **[done]** (in toolbar) | — |
| Context-aware button labels (TP: "Confirm" / FP: "Add to GT" / FN: "Keep GT") | Fixed "Accept / Edit / Reject" | **T1** |
| Detection status label: `(not reviewed)`, `(accepted)`, `(rejected)`, `(edited)` | Shown inline but not in status bar | T2 |
| Detection type badge (upper-right overlay: `[TP]` / `[FP]` / `[FN]` in color) | — | **T1** |
| GT visibility toggle (`GT` checkbox) | — | **T1** |
| Prediction visibility toggle (`Pred` checkbox) | — | **T1** |
| Reviewed GT drawn with stipple fill (ghost appearance) | — | **T1** |
| Focused GT drawn in gold (`#FFD700`) with thicker stroke | — | **T1** |
| Class filter dropdown (review-local, populated from class_names) | **[done]** | — |
| Type filter (All / TP / FP / FN) | **[done]** | — |
| Status filter (All / Reviewed / Not Reviewed) | **[done]** | — |
| Filter-exhaustion auto-switch: when current-type filter has nothing unreviewed, cycle to next type before advancing image | — | **T1** |
| Filtered image list: only images with preds or annotations | All images in selection | T2 |
| First-switch zoom to first unreviewed detection | — | T2 |
| Accept on FP → switch to Annotate tab with pred_reference overlay | **[done]** | — |
| Edit → switch to Annotate tab preserving scale/offset/pred_reference | **[done]** | — |
| Confirm-dialog on return from Annotate (Save vs Redo) | — | **T1** |
| `backup_original_labels` once per session | Backend supports it but no UI trigger | T2 |
| Per-detection match bbox computed (union of GT + pred) | **[done]** | — |
| Reviewed halftone + focused-last-drawn order | — | **T1** |

## Class management

| yolo-annotator feature | tcip-web today | Priority |
|---|---|---|
| `classes.json` in state dir (id → {name, color}) | — | **T1** |
| Auto-assign high-contrast default colors per class ID | Hardcoded 4-color palette | **T1** |
| Custom dark color picker (SI palette + Basic palette + hex + system) | — | **T1** |
| `<New Class>` option in dropdown → input dialog | — | **T1** |
| Class counts in dropdown | — | **T1** |
| 0-9 keyboard shortcuts to select class | — | **T1** |
| Class auto-created when predictions reference an unknown class id | — | T2 |
| Editable class name (not just adding) | — | T3 |

## State persistence (`<dataset_root>/.tcip/state/` — or equivalent)

| yolo-annotator feature | tcip-web today | Priority |
|---|---|---|
| `classes.json` | — | **T1** |
| `annotation_stats.json`: sessions[], image_status{}, annotation_authors{} | — | T2 |
| `review_stats.json`: image{<name>: {status, detections[]}}, labels_backed_up | **[done]** (via ReviewEngine) | — |
| Per-image status ("complete" / "partial" / "unannotated") derived + persisted | — | **T1** |
| Per-annotation author metadata | `AnnotationEngine` holds authors; not persisted | T2 |
| Session timer aggregation (total time, avg time per annotation) | — | T2 |
| Auto-migration of legacy file names | — | T3 |

---

## Priority recap

**Tier 1 (must-have for Annotate+Review parity):**

1. **Class management**: classes.json persistence, auto-contrast colors, custom color picker dialog, `<New Class>` input dialog, counts in dropdown, 0-9 hotkeys, color swatch button.
2. **Editable image counter** + status filter + Complete checkbox + Visible checkbox in top bar.
3. **Polygon vertex editing**: drag, insert-on-edge, delete-on-right-click, double-click to close, right-click delete, hover highlighting, preview line.
4. **Snap toggle** (`s`) + visible snap indicator.
5. **Scale-dependent symbology**: discrete zoom levels, stroke width / label size that scales with zoom, halo text, active-class-only filter in box mode.
6. **Review button labels** context-aware (TP/FP/FN), detection-type badge overlay, GT/Pred visibility toggles, reviewed halftone + focused-gold styling, filter-exhaustion auto-switch, confirm-dialog on return from Annotate.
7. **Auto-save on image navigation.**

**Tier 2 (strong polish):**

- Streaming vertex mode (`v`).
- Per-image timer in status bar + session tracking (annotation_stats.json).
- Welcome state when no dataset loaded.
- Auto-zoom-to-first-unreviewed on review tab activation.
- First-time backup of labels (wire UI trigger).
- annotation_authors.json persistence.
- Filtered image list in Review (only images with preds or GT).
- Preview line + dashed preview in polygon drawing.

**Tier 3 (visual / branding):**

- SI logo + Archivo font + "YoloLabeler" branding.
- Spacebar = synthetic left-click.
- In-progress polygon vertex undo (one-vertex pops before full snapshot).
- Editable class name.
- Tab switch behaviours (sync scale, restart timer, etc.).

---

## Questions to batch for PD

1. **Branding**: keep yolo-annotator's "YoloLabeler" title in the browser tab, or use "TCIP"?
2. **SI logo + Archivo font**: ship as-is from `yolo-annotator/src/yololabeler/assets/`? The logo would need licensing/brand-safety confirmation for the web-deployed version.
3. **State directory**: yolo-annotator uses `<image_folder>/state/`. tcip-web uses `<project_root>/.tcip/state/`. Want me to make the annotate classes.json + annotation_stats.json live in `<project_root>/.tcip/state/` (one source of truth per project), or mirror yolo-annotator's per-image-folder convention?
4. **Session tracking**: how important is the annotation_stats.json session log with per-image timers in the Phase 1 timeline? It's breeder-visibility info (who annotated what, when) but nobody relies on it in Phase 1.
5. **"Complete" button semantics** — in yolo-annotator the Complete checkbox just toggles a status flag. Does the PD want any additional side-effect (e.g., lock annotations, show a green border)?
