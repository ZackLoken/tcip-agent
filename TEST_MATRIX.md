# TCIP-Agent GUI Smoke Test Matrix

> Manual smoke test checklist. Mark ✅ pass, ❌ fail, ⚠️ partial, ⏭️ skip.
> Date: ________  Tester: ________

---

## 1. Application Lifecycle

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 1.1 | Launch app | `python -m tcip_gui` | Window opens, status bar shows "Connecting…" then connected | |
| 1.2 | Offline mode (no API key) | Unset ANTHROPIC_API_KEY, launch | EchoClient fallback, status shows "Offline" | |
| 1.3 | Agent crash recovery | Kill Rust agent process externally | Status bar shows reconnecting, agent restarts (max 5 attempts) | |
| 1.4 | Restart cap | Kill agent 6+ times rapidly | After 5th restart shows "Agent failed" error, stops retrying | |
| 1.5 | Window close | Close window | Agent process cleaned up, no orphans in task manager | |
| 1.6 | Tab switching | Click each tab: Annotate, Review, Training, Results | Each panel renders without errors | |

## 2. Chat Overlay (Ctrl+/)

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 2.1 | Toggle open | Press Ctrl+/ | Chat overlay slides in from right | |
| 2.2 | Toggle close | Press Ctrl+/ again or Esc | Overlay hides | |
| 2.3 | Minimize | Click minimize button | Overlay collapses to small bar | |
| 2.4 | Restore | Click minimized bar | Overlay expands back | |
| 2.5 | Send message | Type text, press Enter or click Send | User bubble appears, agent response streams in | |
| 2.6 | Streaming display | Send a message | Response text appears token-by-token | |
| 2.7 | Tool card | Trigger tool use (e.g., "list my datasets") | ToolCard appears with tool name, expandable detail | |
| 2.8 | Tool card error | Trigger tool error | ToolCard auto-expands, shows error in red | |
| 2.9 | Permission card | Trigger action needing permission | PermissionCard shows with Allow/Deny buttons | |
| 2.10 | Permission allow | Click Allow on permission card | Tool executes, result shown | |
| 2.11 | Permission deny | Click Deny on permission card | Tool cancelled, message shown | |
| 2.12 | Error bubble | Force an API error | Error bubble appears with message | |
| 2.13 | Scroll behavior | Send many messages | Auto-scrolls to bottom, manual scroll stops auto-scroll | |

## 3. Annotation Panel

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 3.1 | Load dataset | Open a project with images | Thumbnail grid populates, first image on canvas | |
| 3.2 | Thumbnail click | Click a thumbnail | Image loads on canvas with existing annotations | |
| 3.3 | Select mode | Click Select tool or default | Can click/drag existing boxes and polygons | |
| 3.4 | Box draw mode | Click Box tool | Draw rectangle on canvas by click-drag | |
| 3.5 | Box resize | Select box, drag handles | Box resizes correctly | |
| 3.6 | Box move | Select box, drag center | Box moves, stays in bounds | |
| 3.7 | Polygon draw mode | Click Polygon tool | Click points to create polygon, double-click/Enter closes | |
| 3.8 | Polygon edit | Select polygon, drag vertices | Vertices move correctly | |
| 3.9 | Delete annotation | Select annotation, press Delete | Annotation removed from canvas | |
| 3.10 | Undo | Press Ctrl+Z after delete/draw | Last action undone | |
| 3.11 | Redo | Press Ctrl+Y after undo | Action re-applied | |
| 3.12 | Class selector | Open class dropdown | Shows classes from classes.txt / data.yaml | |
| 3.13 | Change class | Select annotation, pick new class | Annotation color/label updates | |
| 3.14 | Confidence slider | Drag confidence slider | Predictions below threshold hide, above threshold show | |
| 3.15 | Confidence label | Move slider | Label updates "Conf ≥ 0.XX" | |
| 3.16 | Save annotations | Click Save or Ctrl+S | YOLO-format .txt file written, status confirms | |
| 3.17 | Navigate Left/Right | Press Left/Right arrow keys | Previous/next image loads | |
| 3.18 | Canvas zoom | Mouse wheel on canvas | Image zooms in/out | |
| 3.19 | Canvas pan | Middle-click drag or Ctrl+drag | Canvas pans | |

## 4. Review Panel

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 4.1 | Load predictions | Open project with predictions/ dir | Review panel populates with prediction images | |
| 4.2 | Accept prediction | Click Accept on a prediction | Annotation saved, moves to next | |
| 4.3 | Reject prediction | Click Reject | Prediction discarded, moves to next | |
| 4.4 | Edit prediction | Click Edit | Switches to annotation panel with prediction loaded | |
| 4.5 | Filter by class | Select class filter | Only predictions of that class shown | |
| 4.6 | Confidence filter | Adjust confidence slider | Below-threshold predictions hidden | |
| 4.7 | IoU slider | Adjust IoU slider (if present) | Filters overlapping predictions | |
| 4.8 | Navigation | Left/Right arrows | Cycles through filtered predictions | |

## 5. Dataset Browser

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 5.1 | Load directory | Select a dataset directory | Thumbnails load, counts shown | |
| 5.2 | Text search | Type in search box | Filters to matching filenames | |
| 5.3 | Status filter | Select "Annotated" | Only annotated images shown | |
| 5.4 | Status filter 2 | Select "Unannotated" | Only unannotated images shown | |
| 5.5 | Crop filter | Select a crop name | Only images with that crop prefix shown | |
| 5.6 | Date filter | Select a date | Only images from that date shown | |
| 5.7 | Sort annotated first | Select sort: Annotated First | Annotated images appear at top | |
| 5.8 | Sort unannotated first | Select sort: Unannotated First | Unannotated images appear at top | |
| 5.9 | Combined filters | Set search + crop + status | Intersection of all filters applied | |
| 5.10 | Clear filters | Clear all filter fields | All images shown again | |
| 5.11 | Thumbnail click | Click thumbnail in browser | Image opens in annotation panel | |
| 5.12 | Empty directory | Point to empty dir | Graceful empty state, no crash | |
| 5.13 | Nonexistent directory | Point to missing path | No crash, appropriate message | |

## 6. Training Dashboard

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 6.1 | Start training | Trigger training via chat | Dashboard tab activates, progress bar appears | |
| 6.2 | Progress updates | During training | Epoch counter increments, progress bar fills | |
| 6.3 | Metrics display | During/after training | Loss, mAP, precision, recall values update | |
| 6.4 | Pause training | Click Pause button | Training pauses, button changes to Resume | |
| 6.5 | Resume training | Click Resume | Training continues from where it paused | |
| 6.6 | Stop training | Click Stop button | Training stops, final metrics shown | |
| 6.7 | TensorBoard embed | If TensorBoard URL available | WebEngineView loads TensorBoard UI | |
| 6.8 | Ray Dashboard embed | If Ray URL available | WebEngineView loads Ray dashboard | |
| 6.9 | HPO trial table | During HPO run | Trial table populates with trials, status, metrics | |
| 6.10 | Trial table sort | Click column headers | Rows sort by that column | |

## 7. Results Panel

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 7.1 | Overall metrics | After training completes | Shows mAP, precision, recall, F1 | |
| 7.2 | Per-class table | After training | Table with per-class AP, precision, recall | |
| 7.3 | Worst predictions | After training | Grid of worst prediction images | |
| 7.4 | CSV preview | Click CSV preview | Table shows CSV data with sortable columns | |
| 7.5 | Export CSV | Click Export CSV button | File save dialog, CSV written | |
| 7.6 | Export model | Click Export Model button | File save dialog, model file saved | |
| 7.7 | Run inference | Click Run Inference | Inference starts on selected dataset | |

## 8. Status Bar

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 8.1 | Connected state | Agent running | Green indicator, "Connected" text | |
| 8.2 | Disconnected state | Agent stopped | Red/gray indicator, "Disconnected" | |
| 8.3 | Reconnecting state | Agent restarting | Yellow indicator, "Reconnecting…" | |
| 8.4 | Token usage | After API calls | Token count updates | |
| 8.5 | Model display | When connected | Shows current model name | |

## 9. Project Import/Export

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 9.1 | Export project | Via chat: "export this project" | ZIP file created with data/, config, classes | |
| 9.2 | Export with models | Export with include_models=true | ZIP includes .pt model files | |
| 9.3 | Import project | Via chat: "import project from X.zip" | Project extracted to destination, files intact | |
| 9.4 | Import zip-slip safety | Craft ZIP with ../paths (dev test) | Extraction blocked, error returned | |

## 10. Protocol & Cross-Cutting

| # | Test | Steps | Expected | Result |
|---|------|-------|----------|--------|
| 10.1 | Multiple tabs active | Switch between tabs during training | No state corruption, each tab independent | |
| 10.2 | Rapid tab switching | Click tabs quickly | No crashes or rendering glitches | |
| 10.3 | Large image | Open a very large image (>4000px) | Canvas handles it, zoom works | |
| 10.4 | Many annotations | Image with 100+ boxes | Canvas renders all, no major lag | |
| 10.5 | Long chat history | Send 50+ messages | Scroll works, no memory bloat | |
| 10.6 | Agent message types | Various agent interactions | All 22+ message types handled without unknown-type errors | |
| 10.7 | Keyboard shortcuts | Test all: Ctrl+/, Esc, Delete, Ctrl+Z/Y, C, Left/Right, Enter | Each shortcut works in correct context | |

---

## Summary

| Section | Total Tests | Pass | Fail | Partial | Skip |
|---------|------------|------|------|---------|------|
| 1. App Lifecycle | 6 | | | | |
| 2. Chat Overlay | 13 | | | | |
| 3. Annotation | 19 | | | | |
| 4. Review | 8 | | | | |
| 5. Dataset Browser | 13 | | | | |
| 6. Training Dashboard | 10 | | | | |
| 7. Results Panel | 7 | | | | |
| 8. Status Bar | 5 | | | | |
| 9. Import/Export | 4 | | | | |
| 10. Cross-Cutting | 7 | | | | |
| **TOTAL** | **92** | | | | |
