# Panel postMessage Protocol

All TCIP webview panels communicate with the host (VS Code extension or
standalone bridge) exclusively via `window.postMessage` / `message` events.
The `vscode.acquireVsCodeApi()` call is isolated in **shared.js**; the
individual panel files never reference it directly.

## Shared API (`shared.js`)

| Function | Purpose |
|----------|---------|
| `postToHost(type, data?)` | Send `{type, ...data}` to host |
| `onHostMessage(handler)` | Register `window.message` listener |
| `saveState(key, value)` | Persist key in webview state |
| `loadState(key, default?)` | Read persisted key |
| `formatNumber(n)` | Locale-aware number formatting |
| `clamp(v, min, max)` | Numeric clamp |

---

## Annotation Panel (`annotation.js`)

### Host → Panel (incoming)

| `type` | Payload | Description |
|--------|---------|-------------|
| `loadImage` | `{uri}` | Display image (base64 data URI or URL) |
| `setClasses` | `{classes: string[]}` | Set class list for class selector |
| `loadLabels` | `{detect, segment}` | Load GT labels (YOLO text content) |
| `loadPredictions` | `{detect, segment}` | Load prediction labels |
| `clearAnnotations` | — | Clear all annotations from canvas |
| `sam_result` | `{mask, polygon}` | SAM segmentation result |

### Panel → Host (outgoing)

| `type` | Payload | Description |
|--------|---------|-------------|
| `sam_request` | `{type, point?, box?}` | Request SAM prediction (point or box prompt) |
| `save_detect_annotations` | `{content}` | Save detection labels (YOLO text) |
| `save_segment_annotations` | `{content}` | Save segmentation labels (YOLO text) |

---

## Review Panel (`review.js`)

### Host → Panel

| `type` | Payload | Description |
|--------|---------|-------------|
| `image_uri` | `{uri}` | Display image for review |
| `load_matches` | `{detections}` | Load TP/FP/FN match data |
| `load_gt` | `{detect}` | Load GT labels |
| `load_predictions` | `{detect}` | Load prediction labels |
| `set_image_queue` | `{queue: [{imagePath, status}]}` | Load image queue for review |
| `review_state_loaded` | `{state}` | Restore review state |
| `setClasses` | `{classes: string[]}` | Set class list |

### Panel → Host

| `type` | Payload | Description |
|--------|---------|-------------|
| `request_image` | `{path}` | Request image data-URI |
| `request_matches` | `{path, iouThreshold, confThreshold}` | Request match data |
| `save_review_state` | `{state}` | Persist review state |
| `delete_gt` | `{gtIdx, tag}` | Delete a GT annotation |
| `open_in_annotation` | `{path}` | Open image in annotation panel |

---

## Training Panel (`training.js`)

### Host → Panel

| `type` | Payload | Description |
|--------|---------|-------------|
| `training_started` | `{runId, config, tensorboardUrl?}` | Training run started |
| `metrics_update` | `{epoch, metrics: {key: number}}` | Per-epoch metrics (any keys) |
| `training_complete` | `{runId, bestEpoch, metrics}` | Training finished |

### Panel → Host

*None — training panel is read-only.*

---

## HPO Panel (`hpo.js`)

### Host → Panel

| `type` | Payload | Description |
|--------|---------|-------------|
| `hpo_started` | `{experimentId, searchSpace, rayDashboardUrl?}` | HPO started |
| `trial_update` | `{trialId, status, params, metrics, _best?}` | Trial progress |
| `hpo_complete` | `{experimentId, bestTrialId, bestParams, bestMetrics}` | HPO finished |

### Panel → Host

| `type` | Payload | Description |
|--------|---------|-------------|
| `stop_hpo` | — | Request HPO stop |

---

## Inference Panel (`inference.js`)

### Host → Panel

| `type` | Payload | Description |
|--------|---------|-------------|
| `inference_started` | `{modelPath, totalImages}` | Inference started |
| `inference_progress` | `{current, total, imagePath, detections}` | Per-image progress |
| `inference_complete` | `{totalImages, totalDetections, outputDir}` | Inference finished |

### Panel → Host

| `type` | Payload | Description |
|--------|---------|-------------|
| `open_results` | — | Open results panel |
| `open_output_dir` | `{path}` | Open output directory |

---

## Results Panel (`results.js`)

### Host → Panel

| `type` | Payload | Description |
|--------|---------|-------------|
| `results_show` | `{metrics, perClass, csvPath?}` | Show evaluation results |

### Panel → Host

| `type` | Payload | Description |
|--------|---------|-------------|
| `accept_model` | — | Accept current model |
| `retrain` | — | Request retraining |
| `export_csv` | — | Export results as CSV |

---

## Portability Notes

- All six panel JS files depend only on `postToHost()` and `onHostMessage()`
  from `shared.js`, plus standard DOM APIs and Chart.js (training panel only).
- `acquireVsCodeApi()` is called once in `shared.js`. A standalone bridge
  replaces this with a mock that routes messages via `window.postMessage`.
- State persistence (`saveState`/`loadState`) uses `vscode.getState()`/
  `setState()` in VS Code; the mock uses `sessionStorage`.
- No VS Code-specific CSS classes are used. Panels use their own styles
  injected by the TypeScript panel providers.
