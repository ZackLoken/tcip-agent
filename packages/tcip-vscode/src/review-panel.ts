/**
 * Review Panel — WebviewPanel host for reviewing predictions against ground truth.
 * Displays TP/FP/FN overlays with IoU/confidence filtering.
 */

import * as vscode from "vscode";
import * as path from "path";
import * as cp from "child_process";
import { WebviewPanelProvider, getUri, getNonce } from "./webview-base";

export class ReviewPanel extends WebviewPanelProvider {
  constructor(extensionUri: vscode.Uri) {
    super(extensionUri, "tcip-review", "TCIP: Review");
  }

  protected getHtmlContent(
    webview: vscode.Webview,
    nonce: string,
    sharedCssUri: vscode.Uri,
    sharedJsUri: vscode.Uri,
  ): string {
    const reviewJsUri = getUri(webview, this.extensionUri, "media", "review.js");
    const cspSource = webview.cspSource;

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; img-src ${cspSource} data: file:; style-src ${cspSource} 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${sharedCssUri}">
  <style nonce="${nonce}">
    #review-canvas-wrapper {
      flex: 1;
      position: relative;
      overflow: hidden;
    }
    #review-canvas {
      position: absolute;
      top: 0; left: 0;
    }
    .filter-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 6px 12px;
      background: var(--vscode-editorWidget-background);
      border-bottom: 1px solid var(--tcip-border);
      flex-shrink: 0;
    }
    .filter-bar label {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
    }
    .filter-bar .value {
      min-width: 32px;
      text-align: right;
      font-size: 11px;
    }
    .legend {
      display: flex;
      gap: 12px;
    }
    .legend span::before {
      content: '';
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 4px;
    }
    .legend .tp::before { background: #4ec96e; }
    .legend .fp::before { background: #c75050; }
    .legend .fn::before { background: #5090c7; }
    .nav-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 12px;
      background: var(--vscode-editorWidget-background);
      border-top: 1px solid var(--tcip-border);
      flex-shrink: 0;
    }
    .summary-badges {
      display: flex;
      gap: 6px;
    }
  </style>
</head>
<body>
  <div class="panel-container">
    <div class="toolbar">
      <span style="font-weight:600">Review</span>
      <div style="flex:1"></div>
      <div class="legend">
        <span class="tp">TP</span>
        <span class="fp">FP</span>
        <span class="fn">FN</span>
      </div>
    </div>
    <div class="filter-bar">
      <label>IoU ≥</label>
      <input type="range" id="iou-slider" min="0" max="100" value="50" step="5">
      <span class="value" id="iou-value">0.50</span>
      <label>Conf ≥</label>
      <input type="range" id="conf-slider" min="0" max="100" value="25" step="5">
      <span class="value" id="conf-value">0.25</span>
      <div style="flex:1"></div>
      <div class="summary-badges">
        <span class="badge badge-success" id="tp-count">TP: 0</span>
        <span class="badge badge-danger" id="fp-count">FP: 0</span>
        <span class="badge badge-info" id="fn-count">FN: 0</span>
      </div>
    </div>
    <div id="review-canvas-wrapper" class="content">
      <canvas id="review-canvas"></canvas>
      <div id="empty-state" class="empty-state">
        <div class="icon">🔍</div>
        <div>No predictions loaded</div>
        <div class="status-text">Use Copilot to load predictions for review</div>
      </div>
    </div>
    <div class="nav-bar">
      <button id="btn-prev-img" title="Previous image">◀ Prev Image</button>
      <span id="img-counter">0 / 0</span>
      <button id="btn-next-img" title="Next image">Next Image ▶</button>
      <div class="separator" style="width:1px;height:20px;background:var(--tcip-border)"></div>
      <button id="btn-prev-det" title="Previous detection">◀ Prev</button>
      <span id="det-counter">Det 0 / 0</span>
      <button id="btn-next-det" title="Next detection">Next ▶</button>
      <div style="flex:1"></div>
      <button id="btn-accept" class="accent">✓ Accept</button>
      <button id="btn-edit">✎ Edit</button>
      <button id="btn-reject" class="danger">✕ Reject</button>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${reviewJsUri}"></script>
</body>
</html>`;
  }

  /**
   * Show predictions for an image, converting the file path to a webview URI.
   * Sends the image and prediction data to the review webview.
   */
  showPredictions(imagePath: string, predictions: unknown[]): void {
    if (!this.panel) { return; }
    const imageUri = this.panel.webview.asWebviewUri(vscode.Uri.file(imagePath));
    this.postMessage({ type: "image_uri", path: imagePath, uri: imageUri.toString() });
    this.postMessage({ type: "load_predictions", annotations: predictions });
  }

  protected handleMessage(msg: { type: string; [key: string]: unknown }): void {
    switch (msg.type) {
      case "review_complete": {
        // Forward to agent or log
        break;
      }
      case "open_in_annotation": {
        vscode.commands.executeCommand("tcip-agent.openInAnnotation", msg.path as string);
        break;
      }
      case "request_image": {
        const filePath = msg.path as string;
        if (this.panel) {
          const imageUri = this.panel.webview.asWebviewUri(vscode.Uri.file(filePath));
          this.postMessage({ type: "image_uri", path: filePath, uri: imageUri.toString() });
        }
        break;
      }
      case "request_matches": {
        this.handleRequestMatches(msg);
        break;
      }
      case "save_review_state": {
        this.handleSaveReviewState(msg);
        break;
      }
      case "delete_gt": {
        this.handleDeleteGt(msg);
        break;
      }
    }
  }

  /**
   * Invoke the Python matching engine and return TP/FP/FN detection list to webview.
   * Builds a flat array of {tag, classId, box, confidence, type, polygon?} for the webview.
   */
  private handleRequestMatches(msg: { [key: string]: unknown }): void {
    const imagePath = msg.path as string;
    const iouThreshold = (msg.iouThreshold as number) ?? 0.5;
    const confThreshold = (msg.confThreshold as number) ?? 0.25;

    const script = `
import json, sys
from tcip_annotation import (
    parse_detect_labels, parse_segment_labels,
    parse_detect_predictions, parse_segment_predictions,
    compute_matches
)
from pathlib import Path

img = Path(r"${imagePath.replace(/\\/g, "\\\\")}")
root = img.parent.parent
stem = img.stem
img_w, img_h = 1, 1  # normalised; webview will scale
try:
    from PIL import Image as PILImage
    with PILImage.open(str(img)) as im:
        img_w, img_h = im.size
except Exception:
    pass

gt_det_path = str(root / "labels" / "detect" / (stem + ".txt"))
gt_seg_path = str(root / "labels" / "segment" / (stem + ".txt"))
pred_det_path = str(root / "predictions" / "detect" / (stem + ".txt"))
pred_seg_path = str(root / "predictions" / "segment" / (stem + ".txt"))

gt_boxes, _ = parse_detect_labels(gt_det_path, img_w, img_h)
gt_polys, _ = parse_segment_labels(gt_seg_path, img_w, img_h)
pred_boxes, _ = parse_detect_predictions(pred_det_path, img_w, img_h)
pred_polys, _ = parse_segment_predictions(pred_seg_path, img_w, img_h)

result = compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys,
                         ${iouThreshold}, ${confThreshold})

detections = []
for m in result["tp"]:
    gt = gt_boxes[m["gt_idx"]] if m["gt_type"] == "box" else gt_polys[m["gt_idx"]]
    pred = pred_boxes[m["pred_idx"]] if m["pred_type"] == "box" else pred_polys[m["pred_idx"]]
    d = {"tag": "tp", "classId": m["class_id"], "iou": m["iou"], "confidence": m["conf"]}
    if m["pred_type"] == "box":
        d["box"] = [pred.x1/img_w, pred.y1/img_h, (pred.x2-pred.x1)/img_w, (pred.y2-pred.y1)/img_h]
    else:
        d["polygon"] = [[p[0]/img_w, p[1]/img_h] for p in pred.points]
        xs = [p[0] for p in pred.points]; ys = [p[1] for p in pred.points]
        d["box"] = [min(xs)/img_w, min(ys)/img_h, (max(xs)-min(xs))/img_w, (max(ys)-min(ys))/img_h]
    detections.append(d)

for m in result["fp"]:
    pred = pred_boxes[m["pred_idx"]] if m["pred_type"] == "box" else pred_polys[m["pred_idx"]]
    d = {"tag": "fp", "classId": m["class_id"], "confidence": m["conf"]}
    if m["pred_type"] == "box":
        d["box"] = [pred.x1/img_w, pred.y1/img_h, (pred.x2-pred.x1)/img_w, (pred.y2-pred.y1)/img_h]
    else:
        d["polygon"] = [[p[0]/img_w, p[1]/img_h] for p in pred.points]
        xs = [p[0] for p in pred.points]; ys = [p[1] for p in pred.points]
        d["box"] = [min(xs)/img_w, min(ys)/img_h, (max(xs)-min(xs))/img_w, (max(ys)-min(ys))/img_h]
    detections.append(d)

for m in result["fn"]:
    gt = gt_boxes[m["gt_idx"]] if m["gt_type"] == "box" else gt_polys[m["gt_idx"]]
    d = {"tag": "fn", "classId": m["class_id"], "confidence": 0}
    if m["gt_type"] == "box":
        d["box"] = [gt.x1/img_w, gt.y1/img_h, (gt.x2-gt.x1)/img_w, (gt.y2-gt.y1)/img_h]
    else:
        d["polygon"] = [[p[0]/img_w, p[1]/img_h] for p in gt.points]
        xs = [p[0] for p in gt.points]; ys = [p[1] for p in gt.points]
        d["box"] = [min(xs)/img_w, min(ys)/img_h, (max(xs)-min(xs))/img_w, (max(ys)-min(ys))/img_h]
    detections.append(d)

print(json.dumps({"detections": detections, "imgW": img_w, "imgH": img_h}))
`;
    cp.exec(`python -c "${script.replace(/"/g, '\\"').replace(/\n/g, "\\n")}"`, { timeout: 30000 }, (err, stdout) => {
      if (err) {
        this.postMessage({ type: "load_matches", matches: [], error: err.message });
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim());
        this.postMessage({ type: "load_matches", matches: parsed.detections ?? [], imgW: parsed.imgW, imgH: parsed.imgH });
      } catch {
        this.postMessage({ type: "load_matches", matches: [], error: "Failed to parse matching results" });
      }
    });
  }

  /**
   * Save review state (per-image decisions) to review_stats.json next to the images.
   */
  private handleSaveReviewState(msg: { [key: string]: unknown }): void {
    const imagePath = msg.imagePath as string;
    const state = msg.state as Record<string, unknown>;
    if (!imagePath || !state) { return; }
    const dir = path.dirname(path.dirname(imagePath));
    const statsPath = path.join(dir, "review_stats.json");
    const uri = vscode.Uri.file(statsPath);

    // Read existing, merge, write
    vscode.workspace.fs.readFile(uri).then(
      (data) => {
        try {
          const existing = JSON.parse(Buffer.from(data).toString("utf-8")) as Record<string, unknown>;
          Object.assign(existing, state);
          return vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(existing, null, 2), "utf-8"));
        } catch {
          return vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(state, null, 2), "utf-8"));
        }
      },
      () => vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(state, null, 2), "utf-8")),
    );
  }

  /**
   * Delete a ground-truth annotation from its label file by index.
   */
  private handleDeleteGt(msg: { [key: string]: unknown }): void {
    const imagePath = msg.imagePath as string;
    const lineIndex = msg.lineIndex as number;
    const labelType = (msg.labelType as string) ?? "detect";
    if (!imagePath || lineIndex == null) { return; }

    const dir = path.dirname(path.dirname(imagePath));
    const stem = path.basename(imagePath, path.extname(imagePath));
    const labelPath = path.join(dir, "labels", labelType, stem + ".txt");
    const uri = vscode.Uri.file(labelPath);

    vscode.workspace.fs.readFile(uri).then((data) => {
      const lines = Buffer.from(data).toString("utf-8").split("\n");
      if (lineIndex >= 0 && lineIndex < lines.length) {
        lines.splice(lineIndex, 1);
        return vscode.workspace.fs.writeFile(uri, Buffer.from(lines.join("\n"), "utf-8"));
      }
    });
  }
}
