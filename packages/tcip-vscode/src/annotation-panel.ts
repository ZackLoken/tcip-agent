/**
 * Annotation Panel â€” WebviewPanel host for the interactive annotation canvas.
 * Uses Fabric.js for box/polygon drawing, class selection, undo/redo, zoom/pan.
 */

import * as vscode from "vscode";
import * as path from "path";
import { WebviewPanelProvider, getUri, getNonce } from "./webview-base";

export class AnnotationPanel extends WebviewPanelProvider {
  constructor(extensionUri: vscode.Uri) {
    super(extensionUri, "tcip-annotation", "TCIP: Annotation");
  }

  protected getHtmlContent(
    webview: vscode.Webview,
    nonce: string,
    sharedCssUri: vscode.Uri,
    sharedJsUri: vscode.Uri,
  ): string {
    const annotationJsUri = getUri(webview, this.extensionUri, "media", "annotation.js");
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
    #canvas-wrapper {
      flex: 1;
      position: relative;
      overflow: hidden;
      cursor: crosshair;
    }
    #annotation-canvas {
      position: absolute;
      top: 0; left: 0;
    }
    .toolbar .separator {
      width: 1px;
      height: 20px;
      background: var(--tcip-border);
    }
    .class-selector {
      min-width: 120px;
    }
    .zoom-display {
      min-width: 50px;
      text-align: center;
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
    }
    .status-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 4px 12px;
      background: var(--vscode-editorWidget-background);
      border-top: 1px solid var(--tcip-border);
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div class="panel-container">
    <div class="toolbar">
      <button id="btn-select" class="active" title="Select (V)">â–¨ Select</button>
      <button id="btn-box" title="Draw Box (B)">â˜ Box</button>
      <button id="btn-polygon" title="Draw Polygon (P)">â¬  Polygon</button>
      <div class="separator"></div>
      <select id="class-select" class="class-selector" title="Annotation class">
        <option value="0">Class 0</option>
      </select>
      <div class="separator"></div>
      <button id="btn-undo" title="Undo (Ctrl+Z)">â†© Undo</button>
      <button id="btn-redo" title="Redo (Ctrl+Shift+Z)">â†ª Redo</button>
      <div class="separator"></div>
      <button id="btn-delete" title="Delete selected (Del)">ðŸ—‘ Delete</button>
      <button id="btn-clear" title="Clear all">âœ• Clear</button>
      <div class="separator"></div>
      <button id="btn-zoom-in" title="Zoom in">+</button>
      <span class="zoom-display" id="zoom-display">100%</span>
      <button id="btn-zoom-out" title="Zoom out">âˆ’</button>
      <button id="btn-zoom-fit" title="Fit to view">â¬œ Fit</button>
      <div style="flex:1"></div>
      <button id="btn-save" class="accent" title="Save annotations (Ctrl+S)">ðŸ’¾ Save</button>
    </div>
    <div id="canvas-wrapper" class="content">
      <canvas id="annotation-canvas"></canvas>
      <div id="empty-state" class="empty-state">
        <div class="icon">ðŸ–¼</div>
        <div>No image loaded</div>
        <div class="status-text">Use Copilot to load an image, or open one from the Dataset browser</div>
      </div>
    </div>
    <div class="status-bar">
      <span id="status-file">No file</span>
      <span id="status-annotations">0 annotations</span>
      <span id="status-mode">Select</span>
      <span id="status-pos">0, 0</span>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${annotationJsUri}"></script>
</body>
</html>`;
  }

  /** Current image path (used to derive label paths) */
  private currentImagePath: string | undefined;

  /**
   * Load an image into the canvas, converting the file path to a webview URI.
   * Optionally sets class labels and auto-loads existing annotations.
   */
  loadImage(imagePath: string, labels?: string[]): void {
    this.currentImagePath = imagePath;
    if (!this.panel) { return; }
    const imageUri = this.panel.webview.asWebviewUri(vscode.Uri.file(imagePath));
    this.postMessage({ type: "loadImage", uri: imageUri.toString() });
    if (labels && labels.length > 0) {
      this.postMessage({ type: "setClasses", classes: labels });
    }
    // Auto-load existing segment labels if available
    const imgName = imagePath.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "");
    const imgDir = imagePath.replace(/[\\/][^\\/]+$/, "");
    const labelDir = imgDir.replace(/[\\/]images([\\/]|$)/, `${path.sep}labels${path.sep}segment$1`);
    const labelPath = path.join(labelDir, imgName + ".txt");
    const labelUri = vscode.Uri.file(labelPath);
    vscode.workspace.fs.readFile(labelUri).then(
      (data) => this.postMessage({ type: "loadLabels", text: new TextDecoder().decode(data), format: "segment" }),
      () => { /* no labels file — that's fine */ },
    );
  }

  /** Clear all annotations from the canvas. */
  clearCanvas(): void {
    this.postMessage({ type: "clearAnnotations" });
  }

  /** Highlight specific annotation indices on the canvas. */
  highlightAnnotations(indices: number[]): void {
    this.postMessage({ type: "highlight", indices });
  }

  protected handleMessage(msg: { type: string; [key: string]: unknown }): void {
    switch (msg.type) {
      // â”€â”€ Legacy save (keep for backward compat) â”€â”€
      case "save_annotations": {
        const filePath = msg.path as string;
        const content = msg.content as string;
        const uri = vscode.Uri.file(filePath);
        const encoder = new TextEncoder();
        vscode.workspace.fs.writeFile(uri, encoder.encode(content)).then(
          () => this.postMessage({ type: "save_success", path: filePath }),
          (err) => this.postMessage({ type: "save_error", path: filePath, error: String(err) }),
        );
        break;
      }

      // â”€â”€ Dual-output save from rewritten canvas â”€â”€
      case "save_detect_annotations":
      case "save_segment_annotations": {
        if (!this.currentImagePath) {
          this.postMessage({ type: "save_error", error: "No image loaded" });
          break;
        }
        const lines = msg.lines as string[];
        const content = lines.join("\n") + (lines.length > 0 ? "\n" : "");
        const subdir = msg.type === "save_detect_annotations" ? "detect" : "segment";
        const imgName = this.currentImagePath.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "");
        // Derive label dir from image path: data/images/X.jpg â†’ data/labels/{subdir}/X.txt
        const imgDir = this.currentImagePath.replace(/[\\/][^\\/]+$/, "");
        const labelDir = imgDir.replace(/[\\/]images([\\/]|$)/, `${path.sep}labels${path.sep}${subdir}$1`);
        const labelPath = path.join(labelDir, imgName + ".txt");
        const uri = vscode.Uri.file(labelPath);
        const encoder = new TextEncoder();
        vscode.workspace.fs.createDirectory(vscode.Uri.file(labelDir)).then(
          () => vscode.workspace.fs.writeFile(uri, encoder.encode(content)),
          () => vscode.workspace.fs.writeFile(uri, encoder.encode(content)),
        ).then(
          () => this.postMessage({ type: "save_success", path: labelPath }),
          (err) => this.postMessage({ type: "save_error", path: labelPath, error: String(err) }),
        );
        break;
      }

      case "read_labels": {
        if (!this.currentImagePath) break;
        const format = (msg.format as string) || "segment";
        const imgName = this.currentImagePath.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "");
        const imgDir = this.currentImagePath.replace(/[\\/][^\\/]+$/, "");
        const labelDir = imgDir.replace(/[\\/]images([\\/]|$)/, `${path.sep}labels${path.sep}${format}$1`);
        const labelPath = path.join(labelDir, imgName + ".txt");
        const uri = vscode.Uri.file(labelPath);
        vscode.workspace.fs.readFile(uri).then(
          (data) => this.postMessage({ type: "labels_data", text: new TextDecoder().decode(data), format }),
          () => this.postMessage({ type: "labels_data", text: "", format }),
        );
        break;
      }

      case "request_image": {
        // Convert local file path to webview-safe URI
        const filePath = msg.path as string;
        this.currentImagePath = filePath;
        if (this.panel) {
          const imageUri = this.panel.webview.asWebviewUri(vscode.Uri.file(filePath));
          this.postMessage({ type: "image_uri", path: filePath, uri: imageUri.toString() });
        }
        break;
      }

      case "sam_request": {
        if (!this.currentImagePath) {
          this.postMessage({ type: "sam_result", error: "No image loaded" });
          break;
        }
        this.handleSamRequest(msg);
        break;
      }
    }
  }

  private async handleSamRequest(msg: { type: string; [key: string]: unknown }): Promise<void> {
    const imagePath = this.currentImagePath;
    if (!imagePath) {
      this.postMessage({ type: "sam_result", error: "No image loaded" });
      return;
    }

    try {
      // Invoke the SAM MCP tool via terminal/python subprocess
      const requestData = msg as { type: string; [key: string]: unknown };
      let script: string;

      if (requestData.type === "box" || (requestData as Record<string,unknown>).box) {
        const box = (requestData as Record<string,unknown>).box as { x1: number; y1: number; x2: number; y2: number };
        script = `from tcip_annotation.sam_wrapper import predict_from_box; import json; poly = predict_from_box(${JSON.stringify(imagePath)}, ${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}); print(json.dumps([{"x": p[0], "y": p[1]} for p in poly]))`;
      } else {
        const points = (requestData as Record<string,unknown>).points as Array<{ x: number; y: number; label: number }>;
        if (!points || points.length === 0) {
          this.postMessage({ type: "sam_result", error: "No prompts provided" });
          return;
        }
        if (points.length === 1) {
          const p = points[0];
          script = `from tcip_annotation.sam_wrapper import predict_from_point; import json; poly = predict_from_point(${JSON.stringify(imagePath)}, ${p.x}, ${p.y}, label=${p.label}); print(json.dumps([{"x": p[0], "y": p[1]} for p in poly]))`;
        } else {
          const pts = JSON.stringify(points.map(p => [p.x, p.y]));
          const lbls = JSON.stringify(points.map(p => p.label));
          script = `from tcip_annotation.sam_wrapper import predict_from_points; import json; poly = predict_from_points(${JSON.stringify(imagePath)}, ${pts}, ${lbls}); print(json.dumps([{"x": p[0], "y": p[1]} for p in poly]))`;
        }
      }

      // Run Python with the SAM wrapper
      const { exec } = require("child_process") as typeof import("child_process");
      const result = await new Promise<string>((resolve, reject) => {
        exec(
          `python -c "${script.replace(/"/g, '\\"')}"`,
          { timeout: 60000 },
          (error: Error | null, stdout: string, stderr: string) => {
            if (error) { reject(new Error(stderr || error.message)); }
            else { resolve(stdout.trim()); }
          },
        );
      });

      const polygon = JSON.parse(result) as Array<{ x: number; y: number }>;
      this.postMessage({ type: "sam_result", polygon });
    } catch (err) {
      this.postMessage({ type: "sam_result", error: String(err) });
    }
  }
}
