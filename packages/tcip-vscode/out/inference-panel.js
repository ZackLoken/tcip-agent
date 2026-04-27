"use strict";
/**
 * Inference Panel — WebviewPanel host for running model inference.
 * Shows model selection, progress, and live result previews.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.InferencePanel = void 0;
const vscode = __importStar(require("vscode"));
const webview_base_1 = require("./webview-base");
class InferencePanel extends webview_base_1.WebviewPanelProvider {
    constructor(extensionUri) {
        super(extensionUri, "tcip-inference", "TCIP: Inference");
    }
    getHtmlContent(webview, nonce, sharedCssUri, sharedJsUri) {
        const inferenceJsUri = (0, webview_base_1.getUri)(webview, this.extensionUri, "media", "inference.js");
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
    .config-section {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-width: 500px;
    }
    .config-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .config-row label {
      min-width: 100px;
      font-size: 12px;
    }
    .config-row input, .config-row select {
      flex: 1;
    }
    .progress-section {
      padding: 16px;
      display: none;
    }
    .progress-info {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 12px;
    }
    .preview-section {
      padding: 8px;
    }
    .preview-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px;
      border-bottom: 1px solid var(--tcip-border);
    }
    .preview-item img {
      width: 80px;
      height: 80px;
      object-fit: cover;
      border-radius: 4px;
    }
    .preview-item .info {
      font-size: 12px;
    }
    .preview-item .info .filename {
      font-weight: 600;
    }
    .preview-item .info .det-count {
      color: var(--vscode-descriptionForeground);
    }
    .complete-banner {
      display: none;
      margin: 16px;
      padding: 16px;
      background: #2d5a30;
      border: 1px solid var(--tcip-success);
      border-radius: 6px;
      text-align: center;
    }
    .section-header {
      font-size: 12px;
      font-weight: 600;
      padding: 8px 12px 4px;
      color: var(--vscode-descriptionForeground);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  </style>
</head>
<body>
  <div class="panel-container">
    <div class="toolbar">
      <span style="font-weight:600">Inference</span>
      <div style="flex:1"></div>
      <span class="status-text" id="inference-status">Idle</span>
    </div>
    <div class="content">
      <div id="empty-state" class="empty-state">
        <div class="icon">▶</div>
        <div>No inference running</div>
        <div class="status-text">Use Copilot to run inference with a trained model</div>
      </div>

      <div id="progress-section" class="progress-section">
        <div class="progress-info">
          <span id="progress-label">Processing...</span>
          <span id="progress-count">0 / 0</span>
        </div>
        <div class="progress-track">
          <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
        </div>
      </div>

      <div id="complete-banner" class="complete-banner">
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px">✓ Inference Complete</div>
        <div id="complete-summary"></div>
        <div style="margin-top: 12px">
          <button id="btn-open-results" class="accent">Open Results</button>
          <button id="btn-open-dir">Open Output Directory</button>
        </div>
      </div>

      <div id="preview-section" style="display:none">
        <div class="section-header">Recent Results</div>
        <div id="preview-list" class="preview-section"></div>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${inferenceJsUri}"></script>
</body>
</html>`;
    }
    handleMessage(msg) {
        switch (msg.type) {
            case "open_output_dir": {
                const dirPath = msg.path;
                if (dirPath) {
                    vscode.commands.executeCommand("revealFileInOS", vscode.Uri.file(dirPath));
                }
                break;
            }
            case "open_results": {
                vscode.commands.executeCommand("tcip-agent.openResults");
                break;
            }
            case "request_image": {
                const filePath = msg.path;
                if (this.panel) {
                    const imageUri = this.panel.webview.asWebviewUri(vscode.Uri.file(filePath));
                    this.postMessage({ type: "image_uri", path: filePath, uri: imageUri.toString() });
                }
                break;
            }
        }
    }
}
exports.InferencePanel = InferencePanel;
//# sourceMappingURL=inference-panel.js.map