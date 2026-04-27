"use strict";
/**
 * Results Panel — WebviewPanel host for displaying evaluation results.
 * Shows overall + per-class metrics, worst predictions, CSV preview.
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
exports.ResultsPanel = void 0;
const vscode = __importStar(require("vscode"));
const webview_base_1 = require("./webview-base");
class ResultsPanel extends webview_base_1.WebviewPanelProvider {
    constructor(extensionUri) {
        super(extensionUri, "tcip-results", "TCIP: Results");
    }
    getHtmlContent(webview, nonce, sharedCssUri, sharedJsUri) {
        const resultsJsUri = (0, webview_base_1.getUri)(webview, this.extensionUri, "media", "results.js");
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
    .metrics-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      padding: 16px;
    }
    .metric-card {
      background: var(--vscode-editorWidget-background);
      border: 1px solid var(--tcip-border);
      border-radius: 6px;
      padding: 12px;
      text-align: center;
    }
    .metric-card .label {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      margin-bottom: 4px;
    }
    .metric-card .value {
      font-size: 28px;
      font-weight: 700;
    }
    .metric-card .value.good { color: var(--tcip-success); }
    .metric-card .value.warn { color: var(--tcip-warning); }
    .metric-card .value.bad  { color: var(--tcip-danger); }
    .per-class-section {
      padding: 0 16px 16px;
      overflow: auto;
    }
    .per-class-section table {
      max-width: 800px;
    }
    .worst-section {
      padding: 0 16px 16px;
    }
    .csv-preview {
      padding: 0 16px 16px;
      overflow: auto;
    }
    .csv-preview table {
      font-size: 11px;
    }
    .action-bar {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      border-top: 1px solid var(--tcip-border);
      background: var(--vscode-editorWidget-background);
      flex-shrink: 0;
    }
    .section-header {
      font-size: 12px;
      font-weight: 600;
      padding: 8px 16px 4px;
      color: var(--vscode-descriptionForeground);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  </style>
</head>
<body>
  <div class="panel-container">
    <div class="toolbar">
      <span style="font-weight:600">Results</span>
      <div style="flex:1"></div>
    </div>
    <div class="content" id="results-content">
      <div id="empty-state" class="empty-state">
        <div class="icon">📋</div>
        <div>No results to display</div>
        <div class="status-text">Use Copilot to see model evaluation results</div>
      </div>
      <div id="results-section" style="display:none">
        <div class="section-header">Overall Metrics</div>
        <div class="metrics-overview" id="overall-metrics"></div>

        <div class="section-header">Per-Class Metrics</div>
        <div class="per-class-section">
          <table id="per-class-table">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>mAP50</th><th>mAP50-95</th><th>Count</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>

        <div class="section-header">Worst Predictions</div>
        <div class="worst-section">
          <div class="thumb-grid" id="worst-grid"></div>
        </div>

        <div class="section-header">CSV Export Preview</div>
        <div class="csv-preview">
          <table id="csv-table">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="action-bar" id="action-bar" style="display:none">
      <button id="btn-accept" class="accent">✓ Accept Model</button>
      <button id="btn-retrain">↻ Retrain</button>
      <button id="btn-export" class="primary">⬇ Export CSV</button>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${resultsJsUri}"></script>
</body>
</html>`;
    }
    handleMessage(msg) {
        switch (msg.type) {
            case "accept_model":
            case "retrain":
            case "export_csv":
                break;
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
exports.ResultsPanel = ResultsPanel;
//# sourceMappingURL=results-panel.js.map