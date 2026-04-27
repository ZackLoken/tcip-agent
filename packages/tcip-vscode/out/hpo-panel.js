"use strict";
/**
 * HPO Panel — WebviewPanel host for hyperparameter optimization.
 * Embeds Ray Tune dashboard and shows trials table.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.HpoPanel = void 0;
const webview_base_1 = require("./webview-base");
class HpoPanel extends webview_base_1.WebviewPanelProvider {
    constructor(extensionUri) {
        super(extensionUri, "tcip-hpo", "TCIP: HPO");
    }
    getHtmlContent(webview, nonce, sharedCssUri, sharedJsUri) {
        const hpoJsUri = (0, webview_base_1.getUri)(webview, this.extensionUri, "media", "hpo.js");
        const cspSource = webview.cspSource;
        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; frame-src http://localhost:* https://localhost:*; img-src ${cspSource} data:; style-src ${cspSource} 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${sharedCssUri}">
  <style nonce="${nonce}">
    .search-space {
      padding: 12px;
    }
    .search-space table {
      max-width: 600px;
    }
    .best-trial {
      margin: 12px;
      padding: 12px;
      background: var(--vscode-editorWidget-background);
      border: 2px solid var(--tcip-accent);
      border-radius: 6px;
    }
    .best-trial .label {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .trials-table-wrapper {
      padding: 0 12px 12px;
      overflow: auto;
    }
    .trial-status {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .trial-status.running::before { content: '⏳'; }
    .trial-status.complete::before { content: '✅'; }
    .trial-status.error::before { content: '❌'; }
    .trial-status.pending::before { content: '⏸'; }
    .best-row {
      background: rgba(var(--vscode-editorInfo-foreground, 75, 156, 211), 0.1);
      border-left: 3px solid var(--tcip-accent);
    }
    .ray-frame {
      width: 100%;
      height: 400px;
      border: none;
      border-top: 1px solid var(--tcip-border);
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
      <span style="font-weight:600">Hyperparameter Optimization</span>
      <div style="flex:1"></div>
      <span class="status-text" id="experiment-id">No active experiment</span>
      <div class="separator" style="width:1px;height:20px;background:var(--tcip-border)"></div>
      <button id="btn-stop-hpo" class="danger">⏹ Stop</button>
    </div>
    <div class="content" id="hpo-content">
      <div id="empty-state" class="empty-state">
        <div class="icon">⚙</div>
        <div>No HPO experiment running</div>
        <div class="status-text">Use Copilot to start a training run with HPO</div>
      </div>
      <div id="live-section" style="display:none">
        <div id="best-trial-section" class="best-trial" style="display:none">
          <div class="label">Best Trial</div>
          <div id="best-trial-info"></div>
        </div>
        <div class="section-header">Search Space</div>
        <div class="search-space">
          <table id="search-space-table">
            <thead><tr><th>Parameter</th><th>Range / Values</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="section-header">Trials</div>
        <div class="trials-table-wrapper">
          <table id="trials-table">
            <thead><tr><th>Trial</th><th>Status</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div id="ray-section" style="display:none">
        <div class="section-header">Ray Dashboard</div>
        <iframe id="ray-frame" class="ray-frame" sandbox="allow-scripts allow-same-origin"></iframe>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${sharedJsUri}"></script>
  <script nonce="${nonce}" src="${hpoJsUri}"></script>
</body>
</html>`;
    }
    handleMessage(msg) {
        switch (msg.type) {
            case "stop_hpo":
            case "select_trial":
                break;
        }
    }
}
exports.HpoPanel = HpoPanel;
//# sourceMappingURL=hpo-panel.js.map